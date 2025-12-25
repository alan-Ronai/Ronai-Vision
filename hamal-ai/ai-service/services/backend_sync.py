"""Backend Sync - Syncs tracked objects metadata to Node.js backend.

This module provides functions to sync tracked person/vehicle metadata
from the AI service to the backend MongoDB database for persistence.

The backend stores:
- Global IDs (GIDs) assigned by the ReID system
- Gemini analysis results (clothing, weapons, license plates, etc.)
- Appearance history (which cameras, when)
- Armed status and threat levels

Architecture Notes:
- All sync operations use retry logic with exponential backoff
- Rate-limiting uses thread-safe locks to prevent race conditions
- TTL-based cleanup prevents memory leaks from old entries
"""

import asyncio
import logging
import os
import re
import time
import threading
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from collections import OrderedDict
import httpx

logger = logging.getLogger(__name__)


# =============================================================================
# THREAD-SAFE RATE LIMITER WITH TTL
# =============================================================================

class RateLimiter:
    """Thread-safe rate limiter with TTL-based cleanup.

    Prevents duplicate operations within a time window and automatically
    cleans up old entries to prevent memory leaks.
    """

    def __init__(
        self,
        interval_seconds: float,
        max_entries: int = 5000,
        ttl_seconds: float = 3600.0,  # 1 hour default TTL
        name: str = "RateLimiter"
    ):
        self.interval = interval_seconds
        self.max_entries = max_entries
        self.ttl = ttl_seconds
        self.name = name
        self._entries: OrderedDict[str, float] = OrderedDict()
        self._lock = threading.Lock()
        self._last_cleanup = time.time()
        self._cleanup_interval = 300.0  # Cleanup every 5 minutes

        # Statistics tracking
        self._stats = {
            "allowed": 0,           # Operations that were allowed
            "throttled": 0,         # Operations that were rate-limited
            "evictions": 0,         # LRU evictions due to max_entries
            "expired_cleanups": 0,  # Entries removed due to TTL expiration
            "total_cleanups": 0,    # Number of cleanup cycles
        }

    def should_allow(self, key: str) -> bool:
        """Check if operation should be allowed (not rate-limited).

        Returns True if allowed, False if should be throttled.
        Thread-safe.
        """
        now = time.time()

        with self._lock:
            # Periodic cleanup
            if now - self._last_cleanup > self._cleanup_interval:
                self._cleanup_expired(now)

            last_time = self._entries.get(key, 0)
            if now - last_time < self.interval:
                self._stats["throttled"] += 1
                return False  # Throttled

            # Update timestamp and move to end (most recently used)
            self._entries[key] = now
            self._entries.move_to_end(key)
            self._stats["allowed"] += 1

            # Evict oldest if over capacity
            while len(self._entries) > self.max_entries:
                oldest_key, _ = self._entries.popitem(last=False)
                self._stats["evictions"] += 1
                logger.debug(f"{self.name}: Evicted oldest entry {oldest_key}")

            return True

    def _cleanup_expired(self, now: float):
        """Remove entries older than TTL. Must be called with lock held."""
        expired = []
        for key, timestamp in self._entries.items():
            if now - timestamp > self.ttl:
                expired.append(key)
            else:
                break  # OrderedDict is ordered by insertion, so we can stop early

        for key in expired:
            del self._entries[key]

        if expired:
            self._stats["expired_cleanups"] += len(expired)
            logger.debug(f"{self.name}: Cleaned up {len(expired)} expired entries")

        self._stats["total_cleanups"] += 1
        self._last_cleanup = now

    def clear(self):
        """Clear all entries. Thread-safe."""
        with self._lock:
            self._entries.clear()

    def size(self) -> int:
        """Get current number of entries. Thread-safe."""
        with self._lock:
            return len(self._entries)

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics. Thread-safe."""
        with self._lock:
            total = self._stats["allowed"] + self._stats["throttled"]
            throttle_rate = (self._stats["throttled"] / total * 100) if total > 0 else 0
            return {
                "name": self.name,
                "current_entries": len(self._entries),
                "max_entries": self.max_entries,
                "interval_seconds": self.interval,
                "ttl_seconds": self.ttl,
                "allowed": self._stats["allowed"],
                "throttled": self._stats["throttled"],
                "throttle_rate": f"{throttle_rate:.1f}%",
                "evictions": self._stats["evictions"],
                "expired_cleanups": self._stats["expired_cleanups"],
                "total_cleanups": self._stats["total_cleanups"],
            }


# =============================================================================
# RETRY DECORATOR FOR NETWORK OPERATIONS
# =============================================================================

# Global retry statistics (thread-safe)
_retry_stats = {
    "total_calls": 0,
    "successful_first_try": 0,
    "successful_after_retry": 0,
    "failed_all_retries": 0,
    "total_retries": 0,
}
_retry_stats_lock = threading.Lock()


async def retry_async(
    func,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    exceptions: tuple = (httpx.ConnectError, httpx.TimeoutException, httpx.ReadError),
):
    """Retry an async function with exponential backoff.

    Args:
        func: Async function to call (should be a coroutine)
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries
        max_delay: Maximum delay between retries
        exceptions: Tuple of exceptions to retry on

    Returns:
        Result of the function call

    Raises:
        Last exception if all retries fail
    """
    global _retry_stats
    last_exception = None

    with _retry_stats_lock:
        _retry_stats["total_calls"] += 1

    for attempt in range(max_retries):
        try:
            result = await func()
            # Track success
            with _retry_stats_lock:
                if attempt == 0:
                    _retry_stats["successful_first_try"] += 1
                else:
                    _retry_stats["successful_after_retry"] += 1
            return result
        except exceptions as e:
            last_exception = e
            with _retry_stats_lock:
                _retry_stats["total_retries"] += 1
            if attempt < max_retries - 1:
                delay = min(base_delay * (2 ** attempt), max_delay)
                logger.warning(
                    f"Retry {attempt + 1}/{max_retries} after {delay:.1f}s: {e}"
                )
                await asyncio.sleep(delay)
            else:
                logger.error(f"All {max_retries} retries failed: {e}")

    # Track failure
    with _retry_stats_lock:
        _retry_stats["failed_all_retries"] += 1

    raise last_exception


def get_retry_stats() -> Dict[str, Any]:
    """Get retry statistics."""
    with _retry_stats_lock:
        total = _retry_stats["total_calls"]
        success_rate = 0
        if total > 0:
            successes = _retry_stats["successful_first_try"] + _retry_stats["successful_after_retry"]
            success_rate = (successes / total) * 100
        return {
            **_retry_stats.copy(),
            "success_rate": f"{success_rate:.1f}%",
        }


# =============================================================================
# SYNC OPERATION STATISTICS
# =============================================================================

# Sync operation counters (thread-safe)
_sync_counts = {
    "objects_synced": 0,
    "objects_failed": 0,
    "appearances_synced": 0,
    "appearances_failed": 0,
    "analysis_synced": 0,
    "analysis_failed": 0,
    "armed_synced": 0,
    "armed_failed": 0,
}
_sync_counts_lock = threading.Lock()


def _track_sync_success(operation: str):
    """Track successful sync operation."""
    with _sync_counts_lock:
        key = f"{operation}_synced"
        _sync_counts[key] = _sync_counts.get(key, 0) + 1


def _track_sync_failure(operation: str):
    """Track failed sync operation."""
    with _sync_counts_lock:
        key = f"{operation}_failed"
        _sync_counts[key] = _sync_counts.get(key, 0) + 1


def parse_gid(track_id: Union[str, int]) -> Optional[int]:
    """Extract numeric GID from track_id.

    Track IDs can come in various formats:
    - Integer: 123
    - New session format: 'v_0_2', 'p_1_5' (prefix_session_id) - extracts LAST number
    - Old prefixed string: 't_123', 'v_45', 'p_67'
    - Plain string: '123'

    Args:
        track_id: The track ID to parse

    Returns:
        Integer GID, or None if cannot be parsed
    """
    if track_id is None:
        return None

    # Already an integer
    if isinstance(track_id, int):
        return track_id

    # Convert to string and extract numeric part
    track_str = str(track_id)

    # NEW FORMAT: Try to extract from session format 'v_0_2', 'p_1_5' (get the LAST number)
    match = re.search(r'[tvp]_\d+_(\d+)', track_str)
    if match:
        return int(match.group(1))

    # OLD FORMAT: Try to extract number from prefixed format like 't_123', 'v_45', 'p_67'
    match = re.search(r'[tvp]_(\d+)', track_str)
    if match:
        return int(match.group(1))

    # Try plain integer string
    try:
        return int(track_str)
    except ValueError:
        pass

    # Try to find any number in the string
    match = re.search(r'(\d+)', track_str)
    if match:
        return int(match.group(1))

    logger.warning(f"Could not parse GID from track_id: {track_id}")
    return None

# Backend URL from environment
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:3000")

# Global HTTP client (reused for connection pooling)
_http_client: Optional[httpx.AsyncClient] = None


async def get_http_client() -> httpx.AsyncClient:
    """Get or create the HTTP client."""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(timeout=10.0)
    return _http_client


async def close_http_client():
    """Close the HTTP client on shutdown."""
    global _http_client
    if _http_client and not _http_client.is_closed:
        await _http_client.aclose()
        _http_client = None


async def sync_tracked_object(
    gid: int,
    obj_type: str,
    camera_id: str = None,
    camera_name: str = None,
    bbox: List[float] = None,
    confidence: float = None,
    is_armed: bool = False,
    threat_level: str = "none",
    analysis: Dict[str, Any] = None,
    metadata: Dict[str, Any] = None,
    class_name: str = None,
    max_retries: int = 3,
) -> bool:
    """Sync a tracked object to the backend for persistence.

    Uses retry logic with exponential backoff for network resilience.

    Args:
        gid: Global ID assigned by ReID system
        obj_type: 'person' or 'vehicle'
        camera_id: Camera where object was detected
        camera_name: Human-readable camera name
        bbox: Bounding box [x1, y1, x2, y2]
        confidence: Detection confidence 0-1
        is_armed: Whether person is armed (persons only)
        threat_level: none, low, medium, high, critical
        analysis: Gemini analysis results
        metadata: Additional metadata
        class_name: YOLO class name (person, car, truck, etc.)
        max_retries: Maximum retry attempts on network failure

    Returns:
        True if sync successful, False otherwise
    """
    payload = {
        "gid": gid,
        "type": obj_type,
        "lastSeen": datetime.now().isoformat(),
    }

    if camera_id:
        payload["cameraId"] = camera_id
    if camera_name:
        payload["cameraName"] = camera_name
    if bbox:
        payload["bbox"] = bbox
    if confidence is not None:
        payload["confidence"] = confidence
    if class_name:
        payload["class"] = class_name
    if is_armed:
        payload["isArmed"] = True
    if threat_level and threat_level != "none":
        payload["threatLevel"] = threat_level
    if analysis:
        payload["analysis"] = analysis
    if metadata:
        payload["metadata"] = metadata

    async def do_request():
        client = await get_http_client()
        response = await client.post(
            f"{BACKEND_URL}/api/tracked",
            json=payload
        )
        return response

    try:
        response = await retry_async(do_request, max_retries=max_retries)

        if response.status_code in (200, 201):
            logger.debug(f"Synced tracked object GID {gid} ({obj_type})")
            _track_sync_success("objects")
            return True
        else:
            logger.warning(f"Failed to sync GID {gid}: {response.status_code} - {response.text}")
            _track_sync_failure("objects")
            return False

    except Exception as e:
        logger.error(f"Error syncing tracked object GID {gid} after {max_retries} retries: {e}")
        _track_sync_failure("objects")
        return False


async def add_appearance(
    gid: int,
    camera_id: str,
    camera_name: str = None,
    bbox: List[float] = None,
    confidence: float = None,
    local_track_id: int = None,
    snapshot_url: str = None,
    max_retries: int = 3,
) -> bool:
    """Add an appearance record to a tracked object.

    Uses retry logic with exponential backoff for network resilience.

    Args:
        gid: Global ID of the tracked object
        camera_id: Camera where appearance was detected
        camera_name: Human-readable camera name
        bbox: Bounding box [x1, y1, x2, y2]
        confidence: Detection confidence 0-1
        local_track_id: Camera-local track ID
        snapshot_url: URL to snapshot image
        max_retries: Maximum retry attempts on network failure

    Returns:
        True if successful, False otherwise
    """
    payload = {
        "cameraId": camera_id,
        "timestamp": datetime.now().isoformat(),
    }

    if camera_name:
        payload["cameraName"] = camera_name
    if bbox:
        payload["bbox"] = bbox
    if confidence is not None:
        payload["confidence"] = confidence
    if local_track_id is not None:
        payload["localTrackId"] = local_track_id
    if snapshot_url:
        payload["snapshotUrl"] = snapshot_url

    async def do_request():
        client = await get_http_client()
        return await client.post(
            f"{BACKEND_URL}/api/tracked/{gid}/appearance",
            json=payload
        )

    try:
        response = await retry_async(do_request, max_retries=max_retries)

        if response.status_code in (200, 201):
            logger.debug(f"Added appearance for GID {gid} on {camera_id}")
            _track_sync_success("appearances")
            return True
        else:
            logger.warning(f"Failed to add appearance for GID {gid}: {response.status_code}")
            _track_sync_failure("appearances")
            return False

    except Exception as e:
        logger.error(f"Error adding appearance for GID {gid} after {max_retries} retries: {e}")
        _track_sync_failure("appearances")
        return False


async def update_analysis(
    gid: Union[int, str],
    analysis: Dict[str, Any],
    max_retries: int = 3,
) -> bool:
    """Update Gemini analysis results for a tracked object.

    Uses retry logic with exponential backoff for network resilience.

    Args:
        gid: Global ID of the tracked object (can be int or string like 't_2', 'v_5')
        analysis: Analysis results dict with fields like:
            - clothing, clothingColor, gender, ageRange (persons)
            - color, make, model, licensePlate, vehicleType (vehicles)
            - armed, weaponType, suspicious, suspiciousReason
            - description, confidence
        max_retries: Maximum retry attempts on network failure

    Returns:
        True if successful, False otherwise
    """
    # Parse GID from track_id format
    parsed_gid = parse_gid(gid)
    if parsed_gid is None:
        logger.warning(f"Could not parse GID from: {gid}")
        return False

    async def do_request():
        client = await get_http_client()
        return await client.patch(
            f"{BACKEND_URL}/api/tracked/{parsed_gid}/analysis",
            json=analysis
        )

    try:
        response = await retry_async(do_request, max_retries=max_retries)

        if response.status_code in (200, 201):
            armed = analysis.get("armed", False)
            logger.info(f"Updated analysis for GID {parsed_gid}" + (" [ARMED]" if armed else ""))
            _track_sync_success("analysis")
            if armed:
                _track_sync_success("armed")
            return True
        else:
            logger.warning(f"Failed to update analysis for GID {parsed_gid}: {response.status_code}")
            _track_sync_failure("analysis")
            return False

    except Exception as e:
        logger.error(f"Error updating analysis for GID {parsed_gid} after {max_retries} retries: {e}")
        _track_sync_failure("analysis")
        return False


async def mark_armed(
    gid: Union[int, str],
    weapon_type: str = None,
    confidence: float = None,
    max_retries: int = 3,
) -> bool:
    """Mark a tracked person as armed.

    Uses retry logic with exponential backoff for network resilience.

    Args:
        gid: Global ID of the tracked person (can be int or string like 't_2', 'p_5')
        weapon_type: Type of weapon detected
        confidence: Detection confidence
        max_retries: Maximum retry attempts on network failure

    Returns:
        True if successful, False otherwise
    """
    analysis = {
        "armed": True,
    }
    if weapon_type:
        analysis["weaponType"] = weapon_type
    if confidence is not None:
        analysis["armedConfidence"] = confidence

    return await update_analysis(gid, analysis, max_retries=max_retries)


async def sync_new_objects(
    camera_id: str,
    camera_name: str,
    new_persons: List[Dict],
    new_vehicles: List[Dict],
) -> Dict[str, int]:
    """Sync ONLY newly detected objects to backend (event-driven, thread-safe).

    This is called when new tracks are created by the tracker.
    It does NOT sync:
    - Already tracked objects (no need to re-sync)
    - Armed status (handled by Gemini analysis callback)
    - Analysis updates (handled by Gemini analysis callback)

    Uses RateLimiter for thread-safe deduplication with automatic TTL cleanup.

    Args:
        camera_id: Source camera ID
        camera_name: Human-readable camera name
        new_persons: List of newly detected persons (from tracker)
        new_vehicles: List of newly detected vehicles (from tracker)

    Returns:
        Dict with sync statistics
    """
    stats = {
        "synced": 0,
        "skipped": 0,
        "errors": 0,
    }

    # Sync new persons (create records) - only if not already synced
    for person in new_persons:
        try:
            track_id = person.get("track_id", person.get("id"))
            if track_id is None:
                continue

            gid = parse_gid(track_id)
            if gid is None:
                logger.warning(f"Could not parse GID from track_id: {track_id}")
                stats["errors"] += 1
                continue

            # Skip if already synced (thread-safe check with TTL cleanup)
            sync_key = f"person:{gid}"
            if not _new_objects_limiter.should_allow(sync_key):
                stats["skipped"] += 1
                continue

            metadata = person.get("metadata", {})
            analysis = metadata.get("analysis", {})

            success = await sync_tracked_object(
                gid=gid,
                obj_type="person",
                camera_id=camera_id,
                camera_name=camera_name,
                bbox=person.get("bbox", person.get("box")),
                confidence=person.get("confidence", person.get("conf")),
                is_armed=False,  # Armed status synced separately by Gemini callback
                analysis=analysis if analysis else None,
                class_name="person",
            )

            if success:
                stats["synced"] += 1
            else:
                stats["errors"] += 1

        except Exception as e:
            logger.error(f"Error syncing new person: {e}")
            stats["errors"] += 1

    # Sync new vehicles (create records) - only if not already synced
    for vehicle in new_vehicles:
        try:
            track_id = vehicle.get("track_id", vehicle.get("id"))
            if track_id is None:
                continue

            gid = parse_gid(track_id)
            if gid is None:
                logger.warning(f"Could not parse GID from track_id: {track_id}")
                stats["errors"] += 1
                continue

            # Skip if already synced (thread-safe check with TTL cleanup)
            sync_key = f"vehicle:{gid}"
            if not _new_objects_limiter.should_allow(sync_key):
                stats["skipped"] += 1
                continue

            metadata = vehicle.get("metadata", {})
            analysis = metadata.get("analysis", {})
            class_name = vehicle.get("class", vehicle.get("label", "vehicle"))

            success = await sync_tracked_object(
                gid=gid,
                obj_type="vehicle",
                camera_id=camera_id,
                camera_name=camera_name,
                bbox=vehicle.get("bbox", vehicle.get("box")),
                confidence=vehicle.get("confidence", vehicle.get("conf")),
                analysis=analysis if analysis else None,
                class_name=class_name,
            )

            if success:
                stats["synced"] += 1
            else:
                stats["errors"] += 1

        except Exception as e:
            logger.error(f"Error syncing new vehicle: {e}")
            stats["errors"] += 1

    return stats


# Keep old function for backwards compatibility but mark as deprecated
async def sync_from_detection_result(
    camera_id: str,
    camera_name: str,
    tracked_persons: List[Dict],  # UNUSED - kept for backwards compatibility
    tracked_vehicles: List[Dict],  # UNUSED - kept for backwards compatibility
    new_persons: List[Dict],
    new_vehicles: List[Dict],
    armed_persons: List[Dict],  # IGNORED - armed sync happens in Gemini callback
) -> Dict[str, int]:
    """DEPRECATED: Use sync_new_objects() instead.

    This function now just delegates to sync_new_objects and ignores
    the redundant parameters (tracked_*, armed_persons).
    """
    # Log deprecation warning once
    if not hasattr(sync_from_detection_result, '_warned'):
        logger.warning(
            "sync_from_detection_result is deprecated. "
            "Use sync_new_objects() for new objects, "
            "update_analysis() for Gemini results, "
            "mark_armed() for armed status."
        )
        sync_from_detection_result._warned = True

    # Delegate to new function (ignoring redundant params)
    stats = await sync_new_objects(
        camera_id=camera_id,
        camera_name=camera_name,
        new_persons=new_persons,
        new_vehicles=new_vehicles,
    )

    # Return with backwards-compatible keys
    return {
        "synced": stats["synced"],
        "appearances": 0,  # No longer tracked here
        "armed": 0,  # Now handled by Gemini callback
        "errors": stats["errors"],
    }


# =============================================================================
# RATE LIMITERS (Thread-safe with TTL cleanup)
# =============================================================================

# Rate limiting for appearance updates (avoid flooding the backend)
_appearance_limiter = RateLimiter(
    interval_seconds=5.0,
    max_entries=5000,
    ttl_seconds=3600.0,  # 1 hour
    name="AppearanceSync"
)

# Rate limiting for analysis updates (avoid repeated updates for same track)
_analysis_limiter = RateLimiter(
    interval_seconds=2.0,
    max_entries=5000,
    ttl_seconds=3600.0,
    name="AnalysisSync"
)

# Rate limiting for armed status updates
_armed_limiter = RateLimiter(
    interval_seconds=10.0,
    max_entries=5000,
    ttl_seconds=3600.0,
    name="ArmedSync"
)

# Track synced new objects (thread-safe)
_new_objects_limiter = RateLimiter(
    interval_seconds=0.0,  # No time-based limiting, just deduplication
    max_entries=5000,
    ttl_seconds=3600.0,
    name="NewObjectsSync"
)

async def sync_appearance_throttled(
    gid: int,
    camera_id: str,
    camera_name: str = None,
    bbox: List[float] = None,
    confidence: float = None,
) -> bool:
    """Add appearance with rate limiting (thread-safe with TTL cleanup).

    Returns:
        True if synced, False if throttled or error
    """
    key = f"{gid}:{camera_id}"

    if not _appearance_limiter.should_allow(key):
        return False  # Throttled

    return await add_appearance(
        gid=gid,
        camera_id=camera_id,
        camera_name=camera_name,
        bbox=bbox,
        confidence=confidence,
    )


async def mark_armed_throttled(
    gid: Union[int, str],
    weapon_type: str = None,
    confidence: float = None,
) -> bool:
    """Mark a tracked person as armed with rate limiting (thread-safe with TTL cleanup).

    Returns:
        True if synced, False if throttled or error
    """
    parsed_gid = parse_gid(gid)
    if parsed_gid is None:
        return False

    key = str(parsed_gid)

    if not _armed_limiter.should_allow(key):
        return False  # Throttled

    return await mark_armed(
        gid=gid,
        weapon_type=weapon_type,
        confidence=confidence,
    )


async def update_analysis_throttled(
    gid: Union[int, str],
    analysis: Dict[str, Any],
    force: bool = False,
) -> bool:
    """Update Gemini analysis results with rate limiting (thread-safe with TTL cleanup).

    Args:
        gid: Global ID of the tracked object
        analysis: Analysis results to sync
        force: If True, bypass throttling (e.g., for armed detection)

    Returns:
        True if synced, False if throttled or error
    """
    parsed_gid = parse_gid(gid)
    if parsed_gid is None:
        return False

    # Important updates bypass throttling
    has_important_update = analysis.get("armed") or analysis.get("_typeCorrection")

    if not force and not has_important_update:
        key = str(parsed_gid)
        if not _analysis_limiter.should_allow(key):
            return False  # Throttled

    return await update_analysis(gid=gid, analysis=analysis)


# =============================================================================
# STALE ENTRY CLEANUP
# =============================================================================

async def get_all_tracked() -> List[Dict[str, Any]]:
    """Get all tracked entries from the backend.

    Returns:
        List of tracked entry dicts with gid, type, analysis, etc.
    """
    async def do_request():
        client = await get_http_client()
        return await client.get(f"{BACKEND_URL}/api/tracked")

    try:
        response = await retry_async(do_request, max_retries=2)
        if response.status_code == 200:
            data = response.json()
            # Handle different response formats
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                # Try common nested formats: {data: [...]}, {items: [...]}, {tracked: [...]}
                for key in ['data', 'items', 'tracked', 'objects']:
                    if key in data and isinstance(data[key], list):
                        return data[key]
                # If it's a single object dict, wrap in list
                if 'gid' in data:
                    return [data]
                logger.warning(f"Unexpected tracked response format: {list(data.keys())}")
                return []
            else:
                logger.warning(f"Unexpected tracked response type: {type(data)}")
                return []
        else:
            logger.warning(f"Failed to get tracked entries: {response.status_code}")
            return []
    except Exception as e:
        logger.error(f"Error getting tracked entries: {e}")
        return []


async def delete_tracked(gid: int) -> bool:
    """Delete a tracked entry from the backend.

    Args:
        gid: Global ID to delete

    Returns:
        True if deleted successfully
    """
    async def do_request():
        client = await get_http_client()
        return await client.delete(f"{BACKEND_URL}/api/tracked/{gid}")

    try:
        response = await retry_async(do_request, max_retries=2)
        if response.status_code in (200, 204):
            logger.debug(f"Deleted stale tracked entry GID {gid}")
            return True
        elif response.status_code == 404:
            # Already deleted
            return True
        else:
            logger.warning(f"Failed to delete GID {gid}: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Error deleting GID {gid}: {e}")
        return False


async def cleanup_stale_entries(
    active_gids: set,
    analyzed_gids: set,
    max_age_seconds: float = 60.0,
) -> Dict[str, int]:
    """Clean up stale GID entries that were never properly analyzed.

    Removes entries that:
    - Have no cutout image (analysis incomplete)
    - Are NOT currently active in the scene
    - Were NOT previously analyzed
    - Are older than max_age_seconds

    Args:
        active_gids: Set of GIDs currently active in tracker
        analyzed_gids: Set of GIDs that have been analyzed by Gemini
        max_age_seconds: Minimum age before considering for cleanup

    Returns:
        Dict with cleanup statistics
    """
    stats = {"checked": 0, "deleted": 0, "kept_active": 0, "kept_analyzed": 0, "errors": 0}

    try:
        all_tracked = await get_all_tracked()
        stats["checked"] = len(all_tracked)

        # Debug: log what we got
        if all_tracked and len(all_tracked) > 0:
            sample = all_tracked[0]
            if isinstance(sample, dict):
                logger.debug(f"Cleanup: found {len(all_tracked)} entries, sample keys: {list(sample.keys())[:5]}")
            else:
                logger.warning(f"Cleanup: unexpected entry type: {type(sample)}, value: {str(sample)[:100]}")

        now = datetime.now()

        for entry in all_tracked:
            # Skip non-dict entries (malformed data)
            if not isinstance(entry, dict):
                logger.debug(f"Skipping non-dict entry in tracked: {type(entry)}")
                continue

            gid = entry.get("gid")
            if gid is None:
                continue

            # Keep if currently active in scene
            if gid in active_gids:
                stats["kept_active"] += 1
                continue

            # Keep if already analyzed (has cutout or in analyzed set)
            analysis = entry.get("analysis") or {}
            has_cutout = bool(analysis.get("cutout_image"))
            if has_cutout or gid in analyzed_gids:
                stats["kept_analyzed"] += 1
                continue

            # Check age - only delete if old enough
            last_seen = entry.get("lastSeen") or entry.get("createdAt")
            if last_seen:
                try:
                    # Parse ISO format timestamp
                    if isinstance(last_seen, str):
                        # Handle various ISO formats
                        last_seen = last_seen.replace("Z", "+00:00")
                        from datetime import timezone
                        entry_time = datetime.fromisoformat(last_seen.split("+")[0])
                        age_seconds = (now - entry_time).total_seconds()
                        if age_seconds < max_age_seconds:
                            continue  # Too recent, might still get analyzed
                except Exception:
                    pass  # If we can't parse time, proceed with deletion

            # Delete stale entry
            if await delete_tracked(gid):
                stats["deleted"] += 1
                logger.info(f"🗑️ Cleaned up stale GID {gid} (no cutout, not active)")
            else:
                stats["errors"] += 1

    except Exception as e:
        logger.error(f"Error during stale entry cleanup: {e}")
        stats["errors"] += 1

    if stats["deleted"] > 0:
        logger.info(
            f"🧹 Stale cleanup complete: deleted={stats['deleted']}, "
            f"kept_active={stats['kept_active']}, kept_analyzed={stats['kept_analyzed']}"
        )

    return stats


# =============================================================================
# GLOBAL SYNC STATISTICS AGGREGATOR
# =============================================================================

def get_sync_stats() -> Dict[str, Any]:
    """Get comprehensive backend sync statistics.

    Returns all stats for rate limiters, retry logic, and sync operations.
    """
    with _sync_counts_lock:
        sync_operations = _sync_counts.copy()

    # Calculate success rates
    total_synced = sum(v for k, v in sync_operations.items() if k.endswith("_synced"))
    total_failed = sum(v for k, v in sync_operations.items() if k.endswith("_failed"))
    total_ops = total_synced + total_failed
    success_rate = (total_synced / total_ops * 100) if total_ops > 0 else 100.0

    return {
        "sync_operations": {
            **sync_operations,
            "total_synced": total_synced,
            "total_failed": total_failed,
            "success_rate": f"{success_rate:.1f}%",
        },
        "retry": get_retry_stats(),
        "rate_limiters": {
            "appearance": _appearance_limiter.get_stats(),
            "analysis": _analysis_limiter.get_stats(),
            "armed": _armed_limiter.get_stats(),
            "new_objects": _new_objects_limiter.get_stats(),
        },
        "backend_url": BACKEND_URL,
    }
