"""Backend Sync - Syncs tracked objects metadata to Node.js backend.

This module provides functions to sync tracked person/vehicle metadata
from the AI service to the backend MongoDB database for persistence.

The backend stores:
- Global IDs (GIDs) assigned by the ReID system
- Gemini analysis results (clothing, weapons, license plates, etc.)
- Appearance history (which cameras, when)
- Armed status and threat levels
"""

import asyncio
import logging
import os
import re
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
import httpx

logger = logging.getLogger(__name__)


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
) -> bool:
    """Sync a tracked object to the backend for persistence.

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

    Returns:
        True if sync successful, False otherwise
    """
    try:
        client = await get_http_client()

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

        response = await client.post(
            f"{BACKEND_URL}/api/tracked",
            json=payload
        )

        if response.status_code in (200, 201):
            logger.debug(f"Synced tracked object GID {gid} ({obj_type})")
            return True
        else:
            logger.warning(f"Failed to sync GID {gid}: {response.status_code} - {response.text}")
            return False

    except Exception as e:
        logger.error(f"Error syncing tracked object GID {gid}: {e}")
        return False


async def add_appearance(
    gid: int,
    camera_id: str,
    camera_name: str = None,
    bbox: List[float] = None,
    confidence: float = None,
    local_track_id: int = None,
    snapshot_url: str = None,
) -> bool:
    """Add an appearance record to a tracked object.

    Args:
        gid: Global ID of the tracked object
        camera_id: Camera where appearance was detected
        camera_name: Human-readable camera name
        bbox: Bounding box [x1, y1, x2, y2]
        confidence: Detection confidence 0-1
        local_track_id: Camera-local track ID
        snapshot_url: URL to snapshot image

    Returns:
        True if successful, False otherwise
    """
    try:
        client = await get_http_client()

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

        response = await client.post(
            f"{BACKEND_URL}/api/tracked/{gid}/appearance",
            json=payload
        )

        if response.status_code in (200, 201):
            logger.debug(f"Added appearance for GID {gid} on {camera_id}")
            return True
        else:
            logger.warning(f"Failed to add appearance for GID {gid}: {response.status_code}")
            return False

    except Exception as e:
        logger.error(f"Error adding appearance for GID {gid}: {e}")
        return False


async def update_analysis(
    gid: Union[int, str],
    analysis: Dict[str, Any],
) -> bool:
    """Update Gemini analysis results for a tracked object.

    Args:
        gid: Global ID of the tracked object (can be int or string like 't_2', 'v_5')
        analysis: Analysis results dict with fields like:
            - clothing, clothingColor, gender, ageRange (persons)
            - color, make, model, licensePlate, vehicleType (vehicles)
            - armed, weaponType, suspicious, suspiciousReason
            - description, confidence

    Returns:
        True if successful, False otherwise
    """
    # Parse GID from track_id format
    parsed_gid = parse_gid(gid)
    if parsed_gid is None:
        logger.warning(f"Could not parse GID from: {gid}")
        return False

    try:
        client = await get_http_client()

        response = await client.patch(
            f"{BACKEND_URL}/api/tracked/{parsed_gid}/analysis",
            json=analysis
        )

        if response.status_code in (200, 201):
            armed = analysis.get("armed", False)
            logger.info(f"Updated analysis for GID {parsed_gid}" + (" [ARMED]" if armed else ""))
            return True
        else:
            logger.warning(f"Failed to update analysis for GID {parsed_gid}: {response.status_code}")
            return False

    except Exception as e:
        logger.error(f"Error updating analysis for GID {parsed_gid}: {e}")
        return False


async def mark_armed(
    gid: Union[int, str],
    weapon_type: str = None,
    confidence: float = None,
) -> bool:
    """Mark a tracked person as armed.

    Args:
        gid: Global ID of the tracked person (can be int or string like 't_2', 'p_5')
        weapon_type: Type of weapon detected
        confidence: Detection confidence

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

    return await update_analysis(gid, analysis)


async def sync_from_detection_result(
    camera_id: str,
    camera_name: str,
    tracked_persons: List[Dict],
    tracked_vehicles: List[Dict],
    new_persons: List[Dict],
    new_vehicles: List[Dict],
    armed_persons: List[Dict],
) -> Dict[str, int]:
    """Sync detection results to backend.

    This is a convenience function to sync all tracked objects from a
    DetectionResult. It handles:
    - Creating/updating tracked objects
    - Adding appearance records for existing tracks
    - Syncing Gemini analysis results

    Args:
        camera_id: Source camera ID
        camera_name: Human-readable camera name
        tracked_persons: List of currently tracked persons
        tracked_vehicles: List of currently tracked vehicles
        new_persons: List of newly detected persons
        new_vehicles: List of newly detected vehicles
        armed_persons: List of armed persons detected

    Returns:
        Dict with sync statistics
    """
    stats = {
        "synced": 0,
        "appearances": 0,
        "armed": 0,
        "errors": 0,
    }

    # Sync new persons (create records)
    for person in new_persons:
        try:
            track_id = person.get("track_id", person.get("id"))
            if track_id is None:
                continue

            metadata = person.get("metadata", {})
            analysis = metadata.get("analysis", {})

            gid = parse_gid(track_id)
            if gid is None:
                logger.warning(f"Could not parse GID from track_id: {track_id}")
                stats["errors"] += 1
                continue

            success = await sync_tracked_object(
                gid=gid,
                obj_type="person",
                camera_id=camera_id,
                camera_name=camera_name,
                bbox=person.get("bbox", person.get("box")),
                confidence=person.get("confidence", person.get("conf")),
                is_armed=analysis.get("armed", False),
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

    # Sync new vehicles (create records)
    for vehicle in new_vehicles:
        try:
            track_id = vehicle.get("track_id", vehicle.get("id"))
            if track_id is None:
                continue

            metadata = vehicle.get("metadata", {})
            analysis = metadata.get("analysis", {})
            class_name = vehicle.get("class", vehicle.get("label", "vehicle"))

            gid = parse_gid(track_id)
            if gid is None:
                logger.warning(f"Could not parse GID from track_id: {track_id}")
                stats["errors"] += 1
                continue

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

    # Update armed status for armed persons
    for armed in armed_persons:
        try:
            track_id = armed.get("track_id", armed.get("id"))
            if track_id is None:
                continue

            metadata = armed.get("metadata", {})
            analysis = metadata.get("analysis", {})

            gid = parse_gid(track_id)
            if gid is None:
                logger.warning(f"Could not parse GID from track_id: {track_id}")
                stats["errors"] += 1
                continue

            success = await mark_armed(
                gid=gid,
                weapon_type=analysis.get("weaponType", analysis.get("סוג_נשק")),
                confidence=analysis.get("armedConfidence"),
            )

            if success:
                stats["armed"] += 1
            else:
                stats["errors"] += 1

        except Exception as e:
            logger.error(f"Error syncing armed person: {e}")
            stats["errors"] += 1

    return stats


# Rate limiting for appearance updates (avoid flooding the backend)
_last_appearance_sync: Dict[str, float] = {}  # gid:camera -> timestamp
APPEARANCE_SYNC_INTERVAL = 5.0  # Minimum seconds between appearance syncs


async def sync_appearance_throttled(
    gid: int,
    camera_id: str,
    camera_name: str = None,
    bbox: List[float] = None,
    confidence: float = None,
) -> bool:
    """Add appearance with rate limiting.

    Only syncs if the last sync for this gid:camera was > APPEARANCE_SYNC_INTERVAL seconds ago.

    Returns:
        True if synced, False if throttled or error
    """
    import time

    key = f"{gid}:{camera_id}"
    now = time.time()

    last_sync = _last_appearance_sync.get(key, 0)
    if now - last_sync < APPEARANCE_SYNC_INTERVAL:
        return False  # Throttled

    _last_appearance_sync[key] = now

    return await add_appearance(
        gid=gid,
        camera_id=camera_id,
        camera_name=camera_name,
        bbox=bbox,
        confidence=confidence,
    )
