"""
Persistent ReID Gallery Service

Stores ReID features for tracked objects with configurable TTL.
Provides re-identification across video loops and camera restarts.

Features:
- Local LRU cache for fast matching
- Sync with backend (MongoDB or in-memory store)
- Configurable TTL (default 7 days)
- Support for both person (OSNet 512-dim) and vehicle (TransReID 768-dim) features
"""

import asyncio
import logging
import os
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import OrderedDict
import json
import base64
import httpx

logger = logging.getLogger(__name__)

# Configuration
GALLERY_TTL_DAYS = int(os.environ.get("REID_GALLERY_TTL_DAYS", "7"))
GALLERY_MAX_SIZE = int(os.environ.get("REID_GALLERY_MAX_SIZE", "10000"))
GALLERY_MATCH_THRESHOLD_PERSON = float(os.environ.get("REID_MATCH_THRESHOLD_PERSON", "0.55"))
GALLERY_MATCH_THRESHOLD_VEHICLE = float(os.environ.get("REID_MATCH_THRESHOLD_VEHICLE", "0.50"))
GALLERY_SYNC_INTERVAL = int(os.environ.get("REID_GALLERY_SYNC_INTERVAL", "60"))  # seconds
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:3001")

# Recency-based threshold boost: if entry was updated within this window, require higher similarity
# This prevents GID collision when different people appear shortly after someone leaves
GALLERY_RECENCY_WINDOW_SECONDS = int(os.environ.get("REID_RECENCY_WINDOW", "30"))  # seconds
GALLERY_RECENCY_THRESHOLD_BOOST = float(os.environ.get("REID_RECENCY_BOOST", "0.25"))  # add to threshold

# Local file persistence (for development without backend)
GALLERY_LOCAL_FILE = os.environ.get("REID_GALLERY_LOCAL_FILE", "./data/reid_gallery.json")
GALLERY_USE_LOCAL = os.environ.get("REID_GALLERY_USE_LOCAL", "true").lower() in ("true", "1", "yes")


@dataclass
class GalleryEntry:
    """Single entry in the ReID gallery."""
    gid: int
    object_type: str  # 'person' or 'vehicle'
    feature: np.ndarray  # L2-normalized feature vector
    feature_dim: int  # 512 for person, 768 for vehicle
    last_seen: datetime
    first_seen: datetime
    camera_id: Optional[str] = None
    confidence: float = 0.0
    match_count: int = 1  # Number of times this entry was matched

    def is_expired(self, ttl_days: int = GALLERY_TTL_DAYS) -> bool:
        """Check if entry has expired."""
        expiry = self.last_seen + timedelta(days=ttl_days)
        return datetime.now() > expiry

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization."""
        return {
            "gid": self.gid,
            "type": self.object_type,
            "feature": base64.b64encode(self.feature.tobytes()).decode('utf-8'),
            "feature_dim": self.feature_dim,
            "feature_dtype": str(self.feature.dtype),
            "last_seen": self.last_seen.isoformat(),
            "first_seen": self.first_seen.isoformat(),
            "camera_id": self.camera_id,
            "confidence": self.confidence,
            "match_count": self.match_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GalleryEntry":
        """Create from dict."""
        feature_bytes = base64.b64decode(data["feature"])
        dtype = np.dtype(data.get("feature_dtype", "float32"))
        feature = np.frombuffer(feature_bytes, dtype=dtype)

        return cls(
            gid=data["gid"],
            object_type=data["type"],
            feature=feature,
            feature_dim=data["feature_dim"],
            last_seen=datetime.fromisoformat(data["last_seen"]),
            first_seen=datetime.fromisoformat(data["first_seen"]),
            camera_id=data.get("camera_id"),
            confidence=data.get("confidence", 0.0),
            match_count=data.get("match_count", 1),
        )


class ReIDGallery:
    """
    Persistent ReID Gallery for cross-session re-identification.

    Architecture:
    - Local cache (OrderedDict for LRU behavior)
    - Async sync with backend
    - Separate indices for person and vehicle features
    """

    def __init__(
        self,
        max_size: int = GALLERY_MAX_SIZE,
        ttl_days: int = GALLERY_TTL_DAYS,
        person_threshold: float = GALLERY_MATCH_THRESHOLD_PERSON,
        vehicle_threshold: float = GALLERY_MATCH_THRESHOLD_VEHICLE,
    ):
        self.max_size = max_size
        self.ttl_days = ttl_days
        self.person_threshold = person_threshold
        self.vehicle_threshold = vehicle_threshold

        # Separate galleries for each type (different feature dimensions)
        # Key: gid, Value: GalleryEntry
        self._person_gallery: OrderedDict[int, GalleryEntry] = OrderedDict()
        self._vehicle_gallery: OrderedDict[int, GalleryEntry] = OrderedDict()

        # HTTP client for backend sync
        self._http_client: Optional[httpx.AsyncClient] = None

        # Sync state
        self._last_sync = 0
        self._sync_lock = asyncio.Lock()
        self._initialized = False
        self._dirty = False  # Track if gallery has unsaved changes

        # Stats
        self._stats = {
            "matches_found": 0,
            "matches_missed": 0,
            "entries_added": 0,
            "entries_updated": 0,
            "entries_expired": 0,
            "syncs_completed": 0,
        }

        # Track changes since last save (for better logging)
        self._persons_added_since_save = 0
        self._vehicles_added_since_save = 0

        logger.info(
            f"ReID Gallery initialized: max_size={max_size}, ttl={ttl_days}d, "
            f"person_threshold={person_threshold}, vehicle_threshold={vehicle_threshold}"
        )

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=10.0)
        return self._http_client

    async def close(self):
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    async def initialize(self):
        """Initialize gallery from local file and backend."""
        if self._initialized:
            return

        # First, try to load from local file (faster, works offline)
        self._load_from_local()

        try:
            # Then, try to load from backend (may have more/newer entries)
            await self._load_from_backend()
            self._initialized = True
            logger.info(
                f"Gallery initialized: "
                f"{len(self._person_gallery)} persons, {len(self._vehicle_gallery)} vehicles"
            )

            # Start periodic cleanup task
            asyncio.create_task(self._periodic_cleanup())

        except Exception as e:
            logger.warning(f"Failed to initialize gallery from backend: {e}")
            self._initialized = True  # Continue with local data only

        # Save local file with merged data
        self._save_to_local()

    async def _periodic_cleanup(self):
        """Periodically clean up expired entries and sync to backend/local."""
        cleanup_interval = 3600  # 1 hour
        check_interval = 60  # Check every minute for dirty state

        while True:
            try:
                await asyncio.sleep(check_interval)

                now = time.time()

                # Only save if there are unsaved changes
                if self._dirty:
                    self._save_to_local()
                    self._dirty = False

                # Less frequent: cleanup expired entries (every hour)
                if now % cleanup_interval < check_interval:
                    removed = self.cleanup_expired()
                    if removed > 0:
                        logger.info(f"Periodic cleanup removed {removed} expired gallery entries")
                        self._dirty = True  # Mark dirty after cleanup
                    # Sync to backend
                    await self.sync_all_to_backend()

            except asyncio.CancelledError:
                # Final save before shutdown (only if dirty)
                if self._dirty:
                    self._save_to_local()
                break
            except Exception as e:
                logger.error(f"Error in gallery periodic cleanup: {e}")

    async def _load_from_backend(self):
        """Load gallery entries from backend."""
        try:
            client = await self._get_client()

            # Fetch all tracked objects with features
            response = await client.get(
                f"{BACKEND_URL}/api/tracked/gallery",
                params={"withFeatures": "true", "ttlDays": self.ttl_days}
            )

            if response.status_code == 200:
                data = response.json()
                entries = data.get("entries", [])

                for entry_data in entries:
                    try:
                        entry = GalleryEntry.from_dict(entry_data)
                        if entry.object_type == "person":
                            self._person_gallery[entry.gid] = entry
                        else:
                            self._vehicle_gallery[entry.gid] = entry
                    except Exception as e:
                        logger.debug(f"Failed to parse gallery entry: {e}")

                logger.info(f"Loaded {len(entries)} entries from backend gallery")
            elif response.status_code == 404:
                # Endpoint not yet implemented, that's OK
                logger.debug("Gallery endpoint not available, starting fresh")
            else:
                logger.warning(f"Failed to load gallery: {response.status_code}")

        except Exception as e:
            logger.warning(f"Error loading gallery from backend: {e}")

    def _load_from_local(self):
        """Load gallery entries from local JSON file."""
        if not GALLERY_USE_LOCAL:
            return

        try:
            local_path = GALLERY_LOCAL_FILE
            if not os.path.exists(local_path):
                logger.debug(f"Local gallery file not found: {local_path}")
                return

            with open(local_path, 'r') as f:
                data = json.load(f)

            entries = data.get("entries", [])
            loaded = 0
            for entry_data in entries:
                try:
                    entry = GalleryEntry.from_dict(entry_data)
                    if not entry.is_expired(self.ttl_days):
                        if entry.object_type == "person":
                            self._person_gallery[entry.gid] = entry
                        else:
                            self._vehicle_gallery[entry.gid] = entry
                        loaded += 1
                except Exception as e:
                    logger.debug(f"Failed to parse local gallery entry: {e}")

            if loaded > 0:
                logger.info(
                    f"Loaded {loaded} entries from local gallery file: {local_path}"
                )

        except Exception as e:
            logger.warning(f"Error loading gallery from local file: {e}")

    def _save_to_local(self):
        """Save gallery entries to local JSON file."""
        if not GALLERY_USE_LOCAL:
            logger.debug("Gallery local save disabled (GALLERY_USE_LOCAL=false)")
            return

        try:
            local_path = GALLERY_LOCAL_FILE

            # Ensure directory exists
            dir_path = os.path.dirname(local_path) if os.path.dirname(local_path) else "."
            os.makedirs(dir_path, exist_ok=True)

            # Collect all non-expired entries
            entries = []
            person_count = len(self._person_gallery)
            vehicle_count = len(self._vehicle_gallery)

            for gallery in [self._person_gallery, self._vehicle_gallery]:
                for entry in gallery.values():
                    if not entry.is_expired(self.ttl_days):
                        entries.append(entry.to_dict())

            data = {
                "version": 1,
                "saved_at": datetime.now().isoformat(),
                "entries": entries,
            }

            with open(local_path, 'w') as f:
                json.dump(data, f, indent=2)

            # Only log if there were new entries added since last save
            if self._persons_added_since_save > 0 or self._vehicles_added_since_save > 0:
                parts = []
                if self._persons_added_since_save > 0:
                    parts.append(f"{self._persons_added_since_save} person{'s' if self._persons_added_since_save != 1 else ''}")
                if self._vehicles_added_since_save > 0:
                    parts.append(f"{self._vehicles_added_since_save} vehicle{'s' if self._vehicles_added_since_save != 1 else ''}")
                logger.info(f"💾 Gallery saved: added {' and '.join(parts)} (total: {len(entries)} entries)")

                # Reset counters after logging
                self._persons_added_since_save = 0
                self._vehicles_added_since_save = 0

        except Exception as e:
            logger.warning(f"Error saving gallery to local file: {e}")

    def _get_gallery(self, object_type: str) -> OrderedDict[int, GalleryEntry]:
        """Get the appropriate gallery for object type."""
        return self._person_gallery if object_type == "person" else self._vehicle_gallery

    def _get_threshold(self, object_type: str) -> float:
        """Get match threshold for object type."""
        return self.person_threshold if object_type == "person" else self.vehicle_threshold

    @staticmethod
    def cosine_similarity(feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Calculate cosine similarity between two L2-normalized feature vectors."""
        # Features should already be L2-normalized
        return float(np.dot(feat1, feat2))

    def find_match(
        self,
        feature: np.ndarray,
        object_type: str,
        exclude_gids: Optional[List[int]] = None,
        min_threshold: Optional[float] = None,
    ) -> Optional[Tuple[int, float]]:
        """
        Find best matching GID for a feature vector.

        Args:
            feature: L2-normalized feature vector
            object_type: 'person' or 'vehicle'
            exclude_gids: GIDs to exclude from matching (e.g., currently active tracks)
            min_threshold: Override default threshold

        Returns:
            (gid, similarity) if match found, None otherwise
        """
        gallery = self._get_gallery(object_type)
        base_threshold = min_threshold or self._get_threshold(object_type)
        exclude_gids = set(exclude_gids or [])

        if not gallery:
            return None

        best_match = None
        best_similarity = 0.0
        best_effective_threshold = base_threshold
        all_similarities = []  # Debug: track all similarities
        now = datetime.now()
        recency_window = timedelta(seconds=GALLERY_RECENCY_WINDOW_SECONDS)

        for gid, entry in gallery.items():
            if gid in exclude_gids:
                continue

            if entry.is_expired(self.ttl_days):
                continue

            # Verify feature dimensions match
            if len(feature) != entry.feature_dim:
                logger.debug(
                    f"Gallery dim mismatch: query={len(feature)}, entry={entry.feature_dim} for GID {gid}"
                )
                continue

            similarity = self.cosine_similarity(feature, entry.feature)

            # RECENCY BOOST: If entry was updated recently, require higher similarity
            # This prevents GID collision when different people appear shortly after someone leaves
            time_since_update = now - entry.last_seen
            if time_since_update < recency_window:
                # Apply boost proportional to recency (more recent = higher boost)
                recency_factor = 1.0 - (time_since_update.total_seconds() / GALLERY_RECENCY_WINDOW_SECONDS)
                threshold_boost = GALLERY_RECENCY_THRESHOLD_BOOST * recency_factor
                effective_threshold = base_threshold + threshold_boost
            else:
                effective_threshold = base_threshold

            all_similarities.append((gid, similarity, effective_threshold))

            # Check if this candidate beats the best AND exceeds its threshold
            if similarity > effective_threshold and similarity > best_similarity:
                best_similarity = similarity
                best_match = gid
                best_effective_threshold = effective_threshold

        if best_match is not None:
            self._stats["matches_found"] += 1
            logger.info(
                f"Gallery match found: {object_type} GID {best_match} "
                f"(similarity={best_similarity:.3f}, threshold={best_effective_threshold:.3f})"
            )
            return (best_match, best_similarity)

        self._stats["matches_missed"] += 1
        # Debug: Log why no match was found
        if all_similarities:
            top_3 = sorted(all_similarities, key=lambda x: x[1], reverse=True)[:3]
            logger.debug(
                f"Gallery miss for {object_type}: base_threshold={base_threshold}, "
                f"top candidates: {[(gid, f'{sim:.3f} (thr={thr:.3f})') for gid, sim, thr in top_3]}"
            )
        else:
            logger.debug(f"Gallery miss for {object_type}: no valid candidates in gallery")
        return None

    def find_top_matches(
        self,
        feature: np.ndarray,
        object_type: str,
        top_k: int = 5,
        exclude_gids: Optional[List[int]] = None,
    ) -> List[Tuple[int, float]]:
        """
        Find top-K matching GIDs for a feature vector.

        Returns:
            List of (gid, similarity) tuples, sorted by similarity descending
        """
        gallery = self._get_gallery(object_type)
        threshold = self._get_threshold(object_type)
        exclude_gids = set(exclude_gids or [])

        if not gallery:
            return []

        matches = []

        for gid, entry in gallery.items():
            if gid in exclude_gids:
                continue

            if entry.is_expired(self.ttl_days):
                continue

            if len(feature) != entry.feature_dim:
                continue

            similarity = self.cosine_similarity(feature, entry.feature)

            if similarity > threshold:
                matches.append((gid, similarity))

        # Sort by similarity descending
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:top_k]

    def add_or_update(
        self,
        gid: int,
        feature: np.ndarray,
        object_type: str,
        camera_id: Optional[str] = None,
        confidence: float = 0.0,
    ) -> GalleryEntry:
        """
        Add or update a gallery entry.

        Uses EMA to update features if entry already exists.
        """
        gallery = self._get_gallery(object_type)
        now = datetime.now()

        if gid in gallery:
            # Update existing entry with EMA
            entry = gallery[gid]
            alpha = 0.8  # Weight for existing feature

            if len(feature) == entry.feature_dim:
                # EMA update
                entry.feature = alpha * entry.feature + (1 - alpha) * feature
                # Re-normalize
                entry.feature = entry.feature / np.linalg.norm(entry.feature)

            entry.last_seen = now
            entry.match_count += 1
            if camera_id:
                entry.camera_id = camera_id
            if confidence > entry.confidence:
                entry.confidence = confidence

            # Move to end (most recently used)
            gallery.move_to_end(gid)
            self._stats["entries_updated"] += 1
            self._dirty = True  # Mark for save

        else:
            # Create new entry
            entry = GalleryEntry(
                gid=gid,
                object_type=object_type,
                feature=feature.copy(),
                feature_dim=len(feature),
                last_seen=now,
                first_seen=now,
                camera_id=camera_id,
                confidence=confidence,
            )
            gallery[gid] = entry
            self._stats["entries_added"] += 1
            self._dirty = True  # Mark for save

            # Track new entries for save logging
            if object_type == "person":
                self._persons_added_since_save += 1
            else:
                self._vehicles_added_since_save += 1

            logger.info(
                f"📥 Gallery NEW entry: {object_type} GID {gid} "
                f"(dim={len(feature)}, total={len(self._person_gallery) + len(self._vehicle_gallery)})"
            )

            # Evict oldest if over capacity
            while len(gallery) > self.max_size:
                oldest_gid = next(iter(gallery))
                del gallery[oldest_gid]
                logger.debug(f"Evicted oldest gallery entry: {object_type} GID {oldest_gid}")

        return entry

    def remove(self, gid: int, object_type: str) -> bool:
        """Remove an entry from the gallery."""
        gallery = self._get_gallery(object_type)
        if gid in gallery:
            del gallery[gid]
            return True
        return False

    def cleanup_expired(self) -> int:
        """Remove expired entries from gallery."""
        removed = 0

        for gallery in [self._person_gallery, self._vehicle_gallery]:
            expired_gids = [
                gid for gid, entry in gallery.items()
                if entry.is_expired(self.ttl_days)
            ]
            for gid in expired_gids:
                del gallery[gid]
                removed += 1

        if removed > 0:
            self._stats["entries_expired"] += removed
            logger.info(f"Cleaned up {removed} expired gallery entries")

        return removed

    async def sync_to_backend(self, gid: int, object_type: str):
        """Sync a single entry to backend."""
        gallery = self._get_gallery(object_type)
        entry = gallery.get(gid)

        if not entry:
            return

        try:
            client = await self._get_client()

            response = await client.patch(
                f"{BACKEND_URL}/api/tracked/{gid}/features",
                json=entry.to_dict()
            )

            if response.status_code not in (200, 201):
                logger.debug(f"Failed to sync features for GID {gid}: {response.status_code}")

        except Exception as e:
            logger.debug(f"Error syncing features to backend: {e}")

    async def sync_all_to_backend(self):
        """Sync all gallery entries to backend."""
        async with self._sync_lock:
            if time.time() - self._last_sync < GALLERY_SYNC_INTERVAL:
                return

            try:
                client = await self._get_client()

                # Collect all entries
                all_entries = []
                for gallery in [self._person_gallery, self._vehicle_gallery]:
                    for entry in gallery.values():
                        if not entry.is_expired(self.ttl_days):
                            all_entries.append(entry.to_dict())

                if not all_entries:
                    return

                # Batch sync
                response = await client.post(
                    f"{BACKEND_URL}/api/tracked/gallery/sync",
                    json={"entries": all_entries}
                )

                if response.status_code in (200, 201):
                    self._stats["syncs_completed"] += 1
                    self._last_sync = time.time()
                    logger.debug(f"Synced {len(all_entries)} gallery entries to backend")
                else:
                    logger.debug(f"Failed to sync gallery: {response.status_code}")

            except Exception as e:
                logger.debug(f"Error syncing gallery to backend: {e}")

    def clear(self):
        """Clear all gallery entries and reset stats.

        This is a full reset - clears both in-memory galleries and the local file.
        """
        persons_count = len(self._person_gallery)
        vehicles_count = len(self._vehicle_gallery)

        self._person_gallery.clear()
        self._vehicle_gallery.clear()

        # Reset stats
        self._stats = {
            "matches_found": 0,
            "matches_missed": 0,
            "entries_added": 0,
            "entries_updated": 0,
            "entries_expired": 0,
        }

        # Save empty gallery to local file
        self._save_to_local()

        logger.info(f"🗑️ Gallery cleared: {persons_count} persons, {vehicles_count} vehicles")

    def get_stats(self) -> Dict[str, Any]:
        """Get gallery statistics."""
        return {
            "persons": len(self._person_gallery),
            "vehicles": len(self._vehicle_gallery),
            "total": len(self._person_gallery) + len(self._vehicle_gallery),
            "max_size": self.max_size,
            "ttl_days": self.ttl_days,
            **self._stats,
        }

    def get_entry(self, gid: int, object_type: str) -> Optional[GalleryEntry]:
        """Get a specific gallery entry."""
        gallery = self._get_gallery(object_type)
        return gallery.get(gid)

    def has_feature(self, gid: int, object_type: str) -> bool:
        """Check if GID has a stored feature."""
        gallery = self._get_gallery(object_type)
        return gid in gallery


# Global singleton instance
_gallery: Optional[ReIDGallery] = None


def get_reid_gallery() -> ReIDGallery:
    """Get the global ReID gallery instance."""
    global _gallery
    if _gallery is None:
        _gallery = ReIDGallery()
    return _gallery


async def initialize_reid_gallery():
    """Initialize the global ReID gallery."""
    gallery = get_reid_gallery()
    await gallery.initialize()
    return gallery


def reset_reid_gallery():
    """Reset the global ReID gallery (for testing)."""
    global _gallery
    if _gallery:
        _gallery._person_gallery.clear()
        _gallery._vehicle_gallery.clear()
        _gallery._stats = {k: 0 for k in _gallery._stats}
    _gallery = None
