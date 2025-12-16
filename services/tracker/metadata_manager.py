"""
Metadata Manager - Centralized metadata storage and persistence for tracks.

This manager handles:
- Persistent metadata storage beyond track lifetime
- Metadata export/import for persistence across resets
- Cleanup of expired metadata
- Query interface for metadata access
"""

import time
import json
import logging
from typing import Dict, List, Optional
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)


class MetadataManager:
    """Manager for persistent track metadata storage."""

    def __init__(
        self,
        ttl: int = 3600,  # Default TTL: 1 hour after track disappears
        persistence_path: Optional[str] = None,
    ):
        """Initialize metadata manager.

        Args:
            ttl: Time-to-live for metadata after track expires (seconds)
            persistence_path: Path for metadata persistence file
        """
        self.ttl = ttl
        self.persistence_path = persistence_path or "output/track_metadata.json"

        # Track metadata storage: track_id -> metadata
        self._metadata: Dict[int, dict] = {}

        # Track last seen time for cleanup
        self._last_seen: Dict[int, float] = {}

        # Class-based indexing for fast queries
        self._class_index: Dict[int, set] = defaultdict(set)

        # Gemini analysis tracking (track_id -> analysis_count)
        self._gemini_analysis_count: Dict[int, int] = defaultdict(int)

        # Load persisted metadata if exists
        self._load_metadata()

    def update_track_metadata(self, track_id: int, class_id: int, metadata: dict):
        """Update metadata for a track.

        Args:
            track_id: Track ID
            class_id: Class ID for indexing
            metadata: Metadata dictionary from Track object
        """
        # Deep copy to avoid reference issues
        self._metadata[track_id] = json.loads(
            json.dumps(
                metadata, default=lambda x: list(x) if isinstance(x, set) else str(x)
            )
        )

        # Update last seen time
        self._last_seen[track_id] = time.time()

        # Update class index
        self._class_index[class_id].add(track_id)

        # Log periodically
        if track_id % 10 == 1:  # Log for track IDs 1, 11, 21, etc.
            logger.info(
                f"MetadataManager: Updated track {track_id} (class={class_id}), total tracks: {len(self._metadata)}"
            )

    def get_track_metadata(self, track_id: int) -> Optional[dict]:
        """Get metadata for a specific track.

        Args:
            track_id: Track ID

        Returns:
            Metadata dictionary or None if not found
        """
        return self._metadata.get(track_id)

    def get_tracks_by_class(self, class_id: int) -> List[dict]:
        """Get all tracks for a specific class.

        Args:
            class_id: Class ID

        Returns:
            List of metadata dictionaries
        """
        track_ids = self._class_index.get(class_id, set())
        return [
            {"track_id": tid, "metadata": self._metadata.get(tid)}
            for tid in track_ids
            if tid in self._metadata
        ]

    def get_all_metadata(self) -> Dict[int, dict]:
        """Get all stored metadata.

        Returns:
            Dictionary of track_id -> metadata
        """
        return self._metadata.copy()

    def search_metadata(
        self,
        tag: Optional[str] = None,
        alert_type: Optional[str] = None,
        zone: Optional[str] = None,
        behavior: Optional[str] = None,
    ) -> List[dict]:
        """Search metadata by various criteria.

        Args:
            tag: Filter by tag
            alert_type: Filter by alert type
            zone: Filter by visited zone
            behavior: Filter by behavior type

        Returns:
            List of matching tracks with metadata
        """
        results = []

        for track_id, metadata in self._metadata.items():
            # Check tag filter
            if tag and tag not in metadata.get("tags", []):
                continue

            # Check alert filter
            if alert_type:
                alerts = metadata.get("alerts", [])
                if not any(a.get("type") == alert_type for a in alerts):
                    continue

            # Check zone filter
            if zone:
                zones = metadata.get("zones_visited", [])
                if isinstance(zones, set):
                    zones = list(zones)
                if zone not in zones:
                    continue

            # Check behavior filter
            if behavior:
                track_behavior = metadata.get("behavior", {})
                if not track_behavior or track_behavior.get("type") != behavior:
                    continue

            results.append({"track_id": track_id, "metadata": metadata})

        return results

    def cleanup_expired(self) -> int:
        """Remove metadata for tracks that haven't been seen in TTL period.

        Returns:
            Number of tracks cleaned up
        """
        current_time = time.time()
        expired_tracks = []

        for track_id, last_seen in self._last_seen.items():
            if current_time - last_seen > self.ttl:
                expired_tracks.append(track_id)

        # Remove expired tracks
        for track_id in expired_tracks:
            if track_id in self._metadata:
                # Remove from class index
                for class_tracks in self._class_index.values():
                    class_tracks.discard(track_id)

                # Remove metadata
                del self._metadata[track_id]
                del self._last_seen[track_id]

        if expired_tracks:
            logger.info(f"Cleaned up metadata for {len(expired_tracks)} expired tracks")

        return len(expired_tracks)

    def save_metadata(self, path: Optional[str] = None):
        """Save metadata to file for persistence.

        Args:
            path: Path to save to (uses default if None)
        """
        save_path = path or self.persistence_path

        try:
            # Prepare data for serialization
            data = {
                "metadata": self._metadata,
                "last_seen": self._last_seen,
                "class_index": {k: list(v) for k, v in self._class_index.items()},
                "saved_at": time.time(),
            }

            # Ensure directory exists
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)

            # Save to file
            with open(save_path, "w") as f:
                json.dump(data, f, indent=2, default=str)

            logger.info(
                f"Saved metadata for {len(self._metadata)} tracks to {save_path}"
            )

        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    def _load_metadata(self):
        """Load metadata from persistence file."""
        if not Path(self.persistence_path).exists():
            logger.info("No persisted metadata found")
            return

        try:
            with open(self.persistence_path, "r") as f:
                data = json.load(f)

            # Restore metadata
            self._metadata = {int(k): v for k, v in data.get("metadata", {}).items()}

            # Restore last seen times
            self._last_seen = {int(k): v for k, v in data.get("last_seen", {}).items()}

            # Restore class index
            raw_class_index = data.get("class_index", {})
            self._class_index = defaultdict(
                set, {int(k): set(v) for k, v in raw_class_index.items()}
            )

            saved_at = data.get("saved_at", 0)
            age_hours = (time.time() - saved_at) / 3600

            logger.info(
                f"Loaded metadata for {len(self._metadata)} tracks "
                f"(saved {age_hours:.1f} hours ago)"
            )

        except Exception as e:
            logger.error(f"Failed to load persisted metadata: {e}")

    def clear_all(self):
        """Clear all metadata (use with caution)."""
        count = len(self._metadata)
        self._metadata.clear()
        self._last_seen.clear()
        self._class_index.clear()
        logger.warning(f"Cleared all metadata ({count} tracks)")

    def should_analyze_with_gemini(self, track_id: int, max_analyses: int = 2) -> bool:
        """Check if track should be analyzed with Gemini.

        Args:
            track_id: Track ID
            max_analyses: Maximum number of analyses per track

        Returns:
            True if track should be analyzed
        """
        return self._gemini_analysis_count[track_id] < max_analyses

    def record_gemini_analysis(self, track_id: int):
        """Record that a Gemini analysis was performed for a track.

        Args:
            track_id: Track ID
        """
        self._gemini_analysis_count[track_id] += 1
        logger.debug(
            f"Gemini analysis count for track {track_id}: {self._gemini_analysis_count[track_id]}"
        )

    def get_gemini_analysis_count(self, track_id: int) -> int:
        """Get number of Gemini analyses performed for a track.

        Args:
            track_id: Track ID

        Returns:
            Analysis count
        """
        return self._gemini_analysis_count[track_id]

    def get_stats(self) -> dict:
        """Get metadata statistics.

        Returns:
            Dictionary with statistics
        """
        total_tracks = len(self._metadata)
        total_notes = sum(len(m.get("notes", [])) for m in self._metadata.values())
        total_alerts = sum(len(m.get("alerts", [])) for m in self._metadata.values())

        # Tracks by class
        tracks_by_class = {
            class_id: len(tracks) for class_id, tracks in self._class_index.items()
        }

        # Recently active tracks (last 5 minutes)
        recent_threshold = time.time() - 300
        recent_tracks = sum(
            1 for t in self._last_seen.values() if t >= recent_threshold
        )

        return {
            "total_tracks": total_tracks,
            "total_notes": total_notes,
            "total_alerts": total_alerts,
            "tracks_by_class": tracks_by_class,
            "recent_tracks": recent_tracks,
            "oldest_track_age": (
                time.time() - min(self._last_seen.values()) if self._last_seen else 0
            ),
        }


# Global singleton instance
_metadata_manager: Optional[MetadataManager] = None


def get_metadata_manager() -> MetadataManager:
    """Get or create global metadata manager instance.

    Returns:
        MetadataManager instance
    """
    global _metadata_manager
    if _metadata_manager is None:
        _metadata_manager = MetadataManager()
    return _metadata_manager


def reset_metadata_manager():
    """Reset global metadata manager (for testing)."""
    global _metadata_manager
    if _metadata_manager is not None:
        _metadata_manager.save_metadata()  # Save before reset
    _metadata_manager = None
