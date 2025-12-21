"""
ReID (Re-Identification) Metadata Storage

Stores metadata (Gemini analysis results) for tracked objects.
Note: Actual tracking is handled by BoT-SORT (see services/detection/bot_sort_tracker.py).
This class only provides metadata storage and retrieval for tracked objects.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class ReIDTracker:
    """
    Metadata storage for tracked objects.

    Note: Despite the name, this class no longer handles tracking.
    Tracking is done by BoT-SORT. This class stores:
    - Gemini analysis results (color, model, armed, etc.)
    - Appearance history for each track
    - First/last seen timestamps
    """

    def __init__(self, max_age: int = 30, n_init: int = 3):
        """
        Initialize ReID metadata storage.

        Args:
            max_age: (Legacy, unused) Maximum frames to keep a track alive
            n_init: (Legacy, unused) Number of frames before a track is confirmed
        """
        self.max_age = max_age
        self.n_init = n_init

        # Metadata storage: {track_id: {type, analysis, first_seen, last_seen, ...}}
        self.tracked_objects: Dict[str, Dict[str, Any]] = {}

        # Track appearance history for cross-camera matching
        self.appearance_history: Dict[str, List[Dict]] = {}

        # Counter for fallback IDs
        self._fallback_vehicle_id = 0
        self._fallback_person_id = 0

        logger.info("ReID metadata storage initialized (tracking by BoT-SORT)")

    def update_vehicles(
        self,
        detections: List[Tuple[List[float], float, str]],
        frame: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Legacy method - assigns sequential IDs for backward compatibility.
        Note: Actual tracking is done by BoT-SORT.
        """
        if not detections:
            return []
        return self._fallback_track(detections, "vehicle")

    def update_persons(
        self,
        detections: List[Tuple[List[float], float, str]],
        frame: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Legacy method - assigns sequential IDs for backward compatibility.
        Note: Actual tracking is done by BoT-SORT.
        """
        if not detections:
            return []
        return self._fallback_track(detections, "person")

    def _fallback_track(
        self,
        detections: List[Tuple],
        obj_type: str
    ) -> List[Dict[str, Any]]:
        """Assign sequential IDs for legacy compatibility"""
        results = []
        for bbox, conf, class_name in detections:
            if obj_type == "vehicle":
                self._fallback_vehicle_id += 1
                track_id = f"v_{self._fallback_vehicle_id}"
            else:
                self._fallback_person_id += 1
                track_id = f"p_{self._fallback_person_id}"

            results.append({
                "track_id": track_id,
                "bbox": bbox if isinstance(bbox, list) else list(bbox),
                "class": class_name,
                "confirmed": True
            })

        return results

    def save_metadata(self, track_id: str, metadata: Dict[str, Any]) -> None:
        """
        Save/update metadata for a tracked object.

        Args:
            track_id: Unique track identifier
            metadata: Analysis data from Gemini (color, model, armed, etc.)
        """
        if track_id not in self.tracked_objects:
            self.tracked_objects[track_id] = {
                "first_seen": datetime.now().isoformat(),
                "last_seen": datetime.now().isoformat()
            }

        self.tracked_objects[track_id].update(metadata)
        self.tracked_objects[track_id]["last_seen"] = datetime.now().isoformat()

        logger.debug(f"Saved metadata for {track_id}: {metadata.keys()}")

    def get_metadata(self, track_id: str) -> Optional[Dict[str, Any]]:
        """
        Get stored metadata for a tracked object.

        Args:
            track_id: Unique track identifier

        Returns:
            Metadata dict or None if not found
        """
        return self.tracked_objects.get(track_id)

    def has_been_analyzed(self, track_id: str) -> bool:
        """Check if object has already been analyzed by Gemini"""
        meta = self.get_metadata(track_id)
        return meta is not None and "analysis" in meta

    def add_appearance(
        self,
        track_id: str,
        camera_id: str,
        bbox: List[float],
        timestamp: Optional[str] = None
    ) -> None:
        """
        Record an appearance of a tracked object.
        Useful for cross-camera tracking and history.
        """
        if track_id not in self.appearance_history:
            self.appearance_history[track_id] = []

        self.appearance_history[track_id].append({
            "camera_id": camera_id,
            "bbox": bbox,
            "timestamp": timestamp or datetime.now().isoformat()
        })

    def get_appearance_history(self, track_id: str) -> List[Dict]:
        """Get all recorded appearances for a tracked object"""
        return self.appearance_history.get(track_id, [])

    def get_all_tracked(self, obj_type: Optional[str] = None) -> Dict[str, Dict]:
        """
        Get all tracked objects with their metadata.

        Args:
            obj_type: Filter by type ("vehicle", "person") or None for all

        Returns:
            Dict of track_id -> metadata
        """
        if obj_type is None:
            return self.tracked_objects.copy()

        prefix = "v_" if obj_type == "vehicle" else "p_"
        return {
            k: v for k, v in self.tracked_objects.items()
            if k.startswith(prefix)
        }

    def get_armed_persons(self) -> List[str]:
        """Get track IDs of all persons marked as armed"""
        armed = []
        for track_id, meta in self.tracked_objects.items():
            if track_id.startswith("p_"):
                analysis = meta.get("analysis", {})
                if analysis.get("armed"):
                    armed.append(track_id)
        return armed

    def cleanup_old_tracks(self, max_age_seconds: int = 3600) -> int:
        """
        Remove tracks that haven't been seen for a while.

        Args:
            max_age_seconds: Maximum age in seconds before removal

        Returns:
            Number of tracks removed
        """
        now = datetime.now()
        to_remove = []

        for track_id, meta in self.tracked_objects.items():
            last_seen = meta.get("last_seen")
            if last_seen:
                try:
                    last_seen_dt = datetime.fromisoformat(last_seen)
                    if (now - last_seen_dt).total_seconds() > max_age_seconds:
                        to_remove.append(track_id)
                except ValueError:
                    pass

        for track_id in to_remove:
            del self.tracked_objects[track_id]
            if track_id in self.appearance_history:
                del self.appearance_history[track_id]

        if to_remove:
            logger.debug(f"Cleaned up {len(to_remove)} old tracks")

        return len(to_remove)

    def reset(self) -> None:
        """Reset and clear all stored metadata"""
        self.tracked_objects.clear()
        self.appearance_history.clear()
        self._fallback_vehicle_id = 0
        self._fallback_person_id = 0

        logger.debug("ReID metadata storage reset")

    def get_stats(self) -> Dict[str, Any]:
        """Get tracker statistics"""
        vehicles = len([k for k in self.tracked_objects if k.startswith("v_")])
        persons = len([k for k in self.tracked_objects if k.startswith("p_")])
        armed = len(self.get_armed_persons())

        return {
            "total_tracked": len(self.tracked_objects),
            "vehicles": vehicles,
            "persons": persons,
            "armed_persons": armed
        }
