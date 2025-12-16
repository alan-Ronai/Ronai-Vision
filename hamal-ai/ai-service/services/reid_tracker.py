"""
ReID (Re-Identification) Tracker Service

Uses DeepSort for persistent object tracking across frames.
Each detected object gets a unique ID that persists even if it
temporarily leaves and re-enters the frame.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Try to import deep_sort_realtime
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    DEEPSORT_AVAILABLE = True
except ImportError:
    DEEPSORT_AVAILABLE = False
    logger.warning("deep-sort-realtime not installed. ReID tracking disabled.")


class ReIDTracker:
    """
    Manages persistent tracking of vehicles and persons across video frames.

    Uses separate DeepSort trackers for:
    - Vehicles (cars, trucks, buses, motorcycles)
    - Persons

    Stores metadata (Gemini analysis results) per track ID.
    """

    def __init__(self, max_age: int = 30, n_init: int = 3):
        """
        Initialize ReID trackers.

        Args:
            max_age: Maximum frames to keep a track alive without detections
            n_init: Number of frames before a track is confirmed
        """
        self.max_age = max_age
        self.n_init = n_init

        if DEEPSORT_AVAILABLE:
            # Separate trackers for vehicles and persons
            self.vehicle_tracker = DeepSort(
                max_age=max_age,
                n_init=n_init,
                max_iou_distance=0.7,
                max_cosine_distance=0.3,
                nn_budget=100
            )
            self.person_tracker = DeepSort(
                max_age=max_age,
                n_init=n_init,
                max_iou_distance=0.7,
                max_cosine_distance=0.2,
                nn_budget=100
            )
            logger.info("✅ ReID trackers initialized (DeepSort)")
        else:
            self.vehicle_tracker = None
            self.person_tracker = None
            logger.warning("⚠️ ReID tracking disabled - DeepSort not available")

        # Metadata storage: {track_id: {type, analysis, first_seen, last_seen, ...}}
        self.tracked_objects: Dict[str, Dict[str, Any]] = {}

        # Track appearance history for cross-camera matching
        self.appearance_history: Dict[str, List[Dict]] = {}

        # Counter for fallback IDs when DeepSort unavailable
        self._fallback_vehicle_id = 0
        self._fallback_person_id = 0

    def update_vehicles(
        self,
        detections: List[Tuple[List[float], float, str]],
        frame: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Update vehicle tracker with new detections.

        Args:
            detections: List of (bbox, confidence, class_name) tuples
                        bbox format: [x1, y1, x2, y2]
            frame: Current frame (numpy array, BGR format)

        Returns:
            List of tracked objects with persistent IDs:
            [{"track_id": str, "bbox": [x1,y1,x2,y2], "class": str, "confirmed": bool}]
        """
        if not detections:
            return []

        if self.vehicle_tracker is None:
            # Fallback: assign sequential IDs (no persistence)
            return self._fallback_track(detections, "vehicle")

        # Format detections for DeepSort: [[x1, y1, x2, y2, conf, class], ...]
        formatted_dets = []
        for bbox, conf, class_name in detections:
            x1, y1, x2, y2 = bbox
            # DeepSort expects [left, top, width, height]
            formatted_dets.append(([x1, y1, x2-x1, y2-y1], conf, class_name))

        # Update tracker
        tracks = self.vehicle_tracker.update_tracks(formatted_dets, frame=frame)

        # Convert to output format
        results = []
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = f"v_{track.track_id}"
            ltrb = track.to_ltrb()  # [left, top, right, bottom]

            results.append({
                "track_id": track_id,
                "bbox": [float(x) for x in ltrb],
                "class": track.det_class if hasattr(track, 'det_class') else "vehicle",
                "confirmed": True
            })

            # Update last seen
            if track_id in self.tracked_objects:
                self.tracked_objects[track_id]["last_seen"] = datetime.now().isoformat()

        return results

    def update_persons(
        self,
        detections: List[Tuple[List[float], float, str]],
        frame: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Update person tracker with new detections.

        Args:
            detections: List of (bbox, confidence, class_name) tuples
            frame: Current frame (numpy array, BGR format)

        Returns:
            List of tracked persons with persistent IDs
        """
        if not detections:
            return []

        if self.person_tracker is None:
            return self._fallback_track(detections, "person")

        # Format detections for DeepSort
        formatted_dets = []
        for bbox, conf, class_name in detections:
            x1, y1, x2, y2 = bbox
            formatted_dets.append(([x1, y1, x2-x1, y2-y1], conf, class_name))

        # Update tracker
        tracks = self.person_tracker.update_tracks(formatted_dets, frame=frame)

        # Convert to output format
        results = []
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = f"p_{track.track_id}"
            ltrb = track.to_ltrb()

            results.append({
                "track_id": track_id,
                "bbox": [float(x) for x in ltrb],
                "class": "person",
                "confirmed": True
            })

            # Update last seen
            if track_id in self.tracked_objects:
                self.tracked_objects[track_id]["last_seen"] = datetime.now().isoformat()

        return results

    def _fallback_track(
        self,
        detections: List[Tuple],
        obj_type: str
    ) -> List[Dict[str, Any]]:
        """Fallback tracking when DeepSort unavailable - just assign sequential IDs"""
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
            logger.info(f"Cleaned up {len(to_remove)} old tracks")

        return len(to_remove)

    def reset(self) -> None:
        """Reset all trackers and clear stored data"""
        if DEEPSORT_AVAILABLE:
            self.vehicle_tracker = DeepSort(
                max_age=self.max_age,
                n_init=self.n_init,
                max_iou_distance=0.7,
                max_cosine_distance=0.3
            )
            self.person_tracker = DeepSort(
                max_age=self.max_age,
                n_init=self.n_init,
                max_iou_distance=0.7,
                max_cosine_distance=0.2
            )

        self.tracked_objects.clear()
        self.appearance_history.clear()
        self._fallback_vehicle_id = 0
        self._fallback_person_id = 0

        logger.info("ReID trackers reset")

    def get_stats(self) -> Dict[str, Any]:
        """Get tracker statistics"""
        vehicles = len([k for k in self.tracked_objects if k.startswith("v_")])
        persons = len([k for k in self.tracked_objects if k.startswith("p_")])
        armed = len(self.get_armed_persons())

        return {
            "total_tracked": len(self.tracked_objects),
            "vehicles": vehicles,
            "persons": persons,
            "armed_persons": armed,
            "deepsort_available": DEEPSORT_AVAILABLE
        }
