"""ReID (Re-Identification) Tracker for persistent object tracking.

Uses DeepSort to maintain consistent IDs for objects across frames.
This prevents counting the same person/vehicle multiple times.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

# Try to import deep_sort_realtime
DEEPSORT_AVAILABLE = False
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    DEEPSORT_AVAILABLE = True
except ImportError:
    logger.warning(
        "deep-sort-realtime not installed. Install with: pip install deep-sort-realtime"
    )


@dataclass
class TrackedObject:
    """Represents a tracked object with metadata."""
    track_id: int
    object_type: str  # 'person' or 'vehicle'
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    analyzed: bool = False  # Whether Gemini analysis was done

    def to_dict(self) -> Dict:
        return {
            "track_id": self.track_id,
            "object_type": self.object_type,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "metadata": self.metadata,
            "analyzed": self.analyzed
        }


class ReIDTracker:
    """ReID tracker for persistent object identification.

    Maintains separate trackers for vehicles and persons.
    Each detected object gets a unique ID that persists across frames.
    """

    def __init__(
        self,
        max_age: int = 30,  # Frames to keep track alive without detection
        n_init: int = 3,    # Detections needed to confirm track
        max_iou_distance: float = 0.7
    ):
        """Initialize ReID tracker.

        Args:
            max_age: Maximum frames to keep track without detection
            n_init: Minimum detections to confirm a track
            max_iou_distance: Maximum IOU distance for matching
        """
        self.max_age = max_age
        self.n_init = n_init
        self.max_iou_distance = max_iou_distance

        if not DEEPSORT_AVAILABLE:
            logger.warning(
                "DeepSort not available. Using fallback tracking (less accurate)."
            )
            self.vehicle_tracker = None
            self.person_tracker = None
        else:
            # Separate trackers for different object types
            self.vehicle_tracker = DeepSort(
                max_age=max_age,
                n_init=n_init,
                max_iou_distance=max_iou_distance
            )
            self.person_tracker = DeepSort(
                max_age=max_age,
                n_init=n_init,
                max_iou_distance=max_iou_distance
            )

        # Store metadata for each track
        self._vehicle_metadata: Dict[int, TrackedObject] = {}
        self._person_metadata: Dict[int, TrackedObject] = {}

        # Fallback counters when DeepSort unavailable
        self._fallback_vehicle_id = 0
        self._fallback_person_id = 0

        # Statistics
        self._stats = {
            "total_vehicles_tracked": 0,
            "total_persons_tracked": 0,
            "active_vehicles": 0,
            "active_persons": 0
        }

        logger.info(f"ReIDTracker initialized (DeepSort: {DEEPSORT_AVAILABLE})")

    def update_vehicles(
        self,
        detections: List[Tuple[List[float], float, str]],
        frame: Optional[np.ndarray] = None
    ) -> List[TrackedObject]:
        """Update vehicle tracker with new detections.

        Args:
            detections: List of (bbox, confidence, class_name) tuples
                       bbox format: [x1, y1, x2, y2]
            frame: Optional frame for appearance features

        Returns:
            List of TrackedObject for confirmed tracks
        """
        if not detections:
            return list(self._vehicle_metadata.values())

        if not DEEPSORT_AVAILABLE or self.vehicle_tracker is None:
            return self._fallback_update_vehicles(detections)

        # Convert to DeepSort format: [[x1, y1, w, h, conf], ...]
        ds_detections = []
        for bbox, conf, class_name in detections:
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            ds_detections.append(([x1, y1, w, h], conf, class_name))

        # Update tracker
        tracks = self.vehicle_tracker.update_tracks(ds_detections, frame=frame)

        # Process confirmed tracks
        result = []
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            bbox = track.to_ltrb().tolist()  # [x1, y1, x2, y2]

            # Get or create metadata
            if track_id not in self._vehicle_metadata:
                self._vehicle_metadata[track_id] = TrackedObject(
                    track_id=track_id,
                    object_type="vehicle",
                    bbox=bbox,
                    confidence=track.det_conf if track.det_conf else 0.0
                )
                self._stats["total_vehicles_tracked"] += 1
                logger.info(f"New vehicle track: {track_id}")
            else:
                # Update existing
                obj = self._vehicle_metadata[track_id]
                obj.bbox = bbox
                obj.last_seen = datetime.now()
                if track.det_conf:
                    obj.confidence = track.det_conf

            result.append(self._vehicle_metadata[track_id])

        self._stats["active_vehicles"] = len(result)
        return result

    def _fallback_update_vehicles(
        self,
        detections: List[Tuple[List[float], float, str]]
    ) -> List[TrackedObject]:
        """Fallback tracking when DeepSort is unavailable."""
        result = []
        for bbox, conf, class_name in detections:
            self._fallback_vehicle_id += 1
            track_id = self._fallback_vehicle_id

            obj = TrackedObject(
                track_id=track_id,
                object_type="vehicle",
                bbox=list(bbox),
                confidence=conf
            )
            self._vehicle_metadata[track_id] = obj
            self._stats["total_vehicles_tracked"] += 1
            result.append(obj)

        self._stats["active_vehicles"] = len(result)
        return result

    def update_persons(
        self,
        detections: List[Tuple[List[float], float, str]],
        frame: Optional[np.ndarray] = None
    ) -> List[TrackedObject]:
        """Update person tracker with new detections.

        Args:
            detections: List of (bbox, confidence, class_name) tuples
            frame: Optional frame for appearance features

        Returns:
            List of TrackedObject for confirmed tracks
        """
        if not detections:
            return list(self._person_metadata.values())

        if not DEEPSORT_AVAILABLE or self.person_tracker is None:
            return self._fallback_update_persons(detections)

        # Convert to DeepSort format
        ds_detections = []
        for bbox, conf, class_name in detections:
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            ds_detections.append(([x1, y1, w, h], conf, class_name))

        # Update tracker
        tracks = self.person_tracker.update_tracks(ds_detections, frame=frame)

        # Process confirmed tracks
        result = []
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            bbox = track.to_ltrb().tolist()

            if track_id not in self._person_metadata:
                self._person_metadata[track_id] = TrackedObject(
                    track_id=track_id,
                    object_type="person",
                    bbox=bbox,
                    confidence=track.det_conf if track.det_conf else 0.0
                )
                self._stats["total_persons_tracked"] += 1
                logger.info(f"New person track: {track_id}")
            else:
                obj = self._person_metadata[track_id]
                obj.bbox = bbox
                obj.last_seen = datetime.now()
                if track.det_conf:
                    obj.confidence = track.det_conf

            result.append(self._person_metadata[track_id])

        self._stats["active_persons"] = len(result)
        return result

    def _fallback_update_persons(
        self,
        detections: List[Tuple[List[float], float, str]]
    ) -> List[TrackedObject]:
        """Fallback tracking when DeepSort is unavailable."""
        result = []
        for bbox, conf, class_name in detections:
            self._fallback_person_id += 1
            track_id = self._fallback_person_id

            obj = TrackedObject(
                track_id=track_id,
                object_type="person",
                bbox=list(bbox),
                confidence=conf
            )
            self._person_metadata[track_id] = obj
            self._stats["total_persons_tracked"] += 1
            result.append(obj)

        self._stats["active_persons"] = len(result)
        return result

    def save_metadata(self, track_id: int, object_type: str, metadata: Dict[str, Any]):
        """Save analysis metadata for a tracked object.

        Args:
            track_id: Track ID
            object_type: 'person' or 'vehicle'
            metadata: Gemini analysis results or other metadata
        """
        storage = self._person_metadata if object_type == "person" else self._vehicle_metadata

        if track_id in storage:
            storage[track_id].metadata.update(metadata)
            storage[track_id].analyzed = True
            logger.debug(f"Saved metadata for {object_type} {track_id}")

    def get_metadata(self, track_id: int, object_type: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a tracked object.

        Args:
            track_id: Track ID
            object_type: 'person' or 'vehicle'

        Returns:
            Metadata dict or None if not found
        """
        storage = self._person_metadata if object_type == "person" else self._vehicle_metadata

        if track_id in storage:
            return storage[track_id].metadata
        return None

    def is_analyzed(self, track_id: int, object_type: str) -> bool:
        """Check if object has been analyzed by Gemini.

        Args:
            track_id: Track ID
            object_type: 'person' or 'vehicle'

        Returns:
            True if already analyzed
        """
        storage = self._person_metadata if object_type == "person" else self._vehicle_metadata

        if track_id in storage:
            return storage[track_id].analyzed
        return False

    def get_object(self, track_id: int, object_type: str) -> Optional[TrackedObject]:
        """Get tracked object by ID.

        Args:
            track_id: Track ID
            object_type: 'person' or 'vehicle'

        Returns:
            TrackedObject or None
        """
        storage = self._person_metadata if object_type == "person" else self._vehicle_metadata
        return storage.get(track_id)

    def get_all_persons(self) -> List[TrackedObject]:
        """Get all tracked persons."""
        return list(self._person_metadata.values())

    def get_all_vehicles(self) -> List[TrackedObject]:
        """Get all tracked vehicles."""
        return list(self._vehicle_metadata.values())

    def get_armed_persons(self) -> List[TrackedObject]:
        """Get all persons marked as armed."""
        return [
            obj for obj in self._person_metadata.values()
            if obj.metadata.get("armed", False) or obj.metadata.get("חמוש", False)
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get tracking statistics."""
        stats = self._stats.copy()
        stats["deepsort_available"] = DEEPSORT_AVAILABLE
        return stats

    def reset(self):
        """Reset all trackers and metadata."""
        if DEEPSORT_AVAILABLE:
            self.vehicle_tracker = DeepSort(
                max_age=self.max_age,
                n_init=self.n_init,
                max_iou_distance=self.max_iou_distance
            )
            self.person_tracker = DeepSort(
                max_age=self.max_age,
                n_init=self.n_init,
                max_iou_distance=self.max_iou_distance
            )

        self._vehicle_metadata.clear()
        self._person_metadata.clear()
        self._fallback_vehicle_id = 0
        self._fallback_person_id = 0
        self._stats = {
            "total_vehicles_tracked": 0,
            "total_persons_tracked": 0,
            "active_vehicles": 0,
            "active_persons": 0
        }
        logger.info("ReIDTracker reset")


# Global singleton
_tracker: Optional[ReIDTracker] = None


def get_reid_tracker() -> Optional[ReIDTracker]:
    """Get or create global ReID tracker instance."""
    global _tracker

    if _tracker is None:
        try:
            _tracker = ReIDTracker()
        except Exception as e:
            logger.error(f"Failed to initialize ReID tracker: {e}")
            return None

    return _tracker


def reset_reid_tracker():
    """Reset global ReID tracker."""
    global _tracker
    if _tracker:
        _tracker.reset()
    _tracker = None
