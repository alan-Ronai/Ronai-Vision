"""Stable object tracker with Kalman prediction.

Key features:
1. Kalman filter predicts position when YOLO misses detections
2. Track state persists across reconnects (global singleton)
3. Higher miss tolerance before removing tracks
4. Smooth bbox tracking even with intermittent detections

This solves:
- Same objects detected as "new" repeatedly
- Bounding boxes flickering on/off
- Too many false events
"""

import time
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class KalmanBoxTracker:
    """Kalman filter for tracking bounding boxes.

    Simplified 4-state model for stability: [x_center, y_center, width, height]
    Uses exponential smoothing for velocity estimation instead of full Kalman.
    """

    def __init__(self, bbox: Tuple[int, int, int, int]):
        """Initialize Kalman filter with initial bbox."""
        # Convert bbox to state
        x1, y1, x2, y2 = bbox
        self.cx = float((x1 + x2) / 2)
        self.cy = float((y1 + y2) / 2)
        self.w = float(max(10, x2 - x1))
        self.h = float(max(10, y2 - y1))

        # Velocity estimates (exponential moving average)
        self.vx = 0.0
        self.vy = 0.0

        # Smoothing factor for velocity
        self.alpha = 0.3

        # Previous position for velocity calculation
        self._prev_cx = self.cx
        self._prev_cy = self.cy

        self._time_since_update = 0

    def predict(self, dt: float = 1/15) -> Tuple[int, int, int, int]:
        """Predict next state using constant velocity model.

        Args:
            dt: Time delta in seconds

        Returns:
            Predicted bbox (x1, y1, x2, y2)
        """
        # Clamp dt to reasonable range
        dt = max(0.01, min(dt, 0.5))

        # Predict position using velocity
        self.cx += self.vx * dt * 15  # Scale by expected fps
        self.cy += self.vy * dt * 15

        self._time_since_update += 1

        return self._state_to_bbox()

    def update(self, bbox: Tuple[int, int, int, int]):
        """Update state with measurement.

        Args:
            bbox: Measured bbox (x1, y1, x2, y2)
        """
        x1, y1, x2, y2 = bbox

        new_cx = float((x1 + x2) / 2)
        new_cy = float((y1 + y2) / 2)
        new_w = float(max(10, x2 - x1))
        new_h = float(max(10, y2 - y1))

        # Calculate velocity from position change
        if self._time_since_update > 0:
            # Exponential moving average of velocity
            measured_vx = new_cx - self._prev_cx
            measured_vy = new_cy - self._prev_cy

            self.vx = self.alpha * measured_vx + (1 - self.alpha) * self.vx
            self.vy = self.alpha * measured_vy + (1 - self.alpha) * self.vy

        # Update state
        self._prev_cx = self.cx
        self._prev_cy = self.cy

        self.cx = new_cx
        self.cy = new_cy
        self.w = new_w
        self.h = new_h

        self._time_since_update = 0

    def _state_to_bbox(self) -> Tuple[int, int, int, int]:
        """Convert state to bbox."""
        # Ensure valid values
        cx = max(0, self.cx)
        cy = max(0, self.cy)
        w = max(10, self.w)
        h = max(10, self.h)

        return (
            int(cx - w/2),
            int(cy - h/2),
            int(cx + w/2),
            int(cy + h/2)
        )

    def get_bbox(self) -> Tuple[int, int, int, int]:
        """Get current bbox estimate."""
        return self._state_to_bbox()

    @property
    def time_since_update(self) -> int:
        """Frames since last measurement update."""
        return self._time_since_update


@dataclass
class TrackedObject:
    """A tracked object with Kalman filter for smooth tracking."""
    track_id: str
    object_type: str  # 'vehicle' or 'person'
    bbox: Tuple[int, int, int, int]
    confidence: float
    class_name: str = ""
    first_seen: float = 0.0
    last_seen: float = 0.0
    frame_count: int = 1
    metadata: Dict = field(default_factory=dict)
    is_analyzed: bool = False
    is_reported: bool = False
    consecutive_misses: int = 0
    kalman: Optional[KalmanBoxTracker] = None

    def __post_init__(self):
        if self.kalman is None:
            self.kalman = KalmanBoxTracker(self.bbox)
        if self.first_seen == 0.0:
            self.first_seen = time.time()
        if self.last_seen == 0.0:
            self.last_seen = time.time()

    def predict(self, dt: float = 1/15) -> Tuple[int, int, int, int]:
        """Predict next position using Kalman filter."""
        if self.kalman:
            self.bbox = self.kalman.predict(dt)
        return self.bbox

    def update(self, bbox: Tuple[int, int, int, int], confidence: float):
        """Update with new detection."""
        self.bbox = bbox
        self.confidence = confidence
        self.last_seen = time.time()
        self.frame_count += 1
        self.consecutive_misses = 0

        if self.kalman:
            self.kalman.update(bbox)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "track_id": self.track_id,
            "object_type": self.object_type,
            "bbox": list(self.bbox),
            "confidence": self.confidence,
            "class": self.class_name,
            "metadata": self.metadata,
            "is_analyzed": self.is_analyzed,
            "frame_count": self.frame_count,
            "consecutive_misses": self.consecutive_misses,
            "is_predicted": self.consecutive_misses > 0
        }


class StableTracker:
    """Stable object tracker with Kalman prediction and persistence.

    Key features:
    - Objects must be seen for MIN_FRAMES before reported as "new"
    - Objects kept for MAX_MISSES frames using Kalman prediction
    - Same object won't be reported as "new" twice
    - Uses IoU matching with Kalman-predicted positions
    """

    def __init__(
        self,
        min_frames_to_report: int = 3,
        max_consecutive_misses: int = 60,  # ~4 seconds at 15fps
        iou_threshold: float = 0.25,
        min_confidence: float = 0.35,
        target_fps: float = 15.0
    ):
        self.min_frames_to_report = min_frames_to_report
        self.max_consecutive_misses = max_consecutive_misses
        self.iou_threshold = iou_threshold
        self.min_confidence = min_confidence
        self.target_fps = target_fps

        # Tracked objects by type
        self._vehicles: Dict[str, TrackedObject] = {}
        self._persons: Dict[str, TrackedObject] = {}

        # ID counters
        self._next_vehicle_id = 1
        self._next_person_id = 1

        # Timing
        self._last_update_time = time.time()

        # Statistics
        self._stats = {
            "total_vehicles": 0,
            "total_persons": 0,
            "active_vehicles": 0,
            "active_persons": 0,
            "reported_vehicles": 0,
            "reported_persons": 0
        }

    def update(
        self,
        detections: List[Dict],
        object_type: str
    ) -> Tuple[List[TrackedObject], List[TrackedObject]]:
        """Update tracker with new detections.

        CRITICAL: Even with empty detections, predicts positions using Kalman.
        This keeps bboxes visible and stable when YOLO misses frames.

        Args:
            detections: List of {bbox, confidence, class_name}
            object_type: 'vehicle' or 'person'

        Returns:
            (all_tracked, newly_reportable) - all tracks and new ones to report
        """
        tracks = self._vehicles if object_type == 'vehicle' else self._persons
        now = time.time()
        dt = now - self._last_update_time
        self._last_update_time = now

        # Clamp dt to reasonable range
        dt = max(0.01, min(dt, 0.5))

        # Filter low confidence detections
        detections = [d for d in detections if d.get('confidence', 0) >= self.min_confidence]

        # STEP 1: Predict ALL tracks first (even if no detections!)
        for track in tracks.values():
            track.predict(dt)

        # STEP 2: Match detections to existing tracks using IoU with predicted positions
        matched_tracks = set()
        matched_detections = set()

        for det_idx, det in enumerate(detections):
            det_bbox = det.get('bbox', [])
            if len(det_bbox) < 4:
                continue

            det_bbox = tuple(det_bbox[:4])
            best_match = None
            best_iou = self.iou_threshold

            for track_id, track in tracks.items():
                if track_id in matched_tracks:
                    continue

                # Use Kalman-predicted bbox for matching
                iou = self._calculate_iou(det_bbox, track.bbox)

                if iou > best_iou:
                    best_iou = iou
                    best_match = track_id

            if best_match:
                # Update existing track with detection
                track = tracks[best_match]
                track.update(det_bbox, det.get('confidence', 0))

                matched_tracks.add(best_match)
                matched_detections.add(det_idx)

        # STEP 3: Handle unmatched tracks (increment miss counter)
        for track_id in list(tracks.keys()):
            if track_id not in matched_tracks:
                track = tracks[track_id]
                track.consecutive_misses += 1

                # Remove after too many misses
                if track.consecutive_misses > self.max_consecutive_misses:
                    logger.debug(f"Track removed after {track.consecutive_misses} misses: {track_id}")
                    del tracks[track_id]

        # STEP 4: Create new tracks for unmatched detections
        for det_idx, det in enumerate(detections):
            if det_idx in matched_detections:
                continue

            det_bbox = det.get('bbox', [])
            if len(det_bbox) < 4:
                continue

            det_bbox = tuple(det_bbox[:4])

            # Create new track
            if object_type == 'vehicle':
                track_id = f"v_{self._next_vehicle_id}"
                self._next_vehicle_id += 1
                self._stats["total_vehicles"] += 1
            else:
                track_id = f"p_{self._next_person_id}"
                self._next_person_id += 1
                self._stats["total_persons"] += 1

            track = TrackedObject(
                track_id=track_id,
                object_type=object_type,
                bbox=det_bbox,
                confidence=det.get('confidence', 0),
                class_name=det.get('class', det.get('label', object_type)),
                is_analyzed=False,
                is_reported=False
            )

            tracks[track_id] = track
            logger.debug(f"New track created: {track_id}")

        # Get all current tracks
        all_tracked = list(tracks.values())

        # Find newly reportable tracks (seen enough, not yet reported, currently detected)
        newly_reportable = []
        for track in all_tracked:
            if (not track.is_reported and
                track.frame_count >= self.min_frames_to_report and
                track.consecutive_misses == 0):
                track.is_reported = True
                newly_reportable.append(track)

                if object_type == 'vehicle':
                    self._stats["reported_vehicles"] += 1
                else:
                    self._stats["reported_persons"] += 1

                logger.info(f"New {object_type} to report: {track.track_id}")

        # Update stats
        self._stats["active_vehicles"] = len(self._vehicles)
        self._stats["active_persons"] = len(self._persons)

        return all_tracked, newly_reportable

    def get_all_tracks(self, object_type: str) -> List[TrackedObject]:
        """Get all current tracks including predicted positions."""
        tracks = self._vehicles if object_type == 'vehicle' else self._persons
        return list(tracks.values())

    def get_track(self, track_id: str) -> Optional[TrackedObject]:
        """Get track by ID."""
        if track_id.startswith('v_'):
            return self._vehicles.get(track_id)
        else:
            return self._persons.get(track_id)

    def mark_analyzed(self, track_id: str, metadata: Dict):
        """Mark track as analyzed by Gemini."""
        track = self.get_track(track_id)
        if track:
            track.is_analyzed = True
            track.metadata = metadata

    def get_unanalyzed(self, object_type: str) -> List[TrackedObject]:
        """Get tracks that haven't been analyzed yet."""
        tracks = self._vehicles if object_type == 'vehicle' else self._persons
        return [t for t in tracks.values() if not t.is_analyzed and t.is_reported]

    def get_all_vehicles(self) -> List[TrackedObject]:
        """Get all tracked vehicles."""
        return list(self._vehicles.values())

    def get_all_persons(self) -> List[TrackedObject]:
        """Get all tracked persons."""
        return list(self._persons.values())

    def get_armed_persons(self) -> List[TrackedObject]:
        """Get persons marked as armed."""
        return [
            p for p in self._persons.values()
            if p.metadata.get('armed') or p.metadata.get('חמוש')
        ]

    def get_stats(self) -> Dict:
        """Get tracker statistics."""
        return self._stats.copy()

    def reset(self):
        """Reset all tracks (use sparingly - prefer persistence)."""
        self._vehicles.clear()
        self._persons.clear()
        self._next_vehicle_id = 1
        self._next_person_id = 1
        self._stats = {
            "total_vehicles": 0,
            "total_persons": 0,
            "active_vehicles": 0,
            "active_persons": 0,
            "reported_vehicles": 0,
            "reported_persons": 0
        }
        logger.info("Tracker reset")

    def _calculate_iou(
        self,
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int]
    ) -> float:
        """Calculate Intersection over Union."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)

        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0


# Global singleton - persists across reconnects!
_tracker: Optional[StableTracker] = None


def get_stable_tracker() -> StableTracker:
    """Get or create the global stable tracker."""
    global _tracker
    if _tracker is None:
        _tracker = StableTracker(
            min_frames_to_report=3,
            max_consecutive_misses=60,  # Keep tracks for ~4 seconds
            iou_threshold=0.25,
            min_confidence=0.35
        )
    return _tracker


def reset_stable_tracker():
    """Reset the global tracker (use sparingly)."""
    global _tracker
    if _tracker:
        _tracker.reset()
