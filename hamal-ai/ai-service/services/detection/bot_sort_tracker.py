"""BoT-SORT Tracker with full Kalman filter and Hungarian assignment.

Based on BoT-SORT paper: https://arxiv.org/abs/2206.14651

Key features:
1. Full 8-state Kalman filter with covariance matrices
2. Hungarian assignment for optimal matching
3. Combined cost (motion + appearance)
4. Confidence history tracking
5. Track state management (tentative → confirmed → lost)
6. Low-confidence track deletion
"""

import numpy as np
import logging
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


# Configuration Constants
MIN_HITS = 2  # Detections needed to confirm track (lowered from 3 for faster confirmation)
MAX_LOST = 15  # Frames before deletion (reduced from 30 for faster ghost cleanup = 1 second at 15fps)
MAX_LOST_WITH_REID = 90  # Frames before deletion for tracks WITH ReID features (6 seconds at 15fps)
LAMBDA_APP = 0.8  # Appearance weight in cost matrix (increased from 0.7 for better ReID matching)
MAX_COST = 0.85  # Maximum cost for assignment (increased for moving object tolerance)
MAX_COST_REID_ONLY = 0.5  # Maximum cost when using ReID-only matching (no IoU overlap)
MIN_IOU_THRESHOLD = 0.2  # Minimum IoU for matching (lowered from 0.3 for walking persons)

# Low confidence deletion (more tolerant to prevent premature ID switching)
MIN_CONFIDENCE_HISTORY = 0.20  # Avg confidence threshold (lowered from 0.25)
LOW_CONF_MAX_FRAMES = 60  # Frames at low conf before deletion (increased from 30)


@dataclass
class Detection:
    """Single detection."""
    bbox: Tuple[float, float, float, float]  # x, y, w, h
    confidence: float
    class_id: int
    class_name: str
    feature: Optional[np.ndarray] = None  # ReID feature vector


class KalmanBoxTracker:
    """8-state Kalman filter for bounding box tracking.

    State: [cx, cy, w, h, vx, vy, vw, vh]
    - cx, cy: center coordinates
    - w, h: width, height
    - vx, vy: velocity of center
    - vw, vh: velocity of size
    """

    def __init__(self, bbox: Tuple[float, float, float, float]):
        """Initialize Kalman filter with detection bbox (x, y, w, h)."""

        # State vector [cx, cy, w, h, vx, vy, vw, vh]
        self.state = np.zeros(8)
        self.state[0] = bbox[0] + bbox[2] / 2  # cx
        self.state[1] = bbox[1] + bbox[3] / 2  # cy
        self.state[2] = bbox[2]  # w
        self.state[3] = bbox[3]  # h
        # Velocities initialized to 0

        # Covariance matrix (uncertainty)
        self.P = np.eye(8)
        self.P[4:, 4:] *= 100.0  # High uncertainty for velocity

        # Process noise covariance (INCREASED for better adaptation to movement)
        self.Q = np.eye(8)
        self.Q[0:4, 0:4] *= 1.0  # Position noise (increased from 0.01 for faster adaptation)
        self.Q[4:, 4:] *= 0.1  # Velocity noise (increased from 0.01 to allow velocity changes)

        # Measurement noise covariance
        self.R = np.eye(4)
        self.R *= 10.0  # Measurement uncertainty

        # State transition matrix (constant velocity model)
        self.F = np.eye(8)
        # Will be updated with dt in predict()

        # Measurement matrix (we observe position, not velocity)
        self.H = np.zeros((4, 8))
        self.H[0, 0] = 1  # cx
        self.H[1, 1] = 1  # cy
        self.H[2, 2] = 1  # w
        self.H[3, 3] = 1  # h

    def predict(self, dt: float = 1/15) -> Tuple[float, float, float, float]:
        """Predict next state using constant velocity model.

        Use full velocity for position prediction to properly track moving objects.
        This helps the predicted position match the actual next detection.

        Returns:
            Predicted bbox (x, y, w, h)
        """
        # Apply velocity with moderate damping for position
        # Full velocity (1.0) helps prediction follow moving objects
        self.F[0, 4] = 1.0  # cx += vx (full velocity for accurate prediction)
        self.F[1, 5] = 1.0  # cy += vy (this helps match moving objects!)

        # Size changes slower to prevent box size oscillation
        self.F[2, 6] = 0.5  # w += vw * 0.5
        self.F[3, 7] = 0.5  # h += vh * 0.5

        # Prediction step
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Return predicted bbox in (x, y, w, h) format
        cx, cy, w, h = self.state[0:4]
        x = cx - w / 2
        y = cy - h / 2

        # Ensure non-negative dimensions
        w = max(1, w)
        h = max(1, h)

        return (x, y, w, h)

    def update(self, bbox: Tuple[float, float, float, float]):
        """Update state with new measurement (detection).

        Args:
            bbox: Measured bbox (x, y, w, h)
        """
        # Convert bbox to measurement vector [cx, cy, w, h]
        cx = bbox[0] + bbox[2] / 2
        cy = bbox[1] + bbox[3] / 2
        z = np.array([cx, cy, bbox[2], bbox[3]])

        # Innovation (measurement residual)
        y = z - (self.H @ self.state)

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state
        self.state = self.state + K @ y

        # Update covariance
        I = np.eye(8)
        self.P = (I - K @ self.H) @ self.P

    def get_bbox(self) -> Tuple[float, float, float, float]:
        """Get current bounding box (x, y, w, h)."""
        cx, cy, w, h = self.state[0:4]
        x = cx - w / 2
        y = cy - h / 2
        w = max(1, w)
        h = max(1, h)
        return (x, y, w, h)


class Track:
    """Single object track with state management."""

    track_id_counter = 0

    def __init__(self, detection: Detection, track_id: Optional[str] = None, camera_id: Optional[str] = None):
        """Initialize track with first detection."""

        # Generate unique track ID
        if track_id is None:
            Track.track_id_counter += 1
            self.track_id = f"t_{Track.track_id_counter}"
        else:
            self.track_id = track_id

        # Detection info
        self.class_id = detection.class_id
        self.class_name = detection.class_name

        # CRITICAL: Track which camera this object belongs to
        self.camera_id = camera_id
        self.last_seen_camera = camera_id

        # Kalman filter
        self.kalman = KalmanBoxTracker(detection.bbox)
        self.bbox = detection.bbox

        # CRITICAL: Store last actual detection position for matching
        # This solves the Kalman velocity lag problem where predictions
        # don't move on the first few frames because velocity = 0
        self.last_detection_bbox = detection.bbox

        # ReID feature (for person class)
        self.feature = detection.feature

        # Confidence tracking
        self.confidence = detection.confidence
        self.confidence_history = [detection.confidence]

        # Track state
        self.state = "tentative"  # tentative → confirmed → lost
        self.hits = 1  # Total successful updates
        self.hit_streak = 1  # Consecutive successful updates
        self.time_since_update = 0  # Frames since last update

        # Timestamps
        self.first_seen = time.time()
        self.last_seen = time.time()

        # Metadata
        self.metadata = {}
        self.is_reported = False  # For "new object" event logic

    def predict(self, dt: float = 1/15):
        """Predict next position using Kalman filter."""
        self.bbox = self.kalman.predict(dt)
        self.time_since_update += 1

        if self.time_since_update > 0:
            self.hit_streak = 0

    def update(self, detection: Detection, dt: float = 1/15, camera_id: Optional[str] = None):
        """Update track with new detection."""
        # Update Kalman filter
        self.kalman.update(detection.bbox)
        self.bbox = detection.bbox

        # CRITICAL: Store last actual detection position for next frame's matching
        # This ensures we match against real observed position, not Kalman prediction
        self.last_detection_bbox = detection.bbox

        # Update camera tracking (for cross-camera filtering)
        if camera_id:
            self.last_seen_camera = camera_id

        # Update confidence
        self.confidence = detection.confidence
        self.confidence_history.append(detection.confidence)
        if len(self.confidence_history) > 30:
            self.confidence_history.pop(0)

        # Update ReID feature (EMA for robustness)
        if detection.feature is not None:
            if self.feature is None:
                self.feature = detection.feature
            else:
                alpha = 0.9  # EMA weight for existing feature
                self.feature = alpha * self.feature + (1 - alpha) * detection.feature
                # Normalize
                self.feature = self.feature / np.linalg.norm(self.feature)

        # Update state
        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0  # Reset - this track was matched!
        self.last_seen = time.time()

        # State transition: tentative → confirmed
        if self.state == "tentative" and self.hits >= MIN_HITS:
            self.state = "confirmed"
            logger.info(f"Track {self.track_id} ({self.class_name}) confirmed on camera {camera_id}")

    @property
    def avg_confidence(self) -> float:
        """Average confidence over history."""
        return sum(self.confidence_history) / len(self.confidence_history) if self.confidence_history else 0.0

    def should_delete(self) -> bool:
        """Check if track should be deleted.

        Logic:
        - Tracks WITH ReID features are kept longer (for re-identification after video loop)
        - Tracks WITHOUT ReID features use shorter timeout
        - Delete if low confidence AND not being matched
        """
        # Use longer timeout for tracks with ReID features (allows re-identification after video loops)
        max_lost = MAX_LOST_WITH_REID if self.feature is not None else MAX_LOST

        # Delete if lost for too long (not matched for max_lost frames)
        if self.time_since_update > max_lost:
            return True

        # Delete if consistently low confidence AND not being matched recently
        if (self.time_since_update > 5 and  # Not matched for 5+ frames
            len(self.confidence_history) >= LOW_CONF_MAX_FRAMES and
            self.avg_confidence < MIN_CONFIDENCE_HISTORY):
            logger.debug(f"Deleting track {self.track_id} due to low confidence: {self.avg_confidence:.2f}")
            return True

        return False

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "avg_confidence": self.avg_confidence,
            "state": self.state,
            "hits": self.hits,
            "time_since_update": self.time_since_update,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "is_reported": self.is_reported,
            "camera_id": self.camera_id,
            "last_seen_camera": self.last_seen_camera
        }


class BoTSORTTracker:
    """BoT-SORT tracker with Hungarian assignment and ReID."""

    def __init__(self):
        """Initialize tracker."""
        self._persons: Dict[str, Track] = {}
        self._vehicles: Dict[str, Track] = {}

        # Stats
        self._stats = {
            "total_tracks": 0,
            "active_tracks": 0,
            "deleted_tracks": 0
        }

    def update(
        self,
        detections: List[Detection],
        object_type: str,
        dt: float = 1/15,
        camera_id: Optional[str] = None
    ) -> Tuple[List[Track], List[Track]]:
        """Update tracker with new detections.

        Args:
            detections: List of Detection objects
            object_type: 'person' or 'vehicle'
            dt: Time delta since last frame
            camera_id: Camera ID for tracking per-camera objects

        Returns:
            (all_tracks, new_tracks): All current tracks and newly confirmed tracks
        """
        tracks = self._persons if object_type == 'person' else self._vehicles

        # STEP 1: Predict all existing tracks
        for track in list(tracks.values()):
            track.predict(dt)

        # STEP 2: Build cost matrix
        if detections and tracks:
            cost_matrix = self._build_cost_matrix(tracks, detections, object_type)

            # STEP 3: Hungarian assignment
            track_indices, det_indices = linear_sum_assignment(cost_matrix)

            # Filter out high-cost matches
            matched_tracks = set()
            matched_detections = set()

            for track_idx, det_idx in zip(track_indices, det_indices):
                if cost_matrix[track_idx, det_idx] < MAX_COST:
                    track_id = list(tracks.keys())[track_idx]
                    tracks[track_id].update(detections[det_idx], dt, camera_id)
                    matched_tracks.add(track_id)
                    matched_detections.add(det_idx)
        else:
            matched_tracks = set()
            matched_detections = set()

        # STEP 4: Create new tracks for unmatched detections
        new_tracks = []
        for i, detection in enumerate(detections):
            if i not in matched_detections:
                track = Track(detection, camera_id=camera_id)
                tracks[track.track_id] = track
                self._stats["total_tracks"] += 1

                # Only report as "new" when confirmed
                if track.state == "confirmed":
                    new_tracks.append(track)

        # STEP 5: Delete old tracks
        for track_id in list(tracks.keys()):
            if tracks[track_id].should_delete():
                del tracks[track_id]
                self._stats["deleted_tracks"] += 1

        # STEP 6: Find newly confirmed tracks
        for track_id in matched_tracks:
            track = tracks[track_id]
            if track.state == "confirmed" and not track.is_reported:
                new_tracks.append(track)
                track.is_reported = True

        # Update stats
        self._stats["active_tracks"] = len(self._persons) + len(self._vehicles)

        return list(tracks.values()), new_tracks

    def _build_cost_matrix(
        self,
        tracks: Dict[str, Track],
        detections: List[Detection],
        object_type: str
    ) -> np.ndarray:
        """Build combined cost matrix (motion + appearance).

        CRITICAL FIX: Use motion-only cost when ReID features are not available.
        FALLBACK: Use center distance when IoU is low (helps with box size changes).

        Args:
            tracks: Dictionary of active tracks
            detections: List of detections
            object_type: 'person' or 'vehicle'

        Returns:
            Cost matrix (num_tracks x num_detections)
        """
        track_list = list(tracks.values())
        num_tracks = len(track_list)
        num_dets = len(detections)

        cost_matrix = np.zeros((num_tracks, num_dets))

        for i, track in enumerate(track_list):
            for j, det in enumerate(detections):
                # CRITICAL: Use last_detection_bbox instead of track.bbox (Kalman prediction)
                # This solves the velocity lag problem where Kalman has velocity=0 initially
                # and predictions don't move, causing IoU failures for moving objects
                match_bbox = track.last_detection_bbox

                # Motion cost (1 - IoU)
                iou = self._calculate_iou(match_bbox, det.bbox)
                motion_cost = 1.0 - iou

                # FALLBACK: If IoU is low, use center distance
                # This helps when box sizes differ but centers are close (e.g., perspective change)
                if iou < 0.3:
                    center_dist = self._center_distance(match_bbox, det.bbox)
                    # Normalize by average box diagonal
                    avg_diag = (self._box_diagonal(match_bbox) + self._box_diagonal(det.bbox)) / 2
                    normalized_dist = center_dist / (avg_diag + 1e-6)

                    # If centers are very close (< 30% of diagonal), give it a lower cost
                    if normalized_dist < 0.3:
                        motion_cost = min(motion_cost, normalized_dist + 0.2)

                # CRITICAL: Use appearance cost if BOTH have features (for ANY class)
                # Multi-class ReID support:
                # - Persons: OSNet (512-dim)
                # - Vehicles: TransReID (768-dim)
                # - Others: CLIP (768-dim)
                if (track.feature is not None and det.feature is not None):
                    # Verify feature dimensions match (safety check)
                    if track.feature.shape == det.feature.shape:
                        # Both have features with matching dimensions - use combined cost
                        similarity = self._cosine_similarity(track.feature, det.feature)
                        app_cost = 1.0 - similarity

                        # CRITICAL: When IoU is very low (e.g., video loop, object reappears elsewhere),
                        # rely more heavily on ReID similarity for matching
                        if iou < 0.1 and similarity > 0.7:
                            # ReID-only matching: object likely reappeared at different position
                            # Use only appearance cost when ReID similarity is high
                            cost_matrix[i, j] = app_cost * 0.6  # Scale down to favor good ReID matches
                            logger.debug(
                                f"ReID-only matching ({object_type}): track {track.track_id} "
                                f"similarity={similarity:.3f}, cost={cost_matrix[i, j]:.3f} "
                                f"(IoU too low: {iou:.3f})"
                            )
                        else:
                            # Normal combined cost
                            cost_matrix[i, j] = (1 - LAMBDA_APP) * motion_cost + LAMBDA_APP * app_cost
                            logger.debug(
                                f"ReID matching ({object_type}): track {track.track_id} "
                                f"similarity={similarity:.3f}, motion_cost={motion_cost:.3f}, "
                                f"combined_cost={cost_matrix[i, j]:.3f}"
                            )
                    else:
                        # Feature dimension mismatch - use only motion cost
                        logger.warning(
                            f"Feature dimension mismatch for track {track.track_id}: "
                            f"track={track.feature.shape} vs det={det.feature.shape}"
                        )
                        cost_matrix[i, j] = motion_cost
                else:
                    # NO features - use ONLY motion cost
                    cost_matrix[i, j] = motion_cost

        return cost_matrix

    @staticmethod
    def _calculate_iou(bbox1: Tuple[float, float, float, float],
                      bbox2: Tuple[float, float, float, float]) -> float:
        """Calculate IoU between two bboxes (x, y, w, h)."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Convert to (x1, y1, x2, y2)
        box1 = [x1, y1, x1 + w1, y1 + h1]
        box2 = [x2, y2, x2 + w2, y2 + h2]

        # Intersection
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)

        # Union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _cosine_similarity(feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Calculate cosine similarity between two feature vectors."""
        dot_product = np.dot(feat1, feat2)
        norm1 = np.linalg.norm(feat1)
        norm2 = np.linalg.norm(feat2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    @staticmethod
    def _center_distance(bbox1: Tuple[float, float, float, float],
                        bbox2: Tuple[float, float, float, float]) -> float:
        """Calculate center-to-center Euclidean distance between two bboxes."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        cx1, cy1 = x1 + w1/2, y1 + h1/2
        cx2, cy2 = x2 + w2/2, y2 + h2/2
        return np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)

    @staticmethod
    def _box_diagonal(bbox: Tuple[float, float, float, float]) -> float:
        """Calculate diagonal length of bounding box."""
        x, y, w, h = bbox
        return np.sqrt(w**2 + h**2)

    def get_active_tracks(self, object_type: str, camera_id: Optional[str] = None) -> List[Track]:
        """Get all active tracks of given type, optionally filtered by camera.

        Args:
            object_type: 'person' or 'vehicle'
            camera_id: Optional camera ID to filter tracks

        Returns:
            List of tracks (filtered by camera if camera_id provided)
        """
        tracks = self._persons if object_type == 'person' else self._vehicles
        all_tracks = list(tracks.values())

        # CRITICAL: Filter by camera to prevent cross-contamination
        if camera_id is not None:
            return [t for t in all_tracks if t.last_seen_camera == camera_id]

        return all_tracks

    def get_confirmed_tracks(self, object_type: str, camera_id: Optional[str] = None) -> List[Track]:
        """Get only confirmed tracks, optionally filtered by camera."""
        tracks = self.get_active_tracks(object_type, camera_id)
        return [t for t in tracks if t.state == "confirmed"]

    def get_stats(self) -> dict:
        """Get tracker statistics."""
        return {
            **self._stats,
            "persons": len(self._persons),
            "vehicles": len(self._vehicles)
        }

    def reset(self):
        """Reset tracker (clear all tracks)."""
        self._persons.clear()
        self._vehicles.clear()
        Track.track_id_counter = 0
        self._stats = {
            "total_tracks": 0,
            "active_tracks": 0,
            "deleted_tracks": 0
        }
        logger.info("BoT-SORT tracker reset")

    def clear_camera_tracks(self, camera_id: str):
        """Clear all tracks for a specific camera.

        Called when a camera is removed/disconnected to prevent stale tracks
        from triggering events.

        Args:
            camera_id: Camera ID whose tracks should be cleared
        """
        cleared_persons = 0
        cleared_vehicles = 0

        # Clear persons for this camera
        to_delete = [tid for tid, track in self._persons.items()
                     if track.last_seen_camera == camera_id]
        for tid in to_delete:
            del self._persons[tid]
            cleared_persons += 1

        # Clear vehicles for this camera
        to_delete = [tid for tid, track in self._vehicles.items()
                     if track.last_seen_camera == camera_id]
        for tid in to_delete:
            del self._vehicles[tid]
            cleared_vehicles += 1

        if cleared_persons > 0 or cleared_vehicles > 0:
            logger.info(f"Cleared tracks for camera {camera_id}: {cleared_persons} persons, {cleared_vehicles} vehicles")
            self._stats["deleted_tracks"] += cleared_persons + cleared_vehicles

    def get_active_tracks_visible(self, object_type: str, camera_id: Optional[str] = None,
                                   max_frames_since_update: int = 5) -> List[Track]:
        """Get only tracks that are currently visible (recently updated).

        Args:
            object_type: 'person' or 'vehicle'
            camera_id: Optional camera ID to filter tracks
            max_frames_since_update: Maximum frames since last update (default 5 = ~330ms at 15fps)

        Returns:
            List of recently updated tracks (likely still visible)
        """
        tracks = self.get_active_tracks(object_type, camera_id)
        # Filter to only tracks that have been updated recently
        return [t for t in tracks if t.time_since_update <= max_frames_since_update]


# Global singleton
_tracker: Optional[BoTSORTTracker] = None

def get_bot_sort_tracker() -> BoTSORTTracker:
    """Get or create global BoT-SORT tracker."""
    global _tracker
    if _tracker is None:
        _tracker = BoTSORTTracker()
        logger.info("BoT-SORT tracker initialized")
    return _tracker
