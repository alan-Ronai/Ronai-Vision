"""BoT-SORT tracker implementation with Kalman filtering.

Full BoT-SORT implementation combining:
- Kalman filter for motion prediction and box smoothing
- Appearance embeddings (ReID) for robust association
- Proper Hungarian algorithm for optimal assignment
- Track state management (tentative/confirmed/lost)
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

from services.tracker.base_tracker import BaseTracker, Track


class KalmanBoxTracker:
    """Kalman filter for tracking bounding boxes in image space.

    State: [x_center, y_center, width, height, vx, vy, vw, vh]
    - Position and size of box
    - Velocities for motion prediction
    """

    def __init__(self, bbox: np.ndarray):
        """Initialize Kalman filter with initial bounding box.

        Args:
            bbox: [x1, y1, x2, y2] format
        """
        # Convert bbox to [cx, cy, w, h]
        self.bbox = self._convert_bbox_to_z(bbox)

        # State: [cx, cy, w, h, vx, vy, vw, vh]
        self.state = np.zeros(8)
        self.state[:4] = self.bbox

        # State covariance
        self.P = np.eye(8)
        self.P[4:, 4:] *= 1000.0  # High uncertainty in velocity
        self.P *= 10.0

        # Process noise
        self.Q = np.eye(8)
        self.Q[:4, :4] *= 0.1  # Position noise
        self.Q[4:, 4:] *= 0.01  # Velocity noise

        # Measurement noise
        self.R = np.eye(4)
        self.R *= 10.0

        # State transition matrix (constant velocity model)
        self.F = np.eye(8)
        # Position updated by velocity
        self.F[0, 4] = 1.0  # x += vx
        self.F[1, 5] = 1.0  # y += vy
        self.F[2, 6] = 1.0  # w += vw
        self.F[3, 7] = 1.0  # h += vh

        # Measurement matrix (observe position/size only)
        self.H = np.zeros((4, 8))
        self.H[:4, :4] = np.eye(4)

        self.time_since_update = 0
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def _convert_bbox_to_z(self, bbox: np.ndarray) -> np.ndarray:
        """Convert [x1, y1, x2, y2] to [cx, cy, w, h]."""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        cx = bbox[0] + w / 2.0
        cy = bbox[1] + h / 2.0
        return np.array([cx, cy, w, h])

    def _convert_z_to_bbox(self, z: np.ndarray) -> np.ndarray:
        """Convert [cx, cy, w, h] to [x1, y1, x2, y2]."""
        w = z[2]
        h = z[3]
        x1 = z[0] - w / 2.0
        y1 = z[1] - h / 2.0
        x2 = z[0] + w / 2.0
        y2 = z[1] + h / 2.0
        return np.array([x1, y1, x2, y2])

    def predict(self) -> np.ndarray:
        """Predict next state using motion model.

        Returns:
            Predicted bbox in [x1, y1, x2, y2] format
        """
        # Predict state
        self.state = self.F @ self.state

        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Ensure width/height stay positive
        if self.state[2] <= 0:
            self.state[2] = 1.0
        if self.state[3] <= 0:
            self.state[3] = 1.0

        self.age += 1
        self.time_since_update += 1

        return self._convert_z_to_bbox(self.state[:4])

    def update(self, bbox: np.ndarray):
        """Update state with new measurement.

        Args:
            bbox: Detected box in [x1, y1, x2, y2] format
        """
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

        # Convert measurement to state space
        z = self._convert_bbox_to_z(bbox)

        # Innovation (measurement residual)
        y = z - self.H @ self.state

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state
        self.state = self.state + K @ y

        # Update covariance
        I_KH = np.eye(8) - K @ self.H
        self.P = I_KH @ self.P

        # Ensure width/height stay positive
        if self.state[2] <= 0:
            self.state[2] = 1.0
        if self.state[3] <= 0:
            self.state[3] = 1.0

    def get_state(self) -> np.ndarray:
        """Get current state as bbox [x1, y1, x2, y2]."""
        return self._convert_z_to_bbox(self.state[:4])


def _iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Calculate IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / (union + 1e-6)


def _centroid(box: np.ndarray) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance between two feature vectors.

    Returns 1.0 (maximum distance) if either vector is None.
    This happens for non-person detections which have no ReID features.
    """
    if a is None or b is None:
        return 1.0
    # Ensure both vectors have same dimension
    if len(a) != len(b):
        return 1.0
    na = np.linalg.norm(a) + 1e-6
    nb = np.linalg.norm(b) + 1e-6
    return 1.0 - float(np.dot(a, b) / (na * nb))


class BoTSortTracker(BaseTracker):
    """Full BoT-SORT tracker with Kalman filtering and proper Hungarian assignment.

    Features:
    - Kalman filter for motion prediction and box smoothing
    - Appearance (ReID) and motion cost combination
    - Track state management (tentative/confirmed/lost)
    - Optimal assignment via Hungarian algorithm

    Params:
        max_lost: Frames to keep track without detection before deletion
        min_hits: Min detections before track is confirmed
        iou_threshold: IoU threshold for initial association
        lambda_app: Weight for appearance cost (0=motion only, 1=appearance only)
        max_cost: Maximum cost threshold for assignment
        min_confidence_history: Minimum average confidence to keep track (default: 0.25)
    """

    def __init__(
        self,
        max_lost: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        lambda_app: float = 0.7,
        max_cost: float = 0.9,
        min_confidence_history: float = 0.25,
    ):
        self.max_lost = max_lost
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.lambda_app = lambda_app
        self.max_cost = max_cost
        self.min_confidence_history = min_confidence_history

        self.next_id = 1
        self.tracks: Dict[int, Track] = {}
        self.kalman_filters: Dict[int, KalmanBoxTracker] = {}
        self.track_features: Dict[int, np.ndarray] = {}
        self.track_states: Dict[int, str] = {}  # "tentative" or "confirmed"
        self.low_confidence_frames: Dict[int, int] = {}  # Track low-confidence duration

    def update(
        self,
        boxes: np.ndarray,
        class_ids: np.ndarray,
        confidences: np.ndarray,
        features: Optional[np.ndarray] = None,
    ) -> List[Track]:
        """Update tracker with new detections.

        Args:
            boxes: (N, 4) array of [x1, y1, x2, y2]
            class_ids: (N,) array of class indices
            confidences: (N,) array of detection confidences
            features: Optional list of ReID embeddings (can contain None)

        Returns:
            List of confirmed Track objects with smoothed boxes
        """
        N = len(boxes)

        # Predict all tracks forward
        track_ids = list(self.kalman_filters.keys())
        for tid in track_ids:
            predicted_box = self.kalman_filters[tid].predict()
            self.tracks[tid].box = predicted_box

        # Handle empty detections
        if N == 0:
            # Mark all tracks as lost
            to_delete = []
            for tid in track_ids:
                kf = self.kalman_filters[tid]
                if kf.time_since_update > self.max_lost:
                    to_delete.append(tid)

            for tid in to_delete:
                del self.tracks[tid]
                del self.kalman_filters[tid]
                del self.track_states[tid]
                if tid in self.track_features:
                    del self.track_features[tid]

            # Return only confirmed tracks
            return [
                t
                for tid, t in self.tracks.items()
                if self.track_states[tid] == "confirmed"
            ]

        # Initialize tracks if none exist
        if len(track_ids) == 0:
            for i in range(N):
                self._create_new_track(
                    boxes[i], class_ids[i], confidences[i], features, i
                )
            return [
                t
                for tid, t in self.tracks.items()
                if self.track_states[tid] == "confirmed"
            ]

        # Build cost matrix: tracks x detections
        M = len(track_ids)
        cost_matrix = np.zeros((M, N), dtype=np.float32)

        for i, tid in enumerate(track_ids):
            predicted_box = self.tracks[tid].box
            track_feat = self.track_features.get(tid, None)

            for j in range(N):
                det_box = boxes[j]
                det_feat = (
                    features[j] if features is not None and len(features) > j else None
                )

                # IoU-based motion cost
                iou = _iou(predicted_box, det_box)
                motion_cost = 1.0 - iou

                # Appearance cost
                app_cost = _cosine(track_feat, det_feat)

                # Combined cost
                cost_matrix[i, j] = (
                    1.0 - self.lambda_app
                ) * motion_cost + self.lambda_app * app_cost

        # Hungarian assignment
        if cost_matrix.size > 0:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        else:
            row_ind, col_ind = [], []

        matched_tracks = set()
        matched_dets = set()

        # Apply matches
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] > self.max_cost:
                continue

            tid = track_ids[r]

            # Update Kalman filter with measurement
            self.kalman_filters[tid].update(boxes[c])

            # Update track with smoothed box from Kalman
            smoothed_box = self.kalman_filters[tid].get_state()
            self.tracks[tid].box = smoothed_box
            self.tracks[tid].class_id = int(class_ids[c])
            self.tracks[tid].update_confidence(float(confidences[c]))
            self.tracks[tid].hits += 1

            # Update appearance feature
            if features is not None and len(features) > c and features[c] is not None:
                self.track_features[tid] = features[c]

            # Promote to confirmed if enough hits
            if self.kalman_filters[tid].hit_streak >= self.min_hits:
                self.track_states[tid] = "confirmed"

            # Reset low confidence counter on successful match
            self.low_confidence_frames[tid] = 0

            matched_tracks.add(tid)
            matched_dets.add(c)

        # Unmatched detections -> new tracks
        for j in range(N):
            if j in matched_dets:
                continue
            self._create_new_track(boxes[j], class_ids[j], confidences[j], features, j)

        # Unmatched tracks -> check for deletion
        to_delete = []
        for tid in track_ids:
            if tid not in matched_tracks:
                kf = self.kalman_filters[tid]
                if kf.time_since_update > self.max_lost:
                    to_delete.append(tid)
                elif kf.hit_streak > 0:
                    kf.hit_streak = 0  # Reset hit streak on miss

        # Validate tracks based on confidence history
        # Delete tracks with low average confidence (likely false positives)
        for tid in list(self.tracks.keys()):
            if tid in to_delete:
                continue

            track = self.tracks[tid]
            avg_conf = track.get_avg_confidence()

            # Track low confidence duration
            if avg_conf < self.min_confidence_history:
                self.low_confidence_frames[tid] = (
                    self.low_confidence_frames.get(tid, 0) + 1
                )

                # Delete if low confidence for too long (30 frames)
                if self.low_confidence_frames[tid] > 30:
                    print(
                        f"[BoTSORT] Deleting track {tid} due to low confidence (avg={avg_conf:.2f})"
                    )
                    to_delete.append(tid)
            else:
                # Reset counter if confidence recovered
                self.low_confidence_frames[tid] = 0

        for tid in to_delete:
            del self.tracks[tid]
            del self.kalman_filters[tid]
            del self.track_states[tid]
            if tid in self.track_features:
                del self.track_features[tid]
            if tid in self.low_confidence_frames:
                del self.low_confidence_frames[tid]

        # Return only confirmed tracks
        return [
            t for tid, t in self.tracks.items() if self.track_states[tid] == "confirmed"
        ]

    def _create_new_track(
        self,
        box: np.ndarray,
        class_id: int,
        confidence: float,
        features: Optional[List],
        det_idx: int,
    ):
        """Create a new tentative track."""
        tid = self.next_id
        self.next_id += 1

        # Create track
        track = Track(
            track_id=tid,
            box=box.copy(),
            class_id=int(class_id),
            confidence=float(confidence),
        )
        track.update_confidence(float(confidence))  # Initialize confidence history
        self.tracks[tid] = track

        # Initialize Kalman filter
        self.kalman_filters[tid] = KalmanBoxTracker(box)

        # Store feature if available
        if (
            features is not None
            and len(features) > det_idx
            and features[det_idx] is not None
        ):
            self.track_features[tid] = features[det_idx]

        # Start as tentative
        self.track_states[tid] = "tentative"
        self.low_confidence_frames[tid] = 0  # Initialize low confidence counter

    def get_active_tracks(self) -> List[Track]:
        """Return all confirmed tracks."""
        return [
            t for tid, t in self.tracks.items() if self.track_states[tid] == "confirmed"
        ]
