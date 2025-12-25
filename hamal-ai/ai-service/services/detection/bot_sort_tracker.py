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
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


# Configuration Constants
MIN_HITS = 2  # Detections needed to confirm track (lowered from 3 for faster confirmation)
MAX_LOST = 15  # Frames before deletion (reduced from 30 for faster ghost cleanup = 1 second at 15fps)
MAX_LOST_WITH_REID = 90  # Frames before deletion for tracks WITH ReID features (6 seconds at 15fps)
LAMBDA_APP = 0.85  # Appearance weight in cost matrix (increased for better ReID matching)
MAX_COST = 0.9  # Maximum cost for assignment (increased for moving object tolerance)
MAX_COST_REID_ONLY = 0.5  # Maximum cost when using ReID-only matching (no IoU overlap)
MIN_IOU_THRESHOLD = 0.15  # Minimum IoU for matching (lowered for walking persons and fast movement)

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
    is_recovery: bool = False  # True if from low-confidence recovery pass


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
        # Higher values = Kalman filter adapts faster to changes (less smooth, more responsive)
        self.Q = np.eye(8)
        self.Q[0:4, 0:4] *= 2.0  # Position noise (increased for fast-moving objects)
        self.Q[4:, 4:] *= 0.5  # Velocity noise (increased to allow velocity changes)

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

    # CRITICAL: Single global counter for GID to prevent conflicts between persons/vehicles
    # Previously had separate counters which caused v_0_3 and p_0_3 to share GID=3
    global_id_counter = 0
    # Session counter increments on each tracker reset to prevent ID collisions
    # This ensures v_1 from session 1 is different from v_1 from session 2
    session_counter = 0

    def __init__(self, detection: Detection, track_id: Optional[str] = None, camera_id: Optional[str] = None, object_type: str = "person"):
        """Initialize track with first detection.

        Args:
            detection: Detection object
            track_id: Optional explicit track ID
            camera_id: Camera ID
            object_type: 'person' or 'vehicle' - determines track_id prefix
        """

        # Generate unique track ID with session prefix to prevent collisions after restart
        # Format: v_<session>_<gid> or p_<session>_<gid>
        # CRITICAL: Use single global counter for GID to prevent conflicts
        if track_id is None:
            Track.global_id_counter += 1
            prefix = "v" if object_type == "vehicle" else "p"
            self.track_id = f"{prefix}_{Track.session_counter}_{Track.global_id_counter}"
        else:
            self.track_id = track_id

        # Store object type
        self.object_type = object_type

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

    def __init__(self, gallery=None):
        """Initialize tracker.

        Args:
            gallery: Optional ReIDGallery instance for persistent re-identification
        """
        self._persons: Dict[str, Track] = {}
        self._vehicles: Dict[str, Track] = {}

        # ReID Gallery for persistent matching across video loops
        self._gallery = gallery

        # Session counter - incremented on clear to invalidate in-flight operations
        self._session_id = 1
        self._clearing = False  # Flag to indicate clearing is in progress

        # Stats
        self._stats = {
            "total_tracks": 0,
            "active_tracks": 0,
            "deleted_tracks": 0,
            "gallery_matches": 0,  # Tracks recovered from gallery
        }

    def set_gallery(self, gallery):
        """Set the ReID gallery for persistent matching."""
        self._gallery = gallery
        if gallery:
            logger.info("ReID Gallery attached to BoT-SORT tracker")

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
        # CHECK GALLERY FIRST: Try to match against persistent gallery before creating new track
        new_tracks = []
        # Only exclude GIDs active in THIS camera - allows cross-camera re-identification
        active_gids_this_camera = self._get_active_gids(object_type, camera_id)

        # Track GIDs created in THIS frame to prevent duplicates within same frame
        gids_created_this_frame: Dict[int, str] = {}  # gid -> track_id

        for i, detection in enumerate(detections):
            if i not in matched_detections:
                reused_track_id = None

                # Check gallery for match if detection has a feature
                has_gallery = self._gallery is not None
                has_feature = detection.feature is not None

                if has_gallery and has_feature:
                    gallery_stats = self._gallery.get_stats()
                    logger.debug(
                        f"Gallery check: {object_type}, gallery_size={gallery_stats['total']}, "
                        f"feature_dim={len(detection.feature)}, excluding={len(active_gids_this_camera)} GIDs"
                    )
                    gallery_match = self._gallery.find_match(
                        feature=detection.feature,
                        object_type=object_type,
                        exclude_gids=list(active_gids_this_camera),  # Only exclude GIDs in same camera
                    )

                    if gallery_match:
                        matched_gid, similarity = gallery_match
                        # Create track with existing GID from gallery
                        prefix = "v" if object_type == "vehicle" else "p"
                        reused_track_id = f"{prefix}_{Track.session_counter}_{matched_gid}"
                        logger.info(
                            f"Gallery re-identification: {object_type} GID {matched_gid} "
                            f"recovered (similarity={similarity:.3f})"
                        )
                        self._stats["gallery_matches"] += 1

                        # Update gallery with new appearance
                        self._gallery.add_or_update(
                            gid=matched_gid,
                            feature=detection.feature,
                            object_type=object_type,
                            camera_id=camera_id,
                            confidence=detection.confidence,
                        )
                    else:
                        # NO GALLERY MATCH - check if any detection in THIS frame is similar
                        # This prevents multiple GIDs for the same object detected multiple times in one frame
                        threshold = self._gallery._get_threshold(object_type) if self._gallery else 0.5
                        for existing_gid, existing_track_id in gids_created_this_frame.items():
                            existing_track = tracks.get(existing_track_id)
                            if existing_track and existing_track.feature is not None:
                                similarity = self._gallery.cosine_similarity(detection.feature, existing_track.feature)
                                if similarity > threshold:
                                    # Same object as existing track in this frame - merge into it
                                    prefix = "v" if object_type == "vehicle" else "p"
                                    reused_track_id = f"{prefix}_{Track.session_counter}_{existing_gid}"
                                    logger.info(
                                        f"In-frame dedup: {object_type} merged into GID {existing_gid} "
                                        f"(similarity={similarity:.3f})"
                                    )
                                    self._stats["gallery_matches"] += 1
                                    break

                # Create track (with reused ID if gallery matched or in-frame dedup)
                track = Track(detection, track_id=reused_track_id, camera_id=camera_id, object_type=object_type)
                tracks[track.track_id] = track
                self._stats["total_tracks"] += 1

                # CRITICAL: Add NEW tracks to gallery immediately so subsequent detections can match
                # This prevents multiple GIDs for the same object appearing multiple times in the same frame
                if reused_track_id is None and has_gallery and has_feature:
                    # Extract GID from the new track_id
                    match = re.search(r'[vpt]_\d+_(\d+)', track.track_id)
                    if match:
                        new_gid = int(match.group(1))
                        self._gallery.add_or_update(
                            gid=new_gid,
                            feature=detection.feature,
                            object_type=object_type,
                            camera_id=camera_id,
                            confidence=detection.confidence,
                        )
                        # Track this GID for in-frame deduplication
                        gids_created_this_frame[new_gid] = track.track_id
                        logger.debug(f"Immediately added new {object_type} GID {new_gid} to gallery")

                # Only report as "new" when confirmed
                if track.state == "confirmed":
                    new_tracks.append(track)

        # STEP 5: Delete old tracks
        for track_id in list(tracks.keys()):
            if tracks[track_id].should_delete():
                del tracks[track_id]
                self._stats["deleted_tracks"] += 1

        # STEP 5.5: Consolidate overlapping tracks (merge duplicates)
        # Only run every 10 frames to reduce CPU overhead
        self._consolidation_counter = getattr(self, '_consolidation_counter', 0) + 1
        if self._consolidation_counter % 10 == 0:
            self._consolidate_overlapping_tracks(tracks, object_type)

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

                        # CRITICAL FIX FOR TRACK FRAGMENTATION:
                        # When ReID similarity is high, trust appearance over motion.
                        # This prevents ID switches when objects move quickly (low IoU).
                        #
                        # Strategy:
                        # - Very high similarity (>0.8): Trust ReID heavily, minimize motion weight
                        # - High similarity (>0.6): Use ReID-weighted cost even with low IoU
                        # - Moderate similarity: Use standard combined cost
                        if similarity > 0.8:
                            # Very high ReID match - almost certainly the same object
                            # Use mostly appearance cost, minimal motion influence
                            cost_matrix[i, j] = app_cost * 0.4 + motion_cost * 0.1
                            logger.debug(
                                f"Strong ReID match ({object_type}): track {track.track_id} "
                                f"similarity={similarity:.3f}, cost={cost_matrix[i, j]:.3f}"
                            )
                        elif similarity > 0.6 and iou < 0.3:
                            # Good ReID match with low IoU - object likely moved
                            # Trust ReID more than motion
                            cost_matrix[i, j] = app_cost * 0.5 + motion_cost * 0.2
                            logger.debug(
                                f"ReID-preferred match ({object_type}): track {track.track_id} "
                                f"similarity={similarity:.3f}, iou={iou:.3f}, cost={cost_matrix[i, j]:.3f}"
                            )
                        elif iou < 0.1 and similarity > 0.5:
                            # Very low IoU but decent similarity - object reappeared elsewhere
                            cost_matrix[i, j] = app_cost * 0.6
                            logger.debug(
                                f"ReID-only matching ({object_type}): track {track.track_id} "
                                f"similarity={similarity:.3f}, cost={cost_matrix[i, j]:.3f} "
                                f"(IoU too low: {iou:.3f})"
                            )
                        else:
                            # Normal combined cost with standard weights
                            cost_matrix[i, j] = (1 - LAMBDA_APP) * motion_cost + LAMBDA_APP * app_cost
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

    def _consolidate_overlapping_tracks(
        self,
        tracks: Dict[str, Track],
        object_type: str,
        iou_threshold: float = 0.5,
        reid_threshold: float = 0.6,
    ) -> None:
        """Merge overlapping tracks that are clearly the same object.

        This catches cases where multiple tracks were accidentally created
        for the same object (e.g., due to detection flicker or failed matching).

        Keeps the older track (more established) and deletes the newer one.

        Args:
            tracks: Dictionary of active tracks (modified in place)
            object_type: 'person' or 'vehicle'
            iou_threshold: IoU threshold for considering tracks as overlapping
            reid_threshold: ReID similarity threshold for confirming same object
        """
        if len(tracks) < 2:
            return

        track_list = list(tracks.values())
        to_delete = set()

        for i, track1 in enumerate(track_list):
            if track1.track_id in to_delete:
                continue

            for j, track2 in enumerate(track_list):
                if i >= j or track2.track_id in to_delete:
                    continue

                # Check IoU overlap
                iou = self._calculate_iou(track1.last_detection_bbox, track2.last_detection_bbox)

                if iou < iou_threshold:
                    continue

                # High IoU - check if they're the same object
                should_merge = False

                # If both have ReID features, check similarity
                if track1.feature is not None and track2.feature is not None:
                    if track1.feature.shape == track2.feature.shape:
                        similarity = self._cosine_similarity(track1.feature, track2.feature)
                        if similarity > reid_threshold:
                            should_merge = True
                            logger.info(
                                f"Merging overlapping {object_type} tracks: "
                                f"{track1.track_id} and {track2.track_id} "
                                f"(IoU={iou:.2f}, similarity={similarity:.2f})"
                            )
                else:
                    # No ReID features but very high IoU - likely same object
                    if iou > 0.7:
                        should_merge = True
                        logger.info(
                            f"Merging overlapping {object_type} tracks (no ReID): "
                            f"{track1.track_id} and {track2.track_id} (IoU={iou:.2f})"
                        )

                if should_merge:
                    # Keep the older track (track1 if it has more hits, else track2)
                    if track1.hits >= track2.hits:
                        to_delete.add(track2.track_id)
                    else:
                        to_delete.add(track1.track_id)

        # Delete merged tracks
        for track_id in to_delete:
            if track_id in tracks:
                del tracks[track_id]
                self._stats["deleted_tracks"] += 1

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

    def get_track(self, track_id: str) -> Optional[Track]:
        """Get a track by its ID.

        Args:
            track_id: Track ID (e.g., 't_5', 'v_3')

        Returns:
            Track object or None if not found
        """
        # Check persons (t_ prefix)
        if track_id.startswith('t_') or track_id.startswith('p_'):
            return self._persons.get(track_id)
        # Check vehicles (v_ prefix)
        elif track_id.startswith('v_'):
            return self._vehicles.get(track_id)
        else:
            # No prefix - try to find by numeric ID
            # Try both person and vehicle tracks
            for tid, track in self._persons.items():
                if tid.endswith(f"_{track_id}"):
                    return track
            for tid, track in self._vehicles.items():
                if tid.endswith(f"_{track_id}"):
                    return track
        return None

    def get_stats(self, camera_id: Optional[str] = None) -> dict:
        """Get tracker statistics.

        Args:
            camera_id: Optional camera ID to filter stats by

        Returns:
            Dictionary with tracker stats including active/visible counts
        """
        # Count all tracks
        total_persons = len(self._persons)
        total_vehicles = len(self._vehicles)

        # Count visible tracks (time_since_update == 0, meaning actively detected this frame)
        visible_persons = 0
        visible_vehicles = 0

        # Count tracks by camera if specified
        camera_persons = 0
        camera_vehicles = 0
        camera_visible_persons = 0
        camera_visible_vehicles = 0

        for track in self._persons.values():
            is_visible = track.time_since_update == 0
            if is_visible:
                visible_persons += 1

            if camera_id and track.last_seen_camera == camera_id:
                camera_persons += 1
                if is_visible:
                    camera_visible_persons += 1

        for track in self._vehicles.values():
            is_visible = track.time_since_update == 0
            if is_visible:
                visible_vehicles += 1

            if camera_id and track.last_seen_camera == camera_id:
                camera_vehicles += 1
                if is_visible:
                    camera_visible_vehicles += 1

        result = {
            **self._stats,
            "persons": total_persons,
            "vehicles": total_vehicles,
            "visible_persons": visible_persons,
            "visible_vehicles": visible_vehicles,
        }

        if camera_id:
            result["camera_stats"] = {
                "camera_id": camera_id,
                "persons": camera_persons,
                "vehicles": camera_vehicles,
                "visible_persons": camera_visible_persons,
                "visible_vehicles": camera_visible_vehicles,
            }

        return result

    def reset(self):
        """Reset tracker (clear all tracks)."""
        self._persons.clear()
        self._vehicles.clear()
        # Increment session counter BEFORE resetting ID counter
        # This ensures new IDs don't collide with old ones in frontend stores
        Track.session_counter += 1
        Track.global_id_counter = 0  # Single global counter for all GIDs
        self._stats = {
            "total_tracks": 0,
            "active_tracks": 0,
            "deleted_tracks": 0,
            "gallery_matches": 0,
        }
        logger.info("BoT-SORT tracker reset")

    def _get_active_gids(self, object_type: str, camera_id: Optional[str] = None) -> set:
        """Get set of currently active GIDs for a given object type.

        Used by gallery matching to avoid re-identifying objects
        that are already being tracked IN THE SAME CAMERA.

        IMPORTANT: For cross-camera re-identification, we only exclude GIDs
        that are active in the SAME camera. This allows a person tracked in
        Camera A to be re-identified when they appear in Camera B.

        Args:
            object_type: 'person' or 'vehicle'
            camera_id: If provided, only return GIDs active in this camera

        Returns:
            Set of active GID integers
        """
        tracks = self._persons if object_type == 'person' else self._vehicles
        gids = set()

        for track_id, track in tracks.items():
            # If camera_id specified, only include GIDs from that camera
            if camera_id is not None and track.camera_id != camera_id:
                continue

            # Extract GID from track_id format: v_0_2 or p_1_5
            match = re.search(r'[vpt]_\d+_(\d+)', track_id)
            if match:
                gids.add(int(match.group(1)))

        return gids

    def get_all_active_gids(self) -> set:
        """Get all active GIDs across all object types and cameras.

        Used for stale entry cleanup - to know which GIDs are still in scene.

        Returns:
            Set of all active GID integers
        """
        person_gids = self._get_active_gids('person')
        vehicle_gids = self._get_active_gids('vehicle')
        return person_gids | vehicle_gids

    def get_all_active_track_ids(self) -> set:
        """Get all active track IDs across all object types.

        Used for analysis buffer cleanup - to identify orphan buffers.

        Returns:
            Set of all active track_id strings
        """
        track_ids = set(self._persons.keys()) | set(self._vehicles.keys())
        return track_ids

    def sync_track_to_gallery(self, track: Track):
        """Sync a track's feature to the gallery for persistent storage.

        Called after ReID features are extracted for a track.

        Args:
            track: Track with feature to sync
        """
        if not self._gallery:
            logger.debug(f"sync_track_to_gallery: no gallery attached")
            return
        if track.feature is None:
            logger.debug(f"sync_track_to_gallery: track {track.track_id} has no feature")
            return

        # Extract GID from track_id
        match = re.search(r'[vpt]_\d+_(\d+)', track.track_id)
        if not match:
            logger.debug(f"sync_track_to_gallery: could not extract GID from {track.track_id}")
            return

        gid = int(match.group(1))

        logger.info(
            f"📝 Gallery sync: {track.object_type} GID {gid} "
            f"(track={track.track_id}, feature_dim={len(track.feature)})"
        )

        self._gallery.add_or_update(
            gid=gid,
            feature=track.feature,
            object_type=track.object_type,
            camera_id=track.camera_id,
            confidence=track.confidence,
        )

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

    def clear_all(self) -> Dict[str, int]:
        """Clear ALL tracks and reset state.

        This is a full reset - increments session ID to invalidate in-flight operations.
        Returns stats on what was cleared.
        """
        self._clearing = True
        try:
            stats = {
                "persons": len(self._persons),
                "vehicles": len(self._vehicles),
                "session_id": self._session_id
            }

            self._persons.clear()
            self._vehicles.clear()

            # Reset GID counters
            self._next_person_gid = 1
            self._next_vehicle_gid = 1

            # Increment session ID to invalidate any in-flight operations
            self._session_id += 1

            # Reset stats
            self._stats = {
                "total_tracks": 0,
                "active_tracks": 0,
                "deleted_tracks": 0,
                "gallery_matches": 0,
            }

            logger.info(
                f"🗑️ Tracker cleared: {stats['persons']} persons, {stats['vehicles']} vehicles. "
                f"New session ID: {self._session_id}"
            )

            return stats
        finally:
            self._clearing = False

    def get_session_id(self) -> int:
        """Get current session ID. Used to validate in-flight operations."""
        return self._session_id

    def is_clearing(self) -> bool:
        """Check if tracker is currently being cleared."""
        return self._clearing

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


def reset_bot_sort_tracker() -> BoTSORTTracker:
    """Reset and return the global BoT-SORT tracker.

    This creates a NEW tracker instance, clearing all state from previous runs.
    Use this on startup to ensure no ghost tracks persist.
    """
    global _tracker
    _tracker = BoTSORTTracker()
    logger.info("BoT-SORT tracker RESET (new instance created)")
    return _tracker
