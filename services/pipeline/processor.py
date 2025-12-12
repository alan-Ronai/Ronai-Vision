"""Pipeline processor: unified frame processing for all cameras.

This module contains the core frame processing logic (detection -> segmentation -> reid -> tracking)
in a unified, reusable way for all cameras.
"""

import time
import numpy as np
from typing import Tuple, Optional, Dict, Any, List

from services.profiler import profiler


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


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two feature vectors."""
    if a is None or b is None:
        return 0.0
    if len(a) != len(b):
        return 0.0
    na = np.linalg.norm(a) + 1e-6
    nb = np.linalg.norm(b) + 1e-6
    return float(np.dot(a, b) / (na * nb))


class FrameProcessor:
    """Unified frame processing pipeline.

    Handles all detection, segmentation, ReID, and tracking operations
    in a single, reusable method. Used by all camera workers.
    """

    def __init__(
        self,
        detector,
        segmenter,
        reid,
        tracker,
        allowed_classes: Optional[list] = None,
        yolo_confidence: float = 0.25,
        reid_recovery: bool = True,
        recovery_confidence: float = 0.15,
        recovery_iou_thresh: float = 0.5,
        recovery_reid_thresh: float = 0.7,
        recovery_min_track_confidence: float = 0.28,
    ):
        """Initialize processor with models and configuration.

        Args:
            detector: YOLODetector instance
            segmenter: SAM2Segmenter instance
            reid: MultiClassReID dispatcher instance
            tracker: BoTSortTracker instance
            allowed_classes: list of class names to process (None = all)
            yolo_confidence: YOLO confidence threshold for primary detection
            reid_recovery: Enable ReID-based detection recovery for tracked objects
            recovery_confidence: Lower threshold for recovery detection search
            recovery_iou_thresh: IoU threshold for spatial matching with predictions
            recovery_reid_thresh: ReID similarity threshold for appearance matching
            recovery_min_track_confidence: Minimum track avg confidence for recovery eligibility
        """
        self.detector = detector
        self.segmenter = segmenter
        self.reid = reid
        self.tracker = tracker
        self.allowed_classes = allowed_classes
        self.yolo_confidence = yolo_confidence
        self.reid_recovery = reid_recovery
        self.recovery_confidence = recovery_confidence
        self.recovery_iou_thresh = recovery_iou_thresh
        self.recovery_reid_thresh = recovery_reid_thresh
        self.recovery_min_track_confidence = recovery_min_track_confidence
        self._frame_counter = 0  # Track frames for detection skipping

    def process_frame(
        self, frame: np.ndarray, force_detect: bool = False
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Process a single frame through the entire pipeline.

        Args:
            frame: (H, W, 3) BGR uint8 image

        Returns:
            (result, timing) where:
            - result: dict with tracks, masks, features, boxes, etc.
            - timing: dict with timings for each stage (detect, segment, reid, track)
        """
        timing = {}

        # Ensure frame is proper format
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame, dtype=np.uint8)
        elif frame.dtype != np.uint8:
            if frame.dtype in (np.float32, np.float64):
                frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)

        # ====================================================================
        # DETECTION WITH REID-BASED RECOVERY (with optional skipping)
        # ====================================================================
        from config.pipeline_config import PipelineConfig

        detection_skip = PipelineConfig.DETECTION_SKIP

        self._frame_counter += 1
        should_detect = (
            force_detect
            or (detection_skip == 1)
            or (self._frame_counter % detection_skip == 1)
        )

        t0 = time.time()

        if should_detect:
            # Run full detection
            with profiler.profile("yolo_detection"):
                det = self.detector.predict(frame, confidence=self.yolo_confidence)

            boxes = det.boxes
            scores = det.scores
            class_ids = det.class_ids
        else:
            # Skip detection - use empty detections (tracker will use Kalman predictions)
            boxes = np.array([], dtype=np.float32).reshape(0, 4)
            scores = np.array([], dtype=np.float32)
            class_ids = np.array([], dtype=np.int32)
            # Create minimal detection result for class names
            det = type(
                "obj",
                (object,),
                {
                    "boxes": boxes,
                    "scores": scores,
                    "class_ids": class_ids,
                    "class_names": self.detector.model.names,
                },
            )()

        # ReID-based recovery: search lower-confidence detections for tracked objects
        # Skip if no primary detections - nothing to recover
        if (
            self.reid_recovery
            and len(boxes) > 0
            and hasattr(self.tracker, "kalman_filters")
            and len(self.tracker.kalman_filters) > 0
        ):
            with profiler.profile("reid_recovery"):
                recovered_boxes, recovered_scores, recovered_class_ids = (
                    self._recover_missing_detections(
                        frame, det, boxes, class_ids, scores
                    )
                )

                if len(recovered_boxes) > 0:
                    # Deduplicate: remove recovered boxes that overlap with primary detections
                    # This prevents the same object from being counted twice
                    unique_recovered = []
                    unique_scores = []
                    unique_class_ids = []

                    for rec_box, rec_score, rec_class in zip(
                        recovered_boxes, recovered_scores, recovered_class_ids
                    ):
                        is_duplicate = False
                        for prim_box in boxes:
                            if (
                                _iou(rec_box, prim_box) > 0.5
                            ):  # High overlap = duplicate
                                is_duplicate = True
                                break

                        if not is_duplicate:
                            unique_recovered.append(rec_box)
                            unique_scores.append(rec_score)
                            unique_class_ids.append(rec_class)

                    # Merge only unique recovered detections with primary detections
                    if len(unique_recovered) > 0:
                        boxes = np.vstack([boxes, np.array(unique_recovered)])
                        scores = np.concatenate([scores, np.array(unique_scores)])
                        class_ids = np.concatenate(
                            [class_ids, np.array(unique_class_ids)]
                        )

        timing["detect"] = time.time() - t0

        # ====================================================================
        # CLASS FILTERING (optional)
        # ====================================================================
        if self.allowed_classes is not None:
            class_mask = np.isin(
                class_ids,
                [
                    i
                    for i, cn in enumerate(det.class_names)
                    if cn in self.allowed_classes
                ],
            )
            filtered_boxes = boxes[class_mask]
            filtered_class_ids = class_ids[class_mask]
            filtered_scores = scores[class_mask]
        else:
            filtered_boxes = boxes
            filtered_class_ids = class_ids
            filtered_scores = scores

        # ====================================================================
        # SEGMENTATION (OPTIONAL - CONTROLLED BY USE_SEGMENTATION ENV VAR)
        # ====================================================================
        # SAM2 segmentation is expensive (50%+ of frame time) but provides
        # detailed masks for visualization. Disabled by default.
        # Enable with: USE_SEGMENTATION=true
        t0 = time.time()
        masks = np.zeros((0, frame.shape[0], frame.shape[1]), dtype=np.uint8)

        # Import config here to avoid circular dependency
        from config.pipeline_config import PipelineConfig

        if PipelineConfig.USE_SEGMENTATION and len(filtered_boxes) > 0:
            with profiler.profile("sam2_segmentation"):
                if self.allowed_classes is not None:
                    seg = self.segmenter.segment_from_detections(
                        frame,
                        boxes=filtered_boxes,
                        class_ids=filtered_class_ids,
                        class_names=det.class_names,
                        allowed_class_names=self.allowed_classes,
                    )
                else:
                    seg = self.segmenter.segment(frame, boxes=filtered_boxes)
            masks = seg.masks
            if masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks[:, 0, :, :]
        timing["segment"] = time.time() - t0

        # ====================================================================
        # MULTI-CLASS ReID EXTRACTION (PERSON ONLY)
        # ====================================================================
        # Only extract ReID features for person class.
        # Other classes use motion-based tracking only.
        # This avoids feature dimension misalignment and improves speed.
        t0 = time.time()
        feats = None
        if len(filtered_boxes) > 0:
            # Find person-class detections only
            person_class_idx = None
            for idx, name in enumerate(det.class_names):
                if name == "person":
                    person_class_idx = idx
                    break

            if person_class_idx is not None:
                # Create mask for person-only detections
                person_mask = filtered_class_ids == person_class_idx
                person_boxes = filtered_boxes[person_mask]

                if len(person_boxes) > 0:
                    with profiler.profile("multi_class_reid"):
                        # Extract features only for person detections
                        # The CLIP/OSNet will only process person crops
                        reid_results = self.reid.extract_features(
                            frame,
                            person_boxes,
                            np.array([person_class_idx] * len(person_boxes)),
                            det.class_names,
                        )

                    if reid_results is not None:
                        # Create feature array for ALL detections
                        # Person detections get real embeddings, others get None
                        feats = [None] * len(filtered_boxes)
                        person_det_idx = 0
                        for det_idx, is_person in enumerate(person_mask):
                            if is_person:
                                # Find this person in the reid_results
                                for class_result in reid_results.values():
                                    indices = class_result["indices"]
                                    class_feats = class_result["features"]
                                    if person_det_idx in indices:
                                        feat_idx = list(indices).index(person_det_idx)
                                        feats[det_idx] = class_feats[feat_idx]
                                        break
                                person_det_idx += 1
        timing["reid"] = time.time() - t0

        # ====================================================================
        # TRACKING
        # ====================================================================
        t0 = time.time()
        with profiler.profile("botsort_tracking"):
            tracks = self.tracker.update(
                filtered_boxes, filtered_class_ids, filtered_scores, feats
            )
        timing["track"] = time.time() - t0

        return {
            "tracks": tracks,
            "masks": masks,
            "filtered_boxes": filtered_boxes,
            "filtered_class_ids": filtered_class_ids,
            "filtered_scores": filtered_scores,
            "features": feats,
            "class_names": det.class_names,
        }, timing

    def _recover_missing_detections(
        self,
        frame: np.ndarray,
        det_result,
        primary_boxes: np.ndarray,
        primary_class_ids: np.ndarray,
        primary_scores: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Recover missed detections using ReID matching with tracked objects.

        For each active track, searches lower-confidence detections in a cropped region:
        1. Crop frame around Kalman prediction (with padding for PTZ motion)
        2. Run detection on crop at lower confidence
        3. Match appearance with ReID embedding
        4. Transform coordinates back to full frame

        Args:
            frame: Original frame
            det_result: Primary detection result
            primary_boxes: Boxes detected at primary confidence
            primary_class_ids: Class IDs from primary detection
            primary_scores: Scores from primary detection

        Returns:
            Tuple of (recovered_boxes, recovered_scores, recovered_class_ids)
        """
        recovered_boxes = []
        recovered_scores = []
        recovered_class_ids = []

        # Get active tracks with features
        active_tracks = []
        for tid in self.tracker.kalman_filters.keys():
            track = self.tracker.tracks[tid]

            # Validate track confidence before recovery
            # Don't recover for tracks with low average confidence (likely false positives)
            avg_conf = track.get_avg_confidence()
            if avg_conf < self.recovery_min_track_confidence:
                continue  # Skip recovery for low-confidence tracks

            if (
                tid in self.tracker.track_features
                and self.tracker.track_features[tid] is not None
            ):
                predicted_box = self.tracker.kalman_filters[tid].get_state()
                track_class = track.class_id
                track_feat = self.tracker.track_features[tid]
                active_tracks.append((tid, predicted_box, track_class, track_feat))

        if len(active_tracks) == 0:
            return (
                np.array(recovered_boxes),
                np.array(recovered_scores),
                np.array(recovered_class_ids),
            )

        frame_h, frame_w = frame.shape[:2]

        # Process each track independently with cropped detection
        for tid, pred_box, track_class, track_feat in active_tracks:
            # Calculate crop region with PTZ-aware padding
            # Larger padding accounts for camera motion between frames
            x1, y1, x2, y2 = pred_box
            box_w = x2 - x1
            box_h = y2 - y1

            # PTZ-aware padding: 2x box size for potential camera pan/tilt
            # If camera moved, object could be 1-2 box widths away
            pad_x = box_w * 2.0
            pad_y = box_h * 2.0

            crop_x1 = int(max(0, x1 - pad_x))
            crop_y1 = int(max(0, y1 - pad_y))
            crop_x2 = int(min(frame_w, x2 + pad_x))
            crop_y2 = int(min(frame_h, y2 + pad_y))

            # Skip if crop is too small
            if crop_x2 - crop_x1 < 32 or crop_y2 - crop_y1 < 32:
                continue

            # Crop frame to search region
            crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

            # Run detection on crop at lower confidence
            try:
                crop_det = self.detector.predict(
                    crop, confidence=self.recovery_confidence
                )
            except Exception:
                # Detection failed on crop, skip
                continue

            if len(crop_det.boxes) == 0:
                continue

            # Transform crop coordinates back to full frame
            crop_boxes_full = crop_det.boxes.copy()
            crop_boxes_full[:, [0, 2]] += crop_x1  # x coordinates
            crop_boxes_full[:, [1, 3]] += crop_y1  # y coordinates

            # Check each detection in crop
            for i, (crop_box, crop_score, crop_class) in enumerate(
                zip(crop_boxes_full, crop_det.scores, crop_det.class_ids)
            ):
                # Must match class
                if int(crop_class) != int(track_class):
                    continue

                # Skip if already detected at primary confidence
                already_detected = False
                for prim_box in primary_boxes:
                    if _iou(crop_box, prim_box) > 0.7:
                        already_detected = True
                        break

                if already_detected:
                    continue

                # Must overlap with Kalman prediction
                spatial_match = _iou(crop_box, pred_box) > self.recovery_iou_thresh
                if not spatial_match:
                    continue

                # ReID matching for person class
                person_class_idx = None
                for idx, name in enumerate(det_result.class_names):
                    if name == "person":
                        person_class_idx = idx
                        break

                if person_class_idx is not None and int(crop_class) == person_class_idx:
                    try:
                        # Extract ReID on FULL frame (better context)
                        reid_results = self.reid.extract_features(
                            frame,
                            np.array([crop_box]),
                            np.array([person_class_idx]),
                            det_result.class_names,
                        )

                        if reid_results is not None:
                            for class_result in reid_results.values():
                                if len(class_result["features"]) > 0:
                                    det_feat = class_result["features"][0]
                                    similarity = _cosine_similarity(
                                        track_feat, det_feat
                                    )

                                    if similarity > self.recovery_reid_thresh:
                                        recovered_boxes.append(crop_box)
                                        recovered_scores.append(crop_score)
                                        recovered_class_ids.append(crop_class)
                                        break
                    except Exception:
                        pass
                else:
                    # Non-person: spatial match is sufficient
                    # (No ReID available, rely on motion prediction)
                    if spatial_match:
                        recovered_boxes.append(crop_box)
                        recovered_scores.append(crop_score)
                        recovered_class_ids.append(crop_class)

        if len(recovered_boxes) > 0:
            return (
                np.array(recovered_boxes),
                np.array(recovered_scores),
                np.array(recovered_class_ids),
            )
        else:
            return np.array([]), np.array([]), np.array([])
