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


def _associate_weapons_with_persons(
    person_boxes: np.ndarray,
    weapon_boxes: np.ndarray,
    weapon_class_names: List[str],
    iou_threshold: float = 0.05,
) -> Dict[int, List[Dict]]:
    """Associate weapon detections with person detections using spatial overlap.

    Args:
        person_boxes: (N, 4) array of person bounding boxes [x1, y1, x2, y2]
        weapon_boxes: (M, 4) array of weapon bounding boxes [x1, y1, x2, y2]
        weapon_class_names: List of weapon class names for each weapon box
        iou_threshold: Minimum IoU to consider association (low threshold since weapons may be small)

    Returns:
        Dict mapping person_index -> list of associated weapon dicts with 'box' and 'class' keys
    """
    associations = {}

    if len(person_boxes) == 0 or len(weapon_boxes) == 0:
        return associations

    # For each weapon, find the person with highest overlap
    for weapon_idx, weapon_box in enumerate(weapon_boxes):
        best_iou = 0.0
        best_person_idx = -1

        for person_idx, person_box in enumerate(person_boxes):
            iou = _iou(person_box, weapon_box)
            if iou > best_iou:
                best_iou = iou
                best_person_idx = person_idx

        # Associate if IoU exceeds threshold
        if best_iou >= iou_threshold and best_person_idx >= 0:
            if best_person_idx not in associations:
                associations[best_person_idx] = []

            associations[best_person_idx].append(
                {
                    "box": weapon_box,
                    "class": weapon_class_names[weapon_idx],
                    "iou": best_iou,
                }
            )

    return associations


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
        recovery_fullframe_every_n: int = 3,
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
        self.recovery_fullframe_every_n = max(1, int(recovery_fullframe_every_n))

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
            # Get class_names from detection result or detector
            class_names = getattr(det, "class_names", None)
            if not class_names:
                class_names = (
                    self.detector.get_class_names()
                    if hasattr(self.detector, "get_class_names")
                    else []
                )
            boxes = det.boxes
            scores = det.scores
            class_ids = det.class_ids
        else:
            # Skip detection - use empty detections (tracker will use Kalman predictions)
            boxes = np.array([], dtype=np.float32).reshape(0, 4)
            scores = np.array([], dtype=np.float32)
            class_ids = np.array([], dtype=np.int32)
            # Get class_names from detector for empty detection
            class_names = (
                self.detector.get_class_names()
                if hasattr(self.detector, "get_class_names")
                else []
            )
            # Create minimal detection result for class names
            det = type(
                "obj",
                (object,),
                {
                    "boxes": boxes,
                    "scores": scores,
                    "class_ids": class_ids,
                    "class_names": class_names,
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

        # Store UNFILTERED boxes for weapon association (before class filter removes weapons)
        unfiltered_boxes = boxes.copy()
        unfiltered_class_ids = class_ids.copy()

        # ====================================================================
        # CLASS FILTERING (optional)
        # ====================================================================
        if self.allowed_classes is not None:
            class_mask = np.isin(
                class_ids,
                [i for i, cn in enumerate(class_names) if cn in self.allowed_classes],
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
                        class_names=class_names,
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
            for idx, name in enumerate(class_names):
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
                            class_names,
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

        # ====================================================================
        # WEAPON-PERSON ASSOCIATION
        # ====================================================================
        # Associate weapon detections with person tracks and add alerts
        # Use UNFILTERED boxes to include weapons that may have been filtered out
        t0 = time.time()
        self._associate_weapons_and_alert(
            tracks, unfiltered_boxes, unfiltered_class_ids, class_names
        )
        timing["weapon_association"] = time.time() - t0

        return {
            "tracks": tracks,
            "masks": masks,
            "filtered_boxes": filtered_boxes,
            "filtered_class_ids": filtered_class_ids,
            "filtered_scores": filtered_scores,
            "features": feats,
            "class_names": class_names,
        }, timing

    def _associate_weapons_and_alert(
        self,
        tracks: List,
        boxes: np.ndarray,
        class_ids: np.ndarray,
        class_names: List[str],
    ):
        """Associate weapons with person tracks and emit modular actions.

        Args:
            tracks: List of Track objects from tracker
            boxes: All detection boxes
            class_ids: All detection class IDs
            class_names: List of class names
        """
        from services.tracker.metadata_manager import get_metadata_manager

        # Identify weapon and person classes
        weapon_keywords = ["pistol", "rifle", "gun", "knife", "weapon", "firearm"]
        weapon_indices = []
        weapon_boxes = []
        weapon_names = []

        person_class_id = None
        for idx, name in enumerate(class_names):
            if name == "person":
                person_class_id = idx
                break

        # Collect weapon detections
        for i, (box, class_id) in enumerate(zip(boxes, class_ids)):
            class_name = (
                class_names[int(class_id)] if int(class_id) < len(class_names) else ""
            )
            if any(keyword in class_name.lower() for keyword in weapon_keywords):
                weapon_indices.append(i)
                weapon_boxes.append(box)
                weapon_names.append(class_name)

        if len(weapon_boxes) == 0 or person_class_id is None:
            return  # No weapons or no person class defined

        weapon_boxes = np.array(weapon_boxes)

        # Get person tracks and their boxes
        person_tracks = [t for t in tracks if t.class_id == person_class_id]
        if len(person_tracks) == 0:
            return

        person_boxes = np.array([t.box for t in person_tracks])

        # Associate weapons with persons
        associations = _associate_weapons_with_persons(
            person_boxes,
            weapon_boxes,
            weapon_names,
            iou_threshold=0.05,  # Low threshold since weapons can be small
        )

        # Dispatch modular actions via ActionDispatcher
        # Fallback to old behavior if dispatcher not available
        manager = get_metadata_manager()
        try:
            from services.actions.dispatcher import get_dispatcher

            dispatcher = get_dispatcher()
        except Exception:
            dispatcher = None

        for person_idx, weapons in associations.items():
            track = person_tracks[person_idx]

            # Avoid re-triggering if already armed
            if "armed" in track.metadata.get("tags", []):
                continue

            weapon_types = [w["class"] for w in weapons]
            weapon_desc = ", ".join(set(weapon_types))

            event = {
                "type": "armed_person_detected",
                "camera_id": getattr(track, "camera_id", None),
                "track_id": track.track_id,
                "class_id": track.class_id,
                "weapon_types": weapon_types,
                "weapon_desc": weapon_desc,
                "timestamp": time.time(),
            }

            if dispatcher is not None:
                dispatcher.dispatch(event_type=event["type"], event=event, track=track)
            else:
                # Minimal fallback: tag and alert as before
                track.add_tag("armed")
                track.add_alert(
                    alert_type="armed_person",
                    message=f"Person armed with: {weapon_desc}",
                    severity="critical",
                )
                track.set_attribute("weapons_detected", weapon_types)
                track.set_attribute("weapon_detection_count", len(weapons))
                manager.update_track_metadata(
                    track.track_id, track.class_id, track.get_metadata_summary()
                )

    def _recover_missing_detections(
        self,
        frame: np.ndarray,
        det_result,
        primary_boxes: np.ndarray,
        primary_class_ids: np.ndarray,
        primary_scores: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Recover missed detections using ONE full-frame low-threshold pass + batched ReID.

        This replaces the per-track cropped detection (O(tracks) detector calls)
        with a single detector call and vectorized matching, cutting CPU/GPU usage.

        Steps:
          1. Run a single full-frame detection at `recovery_confidence`.
          2. Remove candidates that overlap primary detections (IoU > 0.5).
          3. Gate candidates per track using IoU with Kalman prediction.
          4. For person class, batch-extract ReID features and match by cosine similarity.
          5. Greedily assign best candidate per track (each candidate used once).
        """
        # Throttle recovery frequency to reduce overhead
        if (self._frame_counter % self.recovery_fullframe_every_n) != 0:
            return np.array([]), np.array([]), np.array([])

        recovered_boxes: List[np.ndarray] = []
        recovered_scores: List[float] = []
        recovered_class_ids: List[int] = []

        # Collect active tracks with features/predictions
        active_tracks = []
        for tid in self.tracker.kalman_filters.keys():
            track = self.tracker.tracks[tid]
            avg_conf = track.get_avg_confidence()
            if avg_conf < self.recovery_min_track_confidence:
                continue
            if tid in self.tracker.track_features:
                pred_box = self.tracker.kalman_filters[tid].get_state()
                active_tracks.append(
                    (
                        tid,
                        pred_box,
                        track.class_id,
                        self.tracker.track_features.get(tid, None),
                    )
                )

        if not active_tracks:
            return np.array([]), np.array([]), np.array([])

        # One full-frame low-threshold detection
        try:
            low_det = self.detector.predict(frame, confidence=self.recovery_confidence)
        except Exception:
            return np.array([]), np.array([]), np.array([])

        cand_boxes = low_det.boxes
        cand_scores = low_det.scores
        cand_class_ids = low_det.class_ids
        if len(cand_boxes) == 0:
            return np.array([]), np.array([]), np.array([])

        # Filter out candidates that overlap with primary detections
        keep_mask = []
        for cb in cand_boxes:
            dup = False
            for pb in primary_boxes:
                if _iou(cb, pb) > 0.5:
                    dup = True
                    break
            keep_mask.append(not dup)
        if not any(keep_mask):
            return np.array([]), np.array([]), np.array([])

        cand_boxes = cand_boxes[np.array(keep_mask)]
        cand_scores = cand_scores[np.array(keep_mask)]
        cand_class_ids = cand_class_ids[np.array(keep_mask)]

        # Identify indices for person class
        person_class_idx = None
        for idx, name in enumerate(det_result.class_names):
            if name == "person":
                person_class_idx = idx
                break

        # Prepare person candidate features (batch extraction)
        person_indices = (
            np.where(cand_class_ids == person_class_idx)[0]
            if person_class_idx is not None
            else np.array([], dtype=int)
        )
        person_feats_by_idx: Dict[int, np.ndarray] = {}
        if person_indices.size > 0:
            person_boxes = cand_boxes[person_indices]
            with profiler.profile("reid_recovery_batch_features"):
                reid_res = self.reid.extract_features(
                    frame,
                    person_boxes,
                    np.array([person_class_idx] * len(person_boxes)),
                    det_result.class_names,
                )
            if reid_res is not None:
                # reid_res is a dict per class; aggregate (pos, feat) pairs
                features_accum: List[Tuple[int, np.ndarray]] = []
                for class_result in reid_res.values():
                    # features are aligned to input order by 'indices'
                    idxs = list(class_result.get("indices", []))
                    feats = class_result.get("features", None)
                    if feats is None:
                        continue
                    # Map indices to features
                    for pos, feat in zip(idxs, feats):
                        features_accum.append((int(pos), feat))
                # Assign back to candidate indices
                for pos, feat in features_accum:
                    global_idx = int(person_indices[pos])
                    person_feats_by_idx[global_idx] = feat

        # Build candidate list per track with scores
        assignments: List[Tuple[int, int, float]] = []  # (track_id, cand_idx, score)
        for tid, pred_box, track_class, track_feat in active_tracks:
            best_idx = -1
            best_score = -1.0

            for ci, (cbox, cscore, cclass) in enumerate(
                zip(cand_boxes, cand_scores, cand_class_ids)
            ):
                # Must match class
                if int(cclass) != int(track_class):
                    continue

                # Spatial gate
                iou = _iou(cbox, pred_box)
                if iou < self.recovery_iou_thresh:
                    continue

                score = float(iou)

                # If person, require appearance match
                if person_class_idx is not None and int(cclass) == person_class_idx:
                    cand_feat = person_feats_by_idx.get(ci)
                    if cand_feat is None or track_feat is None:
                        continue
                    sim = _cosine_similarity(track_feat, cand_feat)
                    if sim < self.recovery_reid_thresh:
                        continue
                    # Combine similarity and detector score (weighted)
                    score = 0.7 * sim + 0.3 * float(cscore)

                if score > best_score:
                    best_score = score
                    best_idx = ci

            if best_idx >= 0:
                assignments.append((tid, best_idx, best_score))

        if not assignments:
            return np.array([]), np.array([]), np.array([])

        # Greedy resolution: highest score first, each candidate used once
        assignments.sort(key=lambda x: x[2], reverse=True)
        used_cands = set()
        used_tracks = set()
        for tid, ci, sc in assignments:
            if ci in used_cands or tid in used_tracks:
                continue
            used_cands.add(ci)
            used_tracks.add(tid)
            recovered_boxes.append(cand_boxes[ci])
            recovered_scores.append(cand_scores[ci])
            recovered_class_ids.append(cand_class_ids[ci])

        if recovered_boxes:
            return (
                np.array(recovered_boxes),
                np.array(recovered_scores),
                np.array(recovered_class_ids),
            )
        return np.array([]), np.array([]), np.array([])
