"""Parallel Detection Pipeline - Multi-worker YOLO detection with ordered output.

Architecture:
                         PHASE 1: Bootstrap (first N frames)
                         ┌─────────────────────────────┐
                         │   Single-threaded tracking  │
                         │   (establish stable tracks) │
                         └──────────────┬──────────────┘
                                        │
                         PHASE 2: Parallel Mode (after tracks established)
           ┌────────────────────────────┼────────────────────────────┐
           ▼                            ▼                            ▼
    ┌─────────────┐              ┌─────────────┐              ┌─────────────┐
    │  Worker A   │              │  Worker B   │              │  Worker C   │
    │  YOLO only  │              │  YOLO only  │              │  YOLO only  │
    └──────┬──────┘              └──────┬──────┘              └──────┬──────┘
           │                            │                            │
           └────────────────────────────┼────────────────────────────┘
                                        ▼
                              ┌─────────────────────┐
                              │   Detection Merger  │
                              │  (dedup by IoU/ReID)│
                              └──────────┬──────────┘
                                         ▼
                              ┌─────────────────────┐
                              │  Central Tracker    │
                              │  (BoT-SORT + ReID)  │
                              │  (Global ID Store)  │
                              └──────────┬──────────┘
                                         ▼
                              ┌─────────────────────┐
                              │   ReorderBuffer     │
                              │  (ordered output)   │
                              └─────────────────────┘

Key Features:
- Bootstrap phase: First N frames run sequentially to establish stable tracks
- Parallel YOLO: Multiple workers run YOLO detection concurrently
- Detection Merger: Deduplicates detections across workers using IoU + ReID
- Central Tracker: Single BoT-SORT instance maintains global track IDs
- ReorderBuffer: Ensures frames are output in correct order for streaming
- Global ReID Store: Centralized feature storage, accessed only by central tracker
"""

import logging
import threading
import time
import numpy as np
from queue import Queue, Empty, PriorityQueue
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable, Set
from concurrent.futures import ThreadPoolExecutor, Future
from collections import defaultdict
import os

from .detection import Detection

logger = logging.getLogger(__name__)

# Configuration
BOOTSTRAP_FRAMES = int(os.environ.get("PARALLEL_BOOTSTRAP_FRAMES", "30"))  # Frames before parallel mode
NUM_WORKERS = int(os.environ.get("PARALLEL_NUM_WORKERS", "3"))  # Max number of YOLO workers
FRAME_ID_CYCLE = int(os.environ.get("PARALLEL_FRAME_ID_CYCLE", "1000"))  # Frame ID reset cycle
MERGE_IOU_THRESHOLD = float(os.environ.get("PARALLEL_MERGE_IOU", "0.5"))  # IoU threshold for merging
MERGE_REID_THRESHOLD = float(os.environ.get("PARALLEL_MERGE_REID", "0.7"))  # ReID threshold for merging
REORDER_BUFFER_SIZE = int(os.environ.get("PARALLEL_REORDER_SIZE", "10"))  # Max frames to buffer for reordering
REORDER_TIMEOUT_MS = int(os.environ.get("PARALLEL_REORDER_TIMEOUT_MS", "200"))  # Max wait for missing frame

# Dynamic scaling configuration
SCALE_UP_AFTER_FRAMES = int(os.environ.get("PARALLEL_SCALE_UP_FRAMES", "50"))  # Frames before adding more workers
SCALE_UP_INTERVAL_FRAMES = int(os.environ.get("PARALLEL_SCALE_INTERVAL", "30"))  # Frames between each scale-up


@dataclass
class FrameJob:
    """A frame to be processed by a worker."""
    frame_id: int  # Sequential ID within cycle
    camera_id: str
    frame: np.ndarray
    timestamp: float

    def __lt__(self, other):
        """For priority queue ordering."""
        return self.frame_id < other.frame_id


@dataclass
class DetectionBatch:
    """Detections from a single frame, from a single worker."""
    frame_id: int
    camera_id: str
    timestamp: float
    frame: np.ndarray
    vehicle_detections: List[Detection] = field(default_factory=list)
    person_detections: List[Detection] = field(default_factory=list)
    worker_id: int = 0
    processing_time_ms: float = 0.0

    def __lt__(self, other):
        """For priority queue ordering."""
        return self.frame_id < other.frame_id


@dataclass
class MergedDetections:
    """Deduplicated detections ready for tracking."""
    frame_id: int
    camera_id: str
    timestamp: float
    frame: np.ndarray
    vehicle_detections: List[Detection] = field(default_factory=list)
    person_detections: List[Detection] = field(default_factory=list)
    merge_stats: Dict[str, int] = field(default_factory=dict)


class DetectionMerger:
    """Merges and deduplicates detections from multiple workers.

    When multiple workers process overlapping frames, the same object may be
    detected multiple times with slightly different bounding boxes. This class
    merges duplicate detections using:
    1. IoU (Intersection over Union) - spatial overlap
    2. ReID similarity - appearance matching (when features available)
    """

    def __init__(
        self,
        iou_threshold: float = MERGE_IOU_THRESHOLD,
        reid_threshold: float = MERGE_REID_THRESHOLD,
    ):
        self.iou_threshold = iou_threshold
        self.reid_threshold = reid_threshold
        self._stats = {
            "total_input": 0,
            "total_output": 0,
            "merged_by_iou": 0,
            "merged_by_reid": 0,
        }

    def merge(self, batches: List[DetectionBatch]) -> MergedDetections:
        """Merge detections from multiple workers for the same frame.

        Args:
            batches: List of detection batches from different workers

        Returns:
            MergedDetections with deduplicated detections
        """
        if not batches:
            raise ValueError("No batches to merge")

        # Use first batch as reference for frame info
        ref = batches[0]

        # Collect all detections
        all_vehicles: List[Detection] = []
        all_persons: List[Detection] = []

        for batch in batches:
            all_vehicles.extend(batch.vehicle_detections)
            all_persons.extend(batch.person_detections)

        self._stats["total_input"] += len(all_vehicles) + len(all_persons)

        # Deduplicate
        merged_vehicles = self._deduplicate(all_vehicles)
        merged_persons = self._deduplicate(all_persons)

        self._stats["total_output"] += len(merged_vehicles) + len(merged_persons)

        return MergedDetections(
            frame_id=ref.frame_id,
            camera_id=ref.camera_id,
            timestamp=ref.timestamp,
            frame=ref.frame,
            vehicle_detections=merged_vehicles,
            person_detections=merged_persons,
            merge_stats={
                "input_vehicles": len(all_vehicles),
                "output_vehicles": len(merged_vehicles),
                "input_persons": len(all_persons),
                "output_persons": len(merged_persons),
            }
        )

    def _deduplicate(self, detections: List[Detection]) -> List[Detection]:
        """Remove duplicate detections using IoU and optionally ReID.

        Uses greedy NMS-style approach:
        1. Sort by confidence (highest first)
        2. For each detection, check if it overlaps with any kept detection
        3. If overlap > threshold, merge (keep higher confidence, average bbox)
        4. If no overlap, add to kept list
        """
        if not detections:
            return []

        # Sort by confidence descending
        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)

        kept: List[Detection] = []

        for det in sorted_dets:
            should_keep = True
            merge_target = None

            for i, kept_det in enumerate(kept):
                iou = self._compute_iou(det.bbox, kept_det.bbox)

                if iou >= self.iou_threshold:
                    # High IoU - likely same object
                    should_keep = False
                    merge_target = i
                    self._stats["merged_by_iou"] += 1
                    break

                # If both have ReID features, check similarity
                if det.feature is not None and kept_det.feature is not None:
                    similarity = self._compute_cosine_similarity(det.feature, kept_det.feature)
                    if similarity >= self.reid_threshold and iou >= 0.3:  # Lower IoU with ReID match
                        should_keep = False
                        merge_target = i
                        self._stats["merged_by_reid"] += 1
                        break

            if should_keep:
                kept.append(det)
            elif merge_target is not None:
                # Merge: average the bboxes, keep higher confidence
                kept[merge_target] = self._merge_detections(kept[merge_target], det)

        return kept

    def _merge_detections(self, det1: Detection, det2: Detection) -> Detection:
        """Merge two detections into one.

        Takes the higher confidence detection as base, averages bboxes.
        """
        # Keep higher confidence as primary
        if det2.confidence > det1.confidence:
            det1, det2 = det2, det1

        # Average the bboxes (weighted by confidence)
        w1 = det1.confidence / (det1.confidence + det2.confidence)
        w2 = det2.confidence / (det1.confidence + det2.confidence)

        x1 = det1.bbox[0] * w1 + det2.bbox[0] * w2
        y1 = det1.bbox[1] * w1 + det2.bbox[1] * w2
        w = det1.bbox[2] * w1 + det2.bbox[2] * w2
        h = det1.bbox[3] * w1 + det2.bbox[3] * w2

        return Detection(
            bbox=(x1, y1, w, h),
            confidence=det1.confidence,  # Keep higher confidence
            class_id=det1.class_id,
            class_name=det1.class_name,
            feature=det1.feature if det1.feature is not None else det2.feature,
        )

    @staticmethod
    def _compute_iou(bbox1: Tuple, bbox2: Tuple) -> float:
        """Compute IoU between two bboxes in (x, y, w, h) format."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Convert to xyxy
        x1_min, y1_min = x1, y1
        x1_max, y1_max = x1 + w1, y1 + h1
        x2_min, y2_min = x2, y2
        x2_max, y2_max = x2 + w2, y2 + h2

        # Intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

        # Union
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area

        if union_area <= 0:
            return 0.0

        return inter_area / union_area

    @staticmethod
    def _compute_cosine_similarity(feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Compute cosine similarity between two feature vectors."""
        norm1 = np.linalg.norm(feat1)
        norm2 = np.linalg.norm(feat2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(feat1, feat2) / (norm1 * norm2)

    def get_stats(self) -> Dict[str, Any]:
        """Get merger statistics."""
        return self._stats.copy()


class ReorderBuffer:
    """Buffer that reorders frames to ensure sequential output.

    Frames may arrive out of order from parallel workers. This buffer
    holds frames until all previous frames have been processed, then
    releases them in order.

    Features:
    - Timeout: If a frame is missing too long, skip it
    - Max buffer size: Prevent memory issues
    - Cycle awareness: Handles frame ID wraparound
    """

    def __init__(
        self,
        max_size: int = REORDER_BUFFER_SIZE,
        timeout_ms: int = REORDER_TIMEOUT_MS,
        cycle_size: int = FRAME_ID_CYCLE,
    ):
        self.max_size = max_size
        self.timeout_ms = timeout_ms
        self.cycle_size = cycle_size

        self._buffer: Dict[int, Any] = {}  # frame_id -> result
        self._next_frame_id = 0  # Next frame ID to emit
        self._lock = threading.Lock()
        self._waiting_since: Dict[int, float] = {}  # frame_id -> wait start time

        self._stats = {
            "frames_buffered": 0,
            "frames_emitted": 0,
            "frames_skipped": 0,  # Skipped due to timeout
            "max_buffer_used": 0,
        }

    def add(self, frame_id: int, result: Any) -> List[Any]:
        """Add a processed frame result and return any frames ready for emission.

        Args:
            frame_id: The frame ID
            result: The processed result for this frame

        Returns:
            List of results ready to be emitted (in order)
        """
        with self._lock:
            # Handle cycle wraparound
            if frame_id < self._next_frame_id and \
               self._next_frame_id - frame_id > self.cycle_size // 2:
                # This is a new cycle, reset
                logger.info(f"ReorderBuffer: Detected cycle wraparound at frame {frame_id}")
                self._next_frame_id = frame_id
                self._buffer.clear()
                self._waiting_since.clear()

            # Store the result
            self._buffer[frame_id] = result
            self._stats["frames_buffered"] += 1
            self._stats["max_buffer_used"] = max(
                self._stats["max_buffer_used"],
                len(self._buffer)
            )

            # Try to emit consecutive frames
            ready = []
            while self._next_frame_id in self._buffer:
                ready.append(self._buffer.pop(self._next_frame_id))
                if self._next_frame_id in self._waiting_since:
                    del self._waiting_since[self._next_frame_id]
                self._next_frame_id = (self._next_frame_id + 1) % self.cycle_size
                self._stats["frames_emitted"] += 1

            # Check for timeouts on waiting frames
            if not ready and self._next_frame_id not in self._buffer:
                now = time.time()
                if self._next_frame_id not in self._waiting_since:
                    self._waiting_since[self._next_frame_id] = now
                elif (now - self._waiting_since[self._next_frame_id]) * 1000 > self.timeout_ms:
                    # Timeout - skip this frame
                    logger.warning(
                        f"ReorderBuffer: Skipping frame {self._next_frame_id} (timeout)"
                    )
                    del self._waiting_since[self._next_frame_id]
                    self._next_frame_id = (self._next_frame_id + 1) % self.cycle_size
                    self._stats["frames_skipped"] += 1

                    # Try again with next frame
                    while self._next_frame_id in self._buffer:
                        ready.append(self._buffer.pop(self._next_frame_id))
                        self._next_frame_id = (self._next_frame_id + 1) % self.cycle_size
                        self._stats["frames_emitted"] += 1

            # Prevent buffer overflow
            if len(self._buffer) > self.max_size:
                # Skip oldest waiting frame
                oldest = min(self._buffer.keys())
                logger.warning(
                    f"ReorderBuffer: Buffer overflow, skipping frame {oldest}"
                )
                del self._buffer[oldest]
                if oldest in self._waiting_since:
                    del self._waiting_since[oldest]
                self._stats["frames_skipped"] += 1

            return ready

    def reset(self, next_frame_id: int = 0):
        """Reset the buffer state."""
        with self._lock:
            self._buffer.clear()
            self._waiting_since.clear()
            self._next_frame_id = next_frame_id

    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        with self._lock:
            return {
                **self._stats,
                "current_buffer_size": len(self._buffer),
                "next_frame_id": self._next_frame_id,
                "waiting_frames": list(self._waiting_since.keys()),
            }


class YOLOWorker:
    """A worker that runs YOLO detection on frames.

    Each worker can process frames independently. The worker does NOT
    run tracking - only detection. Tracking is centralized to maintain
    consistent global IDs.
    """

    def __init__(
        self,
        worker_id: int,
        yolo_model,
        class_confidence: Dict[str, float],
        allowed_classes: set,
        min_box_area: int = 500,
    ):
        self.worker_id = worker_id
        self.yolo = yolo_model
        self.class_confidence = class_confidence
        self.allowed_classes = allowed_classes
        self.min_box_area = min_box_area

        self._stats = {
            "frames_processed": 0,
            "detections": 0,
            "avg_process_time_ms": 0.0,
        }
        self._process_times: List[float] = []

    def process(self, job: FrameJob) -> DetectionBatch:
        """Run YOLO detection on a frame.

        Args:
            job: The frame job to process

        Returns:
            DetectionBatch with vehicle and person detections
        """
        start_time = time.time()

        # Run YOLO
        base_confidence = min(self.class_confidence.values()) if self.class_confidence else 0.35

        results = self.yolo(
            job.frame,
            verbose=False,
            conf=base_confidence,
            iou=0.4,
            max_det=50,
            agnostic_nms=True,
        )[0]

        # Parse detections
        vehicle_detections = []
        person_detections = []
        vehicle_classes = ["car", "truck", "bus", "motorcycle", "bicycle"]

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = results.names[cls]
            xyxy = box.xyxy[0].tolist()

            # Filter by allowed classes
            if label not in self.allowed_classes:
                continue

            # Per-class confidence threshold
            min_conf = self.class_confidence.get(label, base_confidence)
            if conf < min_conf:
                continue

            # Convert to xywh
            x1, y1, x2, y2 = xyxy
            bbox = (x1, y1, x2 - x1, y2 - y1)

            # Filter tiny boxes
            if (x2 - x1) * (y2 - y1) < self.min_box_area:
                continue

            det = Detection(
                bbox=bbox,
                confidence=conf,
                class_id=cls,
                class_name=label,
                feature=None,  # ReID done centrally after merge
            )

            if label in vehicle_classes:
                vehicle_detections.append(det)
            elif label == "person":
                person_detections.append(det)

        # Update stats
        process_time = (time.time() - start_time) * 1000
        self._process_times.append(process_time)
        if len(self._process_times) > 100:
            self._process_times.pop(0)
        self._stats["frames_processed"] += 1
        self._stats["detections"] += len(vehicle_detections) + len(person_detections)
        self._stats["avg_process_time_ms"] = sum(self._process_times) / len(self._process_times)

        return DetectionBatch(
            frame_id=job.frame_id,
            camera_id=job.camera_id,
            timestamp=job.timestamp,
            frame=job.frame,
            vehicle_detections=vehicle_detections,
            person_detections=person_detections,
            worker_id=self.worker_id,
            processing_time_ms=process_time,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        return {
            "worker_id": self.worker_id,
            **self._stats,
        }


class ParallelDetectionPipeline:
    """Orchestrates parallel YOLO detection with centralized tracking.

    Three phases with dynamic scaling:
    1. Bootstrap (frames 0-N): Single worker, sequential processing to establish tracks
    2. Scaling (frames N-M): Gradually add workers as tracks stabilize
    3. Full Parallel (frames M+): All workers active, round-robin distribution

    Features:
    - Dynamic worker scaling (start with 1, scale up gradually)
    - Automatic phase transition based on frame count
    - Detection merging to handle duplicates from overlapping processing
    - Ordered output via reorder buffer
    - Centralized tracking maintains global IDs
    - ReID feature extraction happens once (after merge, before tracking)
    """

    def __init__(
        self,
        yolo_model,
        class_confidence: Dict[str, float],
        allowed_classes: set,
        max_workers: int = NUM_WORKERS,
        bootstrap_frames: int = BOOTSTRAP_FRAMES,
        scale_up_after: int = SCALE_UP_AFTER_FRAMES,
        scale_interval: int = SCALE_UP_INTERVAL_FRAMES,
        on_result: Optional[Callable] = None,
    ):
        self.max_workers = max_workers
        self.bootstrap_frames = bootstrap_frames
        self.scale_up_after = scale_up_after
        self.scale_interval = scale_interval
        self.on_result = on_result

        # Store model info for creating workers on-demand
        self._yolo_model = yolo_model
        self._class_confidence = class_confidence
        self._allowed_classes = allowed_classes

        # Start with single worker - more will be added dynamically
        self.workers: List[YOLOWorker] = [
            YOLOWorker(
                worker_id=0,
                yolo_model=yolo_model,
                class_confidence=class_confidence,
                allowed_classes=allowed_classes,
            )
        ]
        self._active_workers = 1  # Currently active workers

        # Thread pool sized for max workers (workers added dynamically)
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="yolo_worker")

        # Detection merger
        self.merger = DetectionMerger()

        # Reorder buffer
        self.reorder_buffer = ReorderBuffer()

        # State
        self._frame_counter: Dict[str, int] = defaultdict(int)  # Per-camera frame counter
        self._in_parallel_mode: Dict[str, bool] = defaultdict(bool)  # Per-camera mode
        self._pending_jobs: Dict[int, List[Future]] = {}  # frame_id -> list of futures
        self._lock = threading.Lock()

        # Stats
        self._stats = {
            "total_frames": 0,
            "bootstrap_frames": 0,
            "parallel_frames": 0,
            "parallel_batches": 0,
            "current_workers": 1,
            "scale_ups": 0,
        }

        logger.info(
            f"ParallelDetectionPipeline initialized: "
            f"max_workers={max_workers}, bootstrap={bootstrap_frames} frames, "
            f"scale_up_after={scale_up_after}, scale_interval={scale_interval}"
        )

    def _maybe_scale_up(self, frame_id: int):
        """Check if we should add more workers based on frame count.

        Scaling strategy:
        - After bootstrap_frames: Enter parallel mode with 1 worker
        - After scale_up_after frames: Start adding workers
        - Every scale_interval frames: Add 1 more worker until max_workers
        """
        if self._active_workers >= self.max_workers:
            return  # Already at max

        if frame_id < self.scale_up_after:
            return  # Not ready to scale yet

        # Calculate how many workers we should have based on frame count
        frames_since_scale_start = frame_id - self.scale_up_after
        target_workers = min(
            self.max_workers,
            1 + (frames_since_scale_start // self.scale_interval) + 1
        )

        if target_workers > self._active_workers:
            self._add_worker()

    def _add_worker(self):
        """Add a new worker to the pool."""
        with self._lock:
            if self._active_workers >= self.max_workers:
                return

            new_worker_id = len(self.workers)
            new_worker = YOLOWorker(
                worker_id=new_worker_id,
                yolo_model=self._yolo_model,
                class_confidence=self._class_confidence,
                allowed_classes=self._allowed_classes,
            )
            self.workers.append(new_worker)
            self._active_workers = len(self.workers)
            self._stats["current_workers"] = self._active_workers
            self._stats["scale_ups"] += 1

            logger.info(
                f"🚀 Scaled up to {self._active_workers} workers "
                f"(max={self.max_workers})"
            )

    def submit_frame(
        self,
        camera_id: str,
        frame: np.ndarray,
        timestamp: float,
    ) -> Optional[int]:
        """Submit a frame for processing.

        In bootstrap mode: Processes synchronously with single worker
        In parallel mode: Distributes to workers based on current scale

        Args:
            camera_id: Camera identifier
            frame: The frame to process
            timestamp: Frame timestamp

        Returns:
            Frame ID if queued for async processing, None if processed sync
        """
        with self._lock:
            frame_id = self._frame_counter[camera_id]
            self._frame_counter[camera_id] = (frame_id + 1) % FRAME_ID_CYCLE
            self._stats["total_frames"] += 1

        job = FrameJob(
            frame_id=frame_id,
            camera_id=camera_id,
            frame=frame,
            timestamp=timestamp,
        )

        # Check if we should be in parallel mode
        if not self._in_parallel_mode[camera_id]:
            if frame_id >= self.bootstrap_frames:
                self._in_parallel_mode[camera_id] = True
                logger.info(
                    f"Camera {camera_id}: Entering parallel mode after "
                    f"{self.bootstrap_frames} bootstrap frames (1 worker active)"
                )

        # Check if we should scale up workers
        self._maybe_scale_up(frame_id)

        if not self._in_parallel_mode[camera_id]:
            # Bootstrap mode: process with single worker synchronously
            self._stats["bootstrap_frames"] += 1
            batch = self.workers[0].process(job)
            return self._handle_batch(batch)
        else:
            # Parallel mode: distribute to active workers
            self._stats["parallel_frames"] += 1
            self._submit_parallel(job)
            return frame_id

    def _submit_parallel(self, job: FrameJob):
        """Submit a job to workers using round-robin distribution.

        Uses only the currently active workers (dynamic scaling).
        """
        # Round-robin across active workers only
        worker_idx = job.frame_id % self._active_workers

        future = self._executor.submit(self.workers[worker_idx].process, job)
        future.add_done_callback(lambda f: self._on_worker_done(job.frame_id, f))

        with self._lock:
            if job.frame_id not in self._pending_jobs:
                self._pending_jobs[job.frame_id] = []
            self._pending_jobs[job.frame_id].append(future)

    def _on_worker_done(self, frame_id: int, future: Future):
        """Callback when a worker completes processing."""
        try:
            batch = future.result()
            self._stats["parallel_batches"] += 1

            # Check if all workers for this frame are done
            with self._lock:
                if frame_id in self._pending_jobs:
                    self._pending_jobs[frame_id].remove(future)
                    remaining = len(self._pending_jobs[frame_id])

                    if remaining == 0:
                        del self._pending_jobs[frame_id]

            # For round-robin (single worker per frame), handle immediately
            self._handle_batch(batch)

        except Exception as e:
            logger.error(f"Worker error for frame {frame_id}: {e}")

    def _handle_batch(self, batch: DetectionBatch) -> Optional[int]:
        """Handle a completed detection batch.

        In current implementation (round-robin), each frame has one batch.
        If using multi-worker redundancy, would collect batches and merge.
        """
        # Add to reorder buffer and get ordered results
        ready_results = self.reorder_buffer.add(batch.frame_id, batch)

        # Emit ready results in order
        for result in ready_results:
            if self.on_result:
                self.on_result(result)

        return batch.frame_id

    def get_merged_detections(
        self,
        batches: List[DetectionBatch],
    ) -> MergedDetections:
        """Merge detections from multiple batches (for multi-worker mode).

        Used when multiple workers process the same frame for redundancy.
        """
        return self.merger.merge(batches)

    def reset_camera(self, camera_id: str):
        """Reset state for a camera (e.g., when camera disconnects)."""
        with self._lock:
            self._frame_counter[camera_id] = 0
            self._in_parallel_mode[camera_id] = False

        self.reorder_buffer.reset()
        logger.info(f"Camera {camera_id}: Reset to bootstrap mode")

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            **self._stats,
            "workers": [w.get_stats() for w in self.workers],
            "merger": self.merger.get_stats(),
            "reorder_buffer": self.reorder_buffer.get_stats(),
            "cameras": {
                cam_id: {
                    "frame_count": count,
                    "in_parallel_mode": self._in_parallel_mode[cam_id],
                }
                for cam_id, count in self._frame_counter.items()
            },
        }

    def shutdown(self):
        """Shutdown the pipeline."""
        self._executor.shutdown(wait=False)
        logger.info("ParallelDetectionPipeline shutdown")


# Singleton instance
_parallel_pipeline: Optional[ParallelDetectionPipeline] = None


def get_parallel_pipeline() -> Optional[ParallelDetectionPipeline]:
    """Get the global parallel pipeline instance."""
    return _parallel_pipeline


def init_parallel_pipeline(
    yolo_model,
    class_confidence: Dict[str, float],
    allowed_classes: set,
    max_workers: int = NUM_WORKERS,
    bootstrap_frames: int = BOOTSTRAP_FRAMES,
    scale_up_after: int = SCALE_UP_AFTER_FRAMES,
    scale_interval: int = SCALE_UP_INTERVAL_FRAMES,
    on_result: Optional[Callable] = None,
) -> ParallelDetectionPipeline:
    """Initialize the global parallel pipeline."""
    global _parallel_pipeline

    if _parallel_pipeline is not None:
        _parallel_pipeline.shutdown()

    _parallel_pipeline = ParallelDetectionPipeline(
        yolo_model=yolo_model,
        class_confidence=class_confidence,
        allowed_classes=allowed_classes,
        max_workers=max_workers,
        bootstrap_frames=bootstrap_frames,
        scale_up_after=scale_up_after,
        scale_interval=scale_interval,
        on_result=on_result,
    )

    return _parallel_pipeline


class TrackMerger:
    """Merges and deduplicates new tracks that may represent the same object.

    When parallel workers process frames, a new object entering the scene may be
    detected by multiple workers before tracking stabilizes. This results in
    multiple track IDs being assigned to the same physical object.

    This class detects and merges such duplicate tracks using:
    1. Temporal proximity - tracks created within N frames of each other
    2. Spatial proximity - tracks with overlapping bounding boxes
    3. ReID similarity - tracks with similar appearance features

    When duplicates are found, the older track ID is preserved and the newer
    track is redirected to it (track ID remapping).
    """

    def __init__(
        self,
        max_age_diff_frames: int = 5,  # Max frame difference for merge candidates
        iou_threshold: float = 0.4,  # Spatial overlap threshold
        reid_threshold: float = 0.7,  # Appearance similarity threshold
    ):
        self.max_age_diff_frames = max_age_diff_frames
        self.iou_threshold = iou_threshold
        self.reid_threshold = reid_threshold

        # Track ID remapping: new_id -> canonical_id
        self._remap: Dict[int, int] = {}

        # Recent tracks for comparison
        self._recent_tracks: Dict[str, List[Dict]] = defaultdict(list)  # class -> list of tracks
        self._recent_tracks_max = 20  # Max recent tracks to keep per class

        self._stats = {
            "tracks_merged": 0,
            "merge_by_iou": 0,
            "merge_by_reid": 0,
            "remaps_active": 0,
        }
        self._lock = threading.Lock()

    def check_and_merge(
        self,
        new_track_id: int,
        class_name: str,
        bbox: Tuple[float, float, float, float],
        feature: Optional[np.ndarray],
        frame_id: int,
    ) -> int:
        """Check if a new track should be merged with an existing one.

        Args:
            new_track_id: The new track ID assigned by tracker
            class_name: Object class ("person", "car", etc.)
            bbox: Bounding box (x, y, w, h)
            feature: ReID feature vector (optional)
            frame_id: Current frame ID

        Returns:
            The canonical track ID (may be different from new_track_id if merged)
        """
        with self._lock:
            # Check if already remapped
            if new_track_id in self._remap:
                return self._remap[new_track_id]

            # Look for merge candidates among recent tracks of same class
            class_key = class_name if class_name != "person" else "person"
            if class_name in ["car", "truck", "bus", "motorcycle", "bicycle"]:
                class_key = "vehicle"

            candidates = self._recent_tracks.get(class_key, [])

            for candidate in candidates:
                # Skip self
                if candidate["track_id"] == new_track_id:
                    continue

                # Check frame age difference
                age_diff = abs(frame_id - candidate["frame_id"])
                if age_diff > self.max_age_diff_frames:
                    continue

                # Check spatial overlap
                iou = DetectionMerger._compute_iou(bbox, candidate["bbox"])
                if iou >= self.iou_threshold:
                    # Merge based on IoU
                    self._merge_tracks(new_track_id, candidate["track_id"], "iou")
                    return candidate["track_id"]

                # Check ReID similarity if both have features
                if feature is not None and candidate.get("feature") is not None:
                    similarity = DetectionMerger._compute_cosine_similarity(
                        feature, candidate["feature"]
                    )
                    if similarity >= self.reid_threshold and iou >= 0.2:
                        # Merge based on ReID
                        self._merge_tracks(new_track_id, candidate["track_id"], "reid")
                        return candidate["track_id"]

            # No merge - add to recent tracks
            self._add_recent_track(class_key, {
                "track_id": new_track_id,
                "bbox": bbox,
                "feature": feature,
                "frame_id": frame_id,
            })

            return new_track_id

    def _merge_tracks(self, new_id: int, canonical_id: int, method: str):
        """Record a track merge."""
        self._remap[new_id] = canonical_id
        self._stats["tracks_merged"] += 1
        self._stats["remaps_active"] = len(self._remap)

        if method == "iou":
            self._stats["merge_by_iou"] += 1
        else:
            self._stats["merge_by_reid"] += 1

        logger.info(
            f"TrackMerger: Merged track {new_id} -> {canonical_id} (method={method})"
        )

    def _add_recent_track(self, class_key: str, track_info: Dict):
        """Add a track to recent tracks list."""
        recent = self._recent_tracks[class_key]
        recent.append(track_info)

        # Trim if too many
        if len(recent) > self._recent_tracks_max:
            recent.pop(0)

    def get_canonical_id(self, track_id: int) -> int:
        """Get the canonical (merged) track ID for a given ID."""
        with self._lock:
            return self._remap.get(track_id, track_id)

    def clear_old_remaps(self, current_active_ids: set):
        """Remove remaps for tracks that are no longer active."""
        with self._lock:
            old_remaps = {k: v for k, v in self._remap.items()
                         if k not in current_active_ids and v not in current_active_ids}
            for k in old_remaps:
                del self._remap[k]
            self._stats["remaps_active"] = len(self._remap)

    def get_stats(self) -> Dict[str, Any]:
        """Get merger statistics."""
        with self._lock:
            return self._stats.copy()


class ParallelDetectionIntegration:
    """Integrates parallel YOLO detection with centralized tracking and ReID.

    This class bridges the parallel detection pipeline with the existing
    DetectionLoop infrastructure:

    1. Parallel YOLO workers generate raw detections
    2. Detections are merged/deduplicated
    3. Central BoT-SORT tracker assigns/maintains track IDs
    4. ReID features are extracted for new tracks (centralized)
    5. Track merging handles duplicate tracks from parallel processing
    6. Results flow to frame selection, Gemini analysis, etc.

    Key invariants maintained:
    - Global track IDs are unique and consistent
    - ReID feature store remains centralized
    - Frame ordering is preserved for streaming
    - All existing hooks (scenarios, rules, etc.) work unchanged
    """

    def __init__(
        self,
        detection_loop,  # Reference to parent DetectionLoop
        max_workers: int = NUM_WORKERS,
        bootstrap_frames: int = BOOTSTRAP_FRAMES,
        scale_up_after: int = SCALE_UP_AFTER_FRAMES,
        scale_interval: int = SCALE_UP_INTERVAL_FRAMES,
    ):
        self.detection_loop = detection_loop
        self.max_workers = max_workers
        self.bootstrap_frames = bootstrap_frames
        self.scale_up_after = scale_up_after
        self.scale_interval = scale_interval

        # Parallel pipeline (YOLO workers)
        self.pipeline: Optional[ParallelDetectionPipeline] = None

        # Track merger for handling duplicate tracks from parallel processing
        self.track_merger = TrackMerger()

        # Per-camera reorder buffers for central tracking
        self._tracking_buffers: Dict[str, ReorderBuffer] = {}

        # Results queue for async result handling
        self._result_queue: Queue = Queue(maxsize=50)

        # Threading
        self._tracking_thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

        # Stats
        self._stats = {
            "frames_received": 0,
            "frames_tracked": 0,
            "tracking_time_ms": 0.0,
        }

        logger.info(
            f"ParallelDetectionIntegration initialized: "
            f"max_workers={max_workers}, bootstrap={bootstrap_frames}, "
            f"scale_up_after={scale_up_after}, scale_interval={scale_interval}"
        )

    def initialize(self):
        """Initialize the parallel pipeline with the detection loop's YOLO model."""
        if self.detection_loop.yolo is None:
            raise RuntimeError("DetectionLoop YOLO model not initialized")

        # Create parallel pipeline with dynamic scaling
        self.pipeline = ParallelDetectionPipeline(
            yolo_model=self.detection_loop.yolo,
            class_confidence=self.detection_loop.class_confidence,
            allowed_classes={"person", "car", "truck", "bus", "motorcycle", "bicycle"},
            max_workers=self.max_workers,
            bootstrap_frames=self.bootstrap_frames,
            scale_up_after=self.scale_up_after,
            scale_interval=self.scale_interval,
            on_result=self._on_detections_ready,
        )

        # Start tracking thread
        self._running = True
        self._tracking_thread = threading.Thread(
            target=self._tracking_loop,
            daemon=True,
            name="parallel_tracking"
        )
        self._tracking_thread.start()

        logger.info("Parallel detection pipeline initialized and tracking thread started")

    def submit_frame(
        self,
        camera_id: str,
        frame: np.ndarray,
        timestamp: float,
    ):
        """Submit a frame for parallel processing.

        This replaces the synchronous `_detect_frame` call in the original loop.
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        self._stats["frames_received"] += 1
        self.pipeline.submit_frame(camera_id, frame, timestamp)

    def _on_detections_ready(self, batch: DetectionBatch):
        """Callback when parallel workers complete detection.

        This is called in worker threads - queue for central tracking.
        """
        try:
            self._result_queue.put_nowait(batch)
        except:
            logger.warning("Detection queue full, dropping batch")

    def _tracking_loop(self):
        """Central tracking loop - runs in dedicated thread.

        Receives detection batches from parallel workers, runs them through
        the central BoT-SORT tracker, extracts ReID features, and produces
        final detection results.
        """
        logger.info("Parallel tracking loop started")

        while self._running:
            try:
                # Get detection batch
                try:
                    batch = self._result_queue.get(timeout=0.1)
                except Empty:
                    continue

                # Run through central tracker
                tracking_start = time.time()
                result = self._run_central_tracking(batch)
                tracking_elapsed = (time.time() - tracking_start) * 1000

                # Update stats
                self._stats["frames_tracked"] += 1
                alpha = 0.1
                self._stats["tracking_time_ms"] = (
                    alpha * tracking_elapsed +
                    (1 - alpha) * self._stats["tracking_time_ms"]
                )

                if result:
                    # Feed back to detection loop for downstream processing
                    self._emit_result(result)

            except Exception as e:
                logger.error(f"Tracking loop error: {e}")

    def _run_central_tracking(self, batch: DetectionBatch) -> Optional[Any]:
        """Run central tracking on a detection batch.

        This mirrors the logic in DetectionLoop._detect_frame but uses
        detections from parallel workers instead of running YOLO.
        """
        dl = self.detection_loop
        bot_sort = dl.bot_sort

        if bot_sort is None:
            logger.warning("BoT-SORT not available for central tracking")
            return None

        camera_id = batch.camera_id
        frame = batch.frame
        vehicle_detections = batch.vehicle_detections
        person_detections = batch.person_detections

        # Step 1: Pre-tracking ReID for lost track recovery
        if dl.use_reid_features:
            self._extract_pre_tracking_reid(
                frame, camera_id, vehicle_detections, person_detections
            )

        # Step 2: Run BoT-SORT tracker update
        all_vehicles, new_vehicle_tracks = bot_sort.update(
            vehicle_detections,
            "vehicle",
            dt=1 / dl.config.detection_fps,
            camera_id=camera_id,
        )
        all_persons, new_person_tracks = bot_sort.update(
            person_detections,
            "person",
            dt=1 / dl.config.detection_fps,
            camera_id=camera_id,
        )

        # Step 3: Filter tracks by camera
        MAX_STALE_FRAMES = 5
        all_vehicles = [
            t for t in all_vehicles
            if t.last_seen_camera == camera_id and t.time_since_update <= MAX_STALE_FRAMES
        ]
        all_persons = [
            t for t in all_persons
            if t.last_seen_camera == camera_id and t.time_since_update <= MAX_STALE_FRAMES
        ]
        new_vehicle_tracks = [t for t in new_vehicle_tracks if t.last_seen_camera == camera_id]
        new_person_tracks = [t for t in new_person_tracks if t.last_seen_camera == camera_id]

        # Step 4: Track deduplication for parallel processing artifacts
        for track in new_vehicle_tracks:
            canonical_id = self.track_merger.check_and_merge(
                track.track_id,
                track.class_name,
                track.bbox,
                track.feature,
                batch.frame_id,
            )
            if canonical_id != track.track_id:
                # This track should be merged - update its ID
                # Note: This modifies the track object which affects the tracker
                track.track_id = canonical_id

        for track in new_person_tracks:
            canonical_id = self.track_merger.check_and_merge(
                track.track_id,
                "person",
                track.bbox,
                track.feature,
                batch.frame_id,
            )
            if canonical_id != track.track_id:
                track.track_id = canonical_id

        # Step 5: Post-tracking ReID for new tracks
        if dl.use_reid_features:
            self._extract_post_tracking_reid(frame, new_vehicle_tracks + new_person_tracks)

        # Step 6: Convert to result format
        from .detection_loop import DetectionResult

        tracked_vehicles = [dl._track_to_dict(t) for t in all_vehicles]
        tracked_persons = [dl._track_to_dict(t) for t in all_persons]
        new_vehicles = [dl._track_to_dict(t) for t in new_vehicle_tracks]
        new_persons = [dl._track_to_dict(t) for t in new_person_tracks]

        # Check for armed persons
        armed_persons = []
        for p in tracked_persons:
            meta = p.get("metadata", {})
            if meta and (meta.get("armed") or meta.get("חמוש")):
                armed_persons.append(p)

        # Draw annotations
        annotated_frame = None
        if dl.config.draw_bboxes:
            annotated_frame = dl.drawer.draw_detections(frame.copy(), tracked_vehicles, "vehicle")
            annotated_frame = dl.drawer.draw_detections(annotated_frame, tracked_persons, "person")
            annotated_frame = dl.drawer.draw_status_overlay(
                annotated_frame,
                camera_id,
                len(tracked_vehicles),
                len(tracked_persons),
                len(armed_persons),
            )

        return DetectionResult(
            camera_id=camera_id,
            timestamp=batch.timestamp,
            frame=frame,
            tracked_vehicles=tracked_vehicles,
            tracked_persons=tracked_persons,
            new_vehicles=new_vehicles,
            new_persons=new_persons,
            armed_persons=armed_persons,
            annotated_frame=annotated_frame,
        )

    def _extract_pre_tracking_reid(
        self,
        frame: np.ndarray,
        camera_id: str,
        vehicle_detections: List[Detection],
        person_detections: List[Detection],
    ):
        """Extract ReID features for lost track recovery (pre-tracking)."""
        dl = self.detection_loop
        bot_sort = dl.bot_sort

        if not bot_sort:
            return

        # Check for lost tracks
        lost_persons = [
            t for t in bot_sort.get_active_tracks("person", camera_id)
            if t.feature is not None and t.time_since_update > 5
        ]
        lost_vehicles = [
            t for t in bot_sort.get_active_tracks("vehicle", camera_id)
            if t.feature is not None and t.time_since_update > 5
        ]

        if not (lost_persons or lost_vehicles):
            return

        reid_count = 0
        max_reid = 5

        # Extract for persons
        if lost_persons:
            for det in person_detections[:max_reid]:
                if det.feature is None and reid_count < max_reid:
                    x, y, w, h = det.bbox
                    feature = dl._extract_reid_feature(frame, [x, y, x + w, y + h], "person")
                    if feature is not None:
                        det.feature = feature
                        reid_count += 1

        # Extract for vehicles
        if lost_vehicles:
            for det in vehicle_detections[:max_reid]:
                if det.feature is None and reid_count < max_reid:
                    x, y, w, h = det.bbox
                    feature = dl._extract_reid_feature(frame, [x, y, x + w, y + h], det.class_name)
                    if feature is not None:
                        det.feature = feature
                        reid_count += 1

    def _extract_post_tracking_reid(
        self,
        frame: np.ndarray,
        new_tracks: List,
    ):
        """Extract ReID features for new tracks (post-tracking)."""
        dl = self.detection_loop

        reid_count = 0
        max_reid = 10

        for track in new_tracks:
            if reid_count >= max_reid:
                break
            if track.feature is None:
                x, y, w, h = track.bbox
                feature = dl._extract_reid_feature(frame, [x, y, x + w, y + h], track.class_name)
                if feature is not None:
                    track.feature = feature
                    reid_count += 1

    def _emit_result(self, result):
        """Emit a tracking result back to the detection loop.

        This feeds into the async result handler for Gemini analysis,
        rule processing, events, etc.
        """
        dl = self.detection_loop

        # Store annotated frame for streaming
        if result.annotated_frame is not None:
            with dl._frame_lock:
                dl._annotated_frames[result.camera_id] = result.annotated_frame

        # Queue for async processing (Gemini, rules, events)
        if result.new_vehicles or result.new_persons or result.armed_persons:
            try:
                dl._result_queue.put_nowait(result)
            except:
                pass

    def shutdown(self):
        """Shutdown the parallel integration."""
        self._running = False

        if self._tracking_thread:
            self._tracking_thread.join(timeout=2.0)

        if self.pipeline:
            self.pipeline.shutdown()

        logger.info("ParallelDetectionIntegration shutdown")

    def get_stats(self) -> Dict[str, Any]:
        """Get integration statistics."""
        stats = {
            **self._stats,
            "track_merger": self.track_merger.get_stats(),
        }
        if self.pipeline:
            stats["pipeline"] = self.pipeline.get_stats()
        return stats
