"""Multi-camera runner: parallel processing of multiple camera feeds.

Orchestrates multiple cameras through a unified pipeline with:
- Parallel per-camera workers for maximum throughput
- Modular frame processing (FrameProcessor)
- Unified camera reading (CameraFrameReader)
- Output publishing (OutputPublisher)
- Cross-camera ReID (CrossCameraReID)

All cameras process frames independently in parallel threads.
"""

import os
import time
import json
import threading
import signal
import queue
from typing import Optional

import numpy as np

# Configure FFmpeg for error-tolerant RTSP decoding
# Must be set before importing cv2
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|err_detect;ignore_err|ec;guess_mvs",
)

from config.pipeline_config import PipelineConfig
from services.camera.manager import CameraManager
from services.camera.frame_reader import CameraFrameReader
from services.detector import create_detector, get_detector_info
from services.segmenter.sam2_segmenter import SAM2Segmenter
from services.reid import get_multi_class_reid
from services.reid.cross_camera import CrossCameraReID
from services.tracker.bot_sort import BoTSortTracker
from services.output.publisher import OutputPublisher
from services.device import choose_device
from services.profiler import profiler
from services.pipeline import FrameProcessor
from services.pipeline.worker_manager import WorkerManager
from services.output.broadcaster import broadcaster

# Public metrics structure populated at runtime for diagnostics
RUN_METRICS = {}

# Global camera manager instance for API access
_camera_manager = None


def get_camera_manager():
    """Get the global camera manager instance.

    Returns:
        CameraManager instance or None if not initialized
    """
    global _camera_manager
    return _camera_manager


def run_loop(
    stop_event: Optional[threading.Event] = None, max_frames: Optional[int] = None
) -> None:
    """Run the multi-camera processing loop in parallel mode.

    Args:
        stop_event: Optional threading.Event instance to signal shutdown
        max_frames: Max frames to process per camera (None = unlimited)
    """
    global RUN_METRICS
    global _camera_manager

    # Print configuration
    PipelineConfig.print_summary()

    # ========================================================================
    # LOAD CAMERAS
    # ========================================================================
    cm = CameraManager()
    _camera_manager = cm  # Store globally for API access
    if os.path.exists(PipelineConfig.CAMERA_CONFIG):
        with open(PipelineConfig.CAMERA_CONFIG, "r") as f:
            camera_config = json.load(f)
        cm.add_from_config(camera_config)
        print(
            f"[INFO] Loaded {len(cm.workers)} cameras from {PipelineConfig.CAMERA_CONFIG}"
        )

        # Register cameras with broadcaster for viewer tracking
        for cam_id in cm.get_camera_ids():
            broadcaster.register_camera(cam_id)
    else:
        print(f"[WARNING] Camera config not found at {PipelineConfig.CAMERA_CONFIG}")

    # Don't start cameras yet - they'll be started on-demand by WorkerManager
    # cm.start_all()  # <- REMOVED

    # ========================================================================
    # INITIALIZE MODELS
    # ========================================================================
    device = (
        PipelineConfig.DEVICE if PipelineConfig.DEVICE != "auto" else choose_device()
    )
    print(f"[INFO] Using device: {device}")

    # Create detector (single or multi based on config)
    detector = create_detector(device=device)
    detector_info = get_detector_info(detector)
    print(f"[INFO] Detector type: {detector_info['type']}")
    if detector_info["weapon_detection_enabled"]:
        print(f"[INFO] Weapon detection: ENABLED")
        print(f"[INFO] Active detectors: {', '.join(detector_info['detectors'])}")
    else:
        print(f"[INFO] Weapon detection: DISABLED")

    segmenter = SAM2Segmenter(model_type=PipelineConfig.SAM2_MODEL, device=device)
    reid = get_multi_class_reid(device=device)

    # ========================================================================
    # INITIALIZE PIPELINE COMPONENTS
    # ========================================================================
    camera_reader = CameraFrameReader(cm)
    output_publisher = OutputPublisher(
        save_frames=PipelineConfig.SAVE_FRAMES,
        stream_url=PipelineConfig.STREAM_SERVER_URL,
        stream_token=PipelineConfig.STREAM_PUBLISH_TOKEN,
        stop_event=stop_event,
    )
    cross_reid = CrossCameraReID()

    # Per-camera trackers (independent tracking per camera)
    trackers = {
        cam_id: BoTSortTracker(
            min_confidence_history=PipelineConfig.MIN_TRACK_CONFIDENCE
        )
        for cam_id in cm.workers.keys()
    }

    # ========================================================================
    # MODEL WARMUP
    # ========================================================================
    try:
        dummy = np.zeros((360, 640, 3), dtype=np.uint8)
        _ = detector.predict(dummy, confidence=PipelineConfig.YOLO_CONFIDENCE)
        print("[INFO] Model warmup complete")
    except Exception as e:
        print(f"[WARNING] Model warmup failed: {e}")

    # ========================================================================
    # TIMING & METRICS
    # ========================================================================
    timing = {
        "detect": 0.0,
        "segment": 0.0,
        "reid": 0.0,
        "track": 0.0,
        "count": 0,
    }
    RUN_METRICS = timing

    # ========================================================================
    # DYNAMIC WORKER MANAGEMENT
    # ========================================================================
    print(f"[INFO] Starting on-demand processing (workers start when viewers connect)")

    agg_queue = queue.Queue()
    global_id_map = {}
    global_id_lock = threading.Lock()

    def aggregator_worker():
        """Consume track embeddings and update global ReID store."""
        agg_count = 0
        while True:
            try:
                item = agg_queue.get(timeout=0.5)
            except queue.Empty:
                if stop_event and stop_event.is_set():
                    break
                continue

            if item is None:
                break

            try:
                cam_id = item["cam_id"]
                local_to_global = cross_reid.upsert_tracks(
                    camera_id=cam_id,
                    tracks=item["tracks"],
                    features=item["features"],
                    boxes=item["boxes"],
                )
                with global_id_lock:
                    global_id_map.update(local_to_global)
                agg_count += 1
                if agg_count % 50 == 0:
                    print(f"[aggregator] processed {agg_count} batches")
            except Exception as e:
                print(f"[aggregator] exception: {e}")
            finally:
                agg_queue.task_done()

    # Start aggregator thread (daemon=False for proper shutdown)
    agg_thread = threading.Thread(
        target=aggregator_worker, daemon=False, name="aggregator"
    )
    agg_thread.start()

    # Factory function to create processors
    def make_processor(cam_id: str, tracker):
        """Create FrameProcessor for a camera."""
        return FrameProcessor(
            detector=detector,
            segmenter=segmenter,
            reid=reid,
            tracker=tracker,
            allowed_classes=PipelineConfig.ALLOWED_CLASSES,
            yolo_confidence=PipelineConfig.YOLO_CONFIDENCE,
            reid_recovery=PipelineConfig.REID_RECOVERY,
            recovery_confidence=PipelineConfig.RECOVERY_CONFIDENCE,
            recovery_iou_thresh=PipelineConfig.RECOVERY_IOU_THRESH,
            recovery_reid_thresh=PipelineConfig.RECOVERY_REID_THRESH,
            recovery_min_track_confidence=PipelineConfig.RECOVERY_MIN_TRACK_CONFIDENCE,
        )

    # Initialize WorkerManager
    worker_manager = WorkerManager(
        camera_manager=cm,
        camera_reader=camera_reader,
        processor_factory=make_processor,
        output_publisher=output_publisher,
        cross_reid=cross_reid,
        agg_queue=agg_queue,
        trackers=trackers,
        stop_event=stop_event,
        grace_period=30.0,  # 30s grace period before stopping inactive cameras
    )

    worker_manager.start()

    print("[INFO] WorkerManager started - cameras will activate when viewers connect")
    print(f"[INFO] Available cameras: {', '.join(cm.get_camera_ids())}")
    print("[INFO] View streams at: http://127.0.0.1:8000/api/stream/mjpeg/<camera_id>")
    print("[INFO] List cameras at: http://127.0.0.1:8000/api/stream/cameras")

    # Wait for shutdown signal
    try:
        while True:
            if stop_event and stop_event.is_set():
                print("[INFO] Stop event detected, shutting down...")
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[INFO] Keyboard interrupt received")
    finally:
        # Shutdown worker manager (stops all active workers)
        print("[INFO] Stopping worker manager...")
        worker_manager.stop()

        # Stop camera manager
        print("[INFO] Stopping cameras...")
        cm.stop_all()

        # Shutdown aggregator
        print("[INFO] Stopping aggregator...")
        agg_queue.put(None)
        agg_thread.join(timeout=2.0)

        print("\n" + "=" * 60)
        print("FINAL PERFORMANCE REPORT")
        print("=" * 60)
        profiler.print_report()


def main():
    """Main entry point with signal handling for graceful shutdown."""
    stop_event = threading.Event()

    def signal_handler(sig, frame):
        print("\n[INFO] Received interrupt signal, setting stop event...")
        stop_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        run_loop(stop_event=stop_event, max_frames=PipelineConfig.MAX_FRAMES)
    except KeyboardInterrupt:
        print("\n[INFO] Keyboard interrupt, shutting down...")
        stop_event.set()
    finally:
        print("[INFO] Cleanup complete")


if __name__ == "__main__":
    main()
