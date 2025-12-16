"""Dynamic worker manager: starts/stops camera workers based on viewer demand.

Monitors viewer tracker for viewer counts and automatically:
- Starts camera workers when viewers connect
- Stops workers after 30s grace period when all viewers disconnect
"""

import threading
import time
from typing import Dict, Optional, Set
import queue

from services.output.broadcaster import broadcaster
from services import viewer_tracker


class WorkerManager:
    """Manages dynamic starting/stopping of camera processing workers."""

    def __init__(
        self,
        camera_manager,
        camera_reader,
        processor_factory,
        output_publisher,
        cross_reid,
        agg_queue: queue.Queue,
        trackers: Dict[str, any],
        stop_event: Optional[threading.Event] = None,
        grace_period: float = 30.0,
    ):
        """Initialize worker manager.

        Args:
            camera_manager: CameraManager instance
            camera_reader: CameraFrameReader instance
            processor_factory: Callable that creates FrameProcessor for a camera
            output_publisher: OutputPublisher instance
            cross_reid: CrossCameraReID instance
            agg_queue: Queue for aggregating ReID results
            trackers: Dict mapping camera_id to tracker instance
            stop_event: Optional event to signal shutdown
            grace_period: Seconds to wait before stopping inactive camera (default: 30s)
        """
        self.camera_manager = camera_manager
        self.camera_reader = camera_reader
        self.processor_factory = processor_factory
        self.output_publisher = output_publisher
        self.cross_reid = cross_reid
        self.agg_queue = agg_queue
        self.trackers = trackers
        self.stop_event = stop_event
        self.grace_period = grace_period

        # Track active workers
        self._active_workers: Dict[str, threading.Thread] = {}
        self._worker_stop_events: Dict[str, threading.Event] = {}
        self._last_viewer_time: Dict[str, float] = {}
        self._lock = threading.Lock()

        # Control thread
        self._control_thread = None
        self._running = False

    def start(self):
        """Start the worker manager control loop."""
        if self._running:
            return

        self._running = True
        self._control_thread = threading.Thread(
            target=self._control_loop, daemon=True, name="worker-manager"
        )
        self._control_thread.start()
        print("[WorkerManager] Started")

    def stop(self):
        """Stop the worker manager and all active workers."""
        if not self._running:
            return

        self._running = False

        # Stop all active workers
        with self._lock:
            camera_ids = list(self._active_workers.keys())

        for cam_id in camera_ids:
            self._stop_worker(cam_id)

        # Wait for control thread
        if self._control_thread:
            self._control_thread.join(timeout=5.0)

        print("[WorkerManager] Stopped")

    def _control_loop(self):
        """Main control loop: monitors viewers and starts/stops workers."""
        while self._running:
            if self.stop_event and self.stop_event.is_set():
                break

            try:
                # Get cameras with active viewers (cross-process safe)
                active_cameras = set(viewer_tracker.get_active_cameras())

                # Get all configured cameras
                all_cameras = set(self.camera_manager.get_camera_ids())

                with self._lock:
                    running_workers = set(self._active_workers.keys())

                # Start workers for cameras with viewers
                for cam_id in active_cameras:
                    if cam_id not in running_workers:
                        print(
                            f"[WorkerManager] {cam_id} has viewers, starting worker..."
                        )
                        self._start_worker(cam_id)
                        with self._lock:
                            self._last_viewer_time[cam_id] = time.time()

                # Update last viewer time for active cameras
                for cam_id in active_cameras:
                    with self._lock:
                        self._last_viewer_time[cam_id] = time.time()

                # Check for workers to stop (grace period expired)
                current_time = time.time()
                for cam_id in running_workers:
                    if cam_id not in active_cameras:
                        with self._lock:
                            last_time = self._last_viewer_time.get(cam_id, 0)

                        # Check if grace period expired
                        if current_time - last_time >= self.grace_period:
                            print(
                                f"[WorkerManager] Grace period expired for {cam_id}, stopping worker"
                            )
                            self._stop_worker(cam_id)

            except Exception as e:
                print(f"[WorkerManager] Control loop error: {e}")

            # Poll every 1 second
            time.sleep(1.0)

    def _start_worker(self, cam_id: str):
        """Start a worker for a specific camera.

        Args:
            cam_id: Camera identifier
        """
        with self._lock:
            # Check if already running
            if cam_id in self._active_workers:
                return

            # Create stop event for this worker
            worker_stop = threading.Event()
            self._worker_stop_events[cam_id] = worker_stop

        # Start camera
        print(f"[WorkerManager] Starting camera: {cam_id}")
        self.camera_manager.start_camera(cam_id)

        # Wait briefly for camera to initialize
        time.sleep(0.5)

        # Create and start worker thread
        worker_func = self._make_worker_func(cam_id, worker_stop)
        worker_thread = threading.Thread(
            target=worker_func, daemon=False, name=f"worker-{cam_id}"
        )

        with self._lock:
            self._active_workers[cam_id] = worker_thread

        worker_thread.start()
        print(f"[WorkerManager] Worker started: {cam_id}")

    def _stop_worker(self, cam_id: str):
        """Stop a worker for a specific camera.

        Args:
            cam_id: Camera identifier
        """
        with self._lock:
            worker_thread = self._active_workers.get(cam_id)
            worker_stop = self._worker_stop_events.get(cam_id)

        if worker_stop:
            # Signal worker to stop
            worker_stop.set()

        if worker_thread:
            # Wait for worker to exit
            print(f"[WorkerManager] Stopping worker: {cam_id}")
            worker_thread.join(timeout=5.0)

            if worker_thread.is_alive():
                print(f"[WARNING] Worker {cam_id} did not stop within timeout")

        # Stop camera
        self.camera_manager.stop_camera(cam_id)

        # Clean up
        with self._lock:
            self._active_workers.pop(cam_id, None)
            self._worker_stop_events.pop(cam_id, None)
            self._last_viewer_time.pop(cam_id, None)

        print(f"[WorkerManager] Worker stopped: {cam_id}")

    def _make_worker_func(self, cam_id: str, worker_stop: threading.Event):
        """Create worker function for a camera.

        Args:
            cam_id: Camera identifier
            worker_stop: Event to signal worker shutdown

        Returns:
            Worker function
        """

        def _worker():
            frame_idx = 0
            tracker = self.trackers[cam_id]
            processor = self.processor_factory(cam_id, tracker)

            print(f"[Worker-{cam_id}] Started processing")

            # Set camera context for per-camera profiling
            from services.profiler import profiler
            from config.pipeline_config import PipelineConfig
            from services.recorder import get_event_recorder
            from services.gemini import get_gemini_analyzer
            from services.tracker.metadata_manager import get_metadata_manager
            from services.logging import get_operational_logger

            profiler.set_camera_context(cam_id)

            # Get event recorder instance
            event_recorder = get_event_recorder()

            # Get Gemini analyzer (optional, may be None if not configured)
            gemini_analyzer = get_gemini_analyzer()
            if gemini_analyzer:
                print(f"[Worker-{cam_id}] Gemini analysis enabled")
            else:
                print(f"[Worker-{cam_id}] Gemini analysis disabled (no API key or library not installed)")

            # Get metadata manager
            metadata_manager = get_metadata_manager()

            # Get operational logger
            op_logger = get_operational_logger()

            detection_skip = PipelineConfig.DETECTION_SKIP

            # Track newly seen track IDs to log only once
            logged_tracks = set()

            if detection_skip > 1:
                print(
                    f"[Worker-{cam_id}] Detection skip enabled: running YOLO every {detection_skip} frames"
                )

            while not worker_stop.is_set():
                # Check global stop event
                if self.stop_event and self.stop_event.is_set():
                    break

                # Get frame
                frame_data = self.camera_reader.get_frame(cam_id)
                if frame_data is None:
                    if frame_idx == 0:
                        print(f"[Worker-{cam_id}] Waiting for frames from camera...")
                    time.sleep(0.005)
                    if worker_stop.is_set():
                        break
                    continue

                frame, ts = frame_data

                # Skip if frame is None
                if frame is None:
                    time.sleep(0.005)
                    continue

                # Process frame (detection skipping handled inside processor)
                try:
                    result, frame_timing = processor.process_frame(frame)

                    # Extract tracks and features
                    tracks = result["tracks"]
                    features = result.get("features")

                    # Add frame to event recorder buffer
                    event_recorder.add_frame(cam_id, frame, ts)

                    # Log new detections
                    for track in tracks:
                        class_name = result["class_names"][track.class_id] if track.class_id < len(result["class_names"]) else "unknown"

                        # Log new tracks (first time seen)
                        if track.track_id not in logged_tracks and class_name in ["person", "car"]:
                            logged_tracks.add(track.track_id)
                            op_logger.log_detection(
                                camera_id=cam_id,
                                track_id=track.track_id,
                                class_name=class_name,
                            )

                    # Perform Gemini analysis for cars and persons (limited per ID)
                    if gemini_analyzer:
                        for track in tracks:
                            # Get class name
                            class_name = result["class_names"][track.class_id] if track.class_id < len(result["class_names"]) else ""

                            # Only analyze cars and persons
                            if class_name not in ["car", "person"]:
                                continue

                            # Check if analysis should be performed (max 2-3 times per ID)
                            if not metadata_manager.should_analyze_with_gemini(track.track_id, max_analyses=2):
                                continue

                            # Perform analysis asynchronously (non-blocking)
                            # Note: For production, this should be done in a separate thread pool
                            # to avoid blocking the main processing loop
                            try:
                                print(f"[Worker-{cam_id}] Analyzing {class_name} (track_id={track.track_id}) with Gemini")

                                if class_name == "car":
                                    analysis = gemini_analyzer.analyze_car(frame, track.box)
                                    track.set_attribute("gemini_car_analysis", analysis)
                                    # Log analysis result
                                    if not analysis.get("error"):
                                        op_logger.log_analysis(
                                            camera_id=cam_id,
                                            track_id=track.track_id,
                                            class_name=class_name,
                                            analysis_result=analysis,
                                        )
                                elif class_name == "person":
                                    analysis = gemini_analyzer.analyze_person(frame, track.box)
                                    track.set_attribute("gemini_person_analysis", analysis)
                                    # Log analysis result
                                    if not analysis.get("error"):
                                        op_logger.log_analysis(
                                            camera_id=cam_id,
                                            track_id=track.track_id,
                                            class_name=class_name,
                                            analysis_result=analysis,
                                        )

                                # Record that analysis was performed
                                metadata_manager.record_gemini_analysis(track.track_id)

                                # Update metadata in manager
                                metadata_manager.update_track_metadata(
                                    track.track_id, track.class_id, track.get_metadata_summary()
                                )

                                print(f"[Worker-{cam_id}] Gemini analysis completed for track {track.track_id}")

                            except Exception as e:
                                logger.error(f"Gemini analysis failed for track {track.track_id}: {e}")

                    # Check for armed person detection and auto-start recording
                    for track in tracks:
                        if "armed" in track.metadata.get("tags", []):
                            # Check if not already recording this track
                            if not event_recorder.is_track_recording(track.track_id):
                                print(
                                    f"[Worker-{cam_id}] Armed person detected (track_id={track.track_id}), starting recording"
                                )
                                event_recorder.start_recording(
                                    camera_id=cam_id,
                                    trigger_type="auto",
                                    trigger_reason="armed_person_detected",
                                    track_id=track.track_id,
                                    include_buffer=True,
                                )

                                # Log armed person alert
                                weapons = track.metadata.get("attributes", {}).get("weapons_detected", [])
                                weapon_desc = ", ".join(weapons) if weapons else "נשק"
                                op_logger.log_alert(
                                    camera_id=cam_id,
                                    track_id=track.track_id,
                                    alert_type="armed_person",
                                    alert_message=f"אדם חמוש זוהה עם {weapon_desc}",
                                    details={"weapons": weapons},
                                )

                                # Log recording started
                                op_logger.log_recording(
                                    camera_id=cam_id,
                                    action="started",
                                    trigger_reason="armed_person_detected",
                                    track_id=track.track_id,
                                )

                    # Check if any armed person left the frame (stop recording)
                    # Get all tracks currently recording for this camera
                    active_sessions = event_recorder.get_active_sessions(camera_id=cam_id)
                    for session in active_sessions:
                        if session["trigger_type"] == "auto" and session["track_id"] is not None:
                            # Check if this track is still present and armed
                            track_id = session["track_id"]
                            track_still_armed = any(
                                t.track_id == track_id and "armed" in t.metadata.get("tags", [])
                                for t in tracks
                            )
                            if not track_still_armed:
                                print(
                                    f"[Worker-{cam_id}] Armed person (track_id={track_id}) no longer detected, stopping recording"
                                )
                                output_path = event_recorder.stop_recording(session["session_id"])

                                # Log recording stopped
                                if output_path:
                                    op_logger.log_recording(
                                        camera_id=cam_id,
                                        action="stopped",
                                        track_id=track_id,
                                        output_path=output_path,
                                    )

                    # Queue for cross-camera ReID
                    if len(tracks) > 0:
                        self.agg_queue.put(
                            {
                                "cam_id": cam_id,
                                "tracks": tracks,
                                "features": features,
                                "boxes": result["filtered_boxes"],
                            }
                        )

                    # Publish output
                    self.output_publisher.publish(
                        frame=frame,
                        camera_id=cam_id,
                        frame_idx=frame_idx,
                        process_result=result,
                    )

                    frame_idx += 1

                    # Log first few frames
                    if frame_idx <= 3 or frame_idx % 100 == 0:
                        print(f"[Worker-{cam_id}] Processed frame {frame_idx}")

                    # Export profiler stats periodically (every 50 frames)
                    if frame_idx % 50 == 0:
                        profiler.export_to_file()

                except KeyboardInterrupt:
                    # Allow shutdown via Ctrl+C
                    break
                except Exception as e:
                    print(f"[Worker-{cam_id}] Processing error: {e}")
                    import traceback

                    traceback.print_exc()
                    # Add small delay after error to prevent tight loop
                    time.sleep(0.05)

                time.sleep(0.01)

            print(f"[Worker-{cam_id}] Stopped after {frame_idx} frames")

        return _worker
