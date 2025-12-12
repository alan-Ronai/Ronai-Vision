"""Camera manager: manage multiple camera readers (simulators or RTSP).

Provides a simple API to register cameras, start/stop them, and poll
frames with camera_id and timestamp. Supports both file simulators and
live RTSP streams.
"""

import threading
import time
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from services.camera.simulator import LocalCameraSimulator
from services.camera.rtsp_reader import RTSPReader


class CameraWorker:
    def __init__(self, camera_id: str, reader):
        self.camera_id = camera_id
        self.reader = reader
        self.lock = threading.Lock()
        self._frame = None
        self._running = False
        self._thread = None
        self.state = "stopped"  # States: stopped, starting, running, stopping

    def start(self):
        if self._running or self.state in ("starting", "running"):
            return
        self.state = "starting"
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"cam-{self.camera_id}"
        )
        self._thread.start()

    def _run(self):
        self.state = "running"
        print(f"[CameraWorker] {self.camera_id} started")

        consecutive_none_count = 0
        max_none_retries = 200  # ~1s of retries before longer sleep

        while self._running:
            try:
                frame = self.reader.get_frame()
                ts = time.time()
                with self.lock:
                    self._frame = (frame, ts)

                if frame is not None:
                    consecutive_none_count = 0
                else:
                    consecutive_none_count += 1
                    if consecutive_none_count == max_none_retries:
                        print(
                            f"[CameraWorker] {self.camera_id} no frames for {max_none_retries * 0.005:.1f}s, waiting for stream recovery..."
                        )
                        consecutive_none_count = 0
                        time.sleep(1.0)  # Wait longer for stream to recover
                        continue

            except Exception as e:
                # Reader might error (e.g., RTSP disconnect); log and retry
                print(f"[CameraWorker] {self.camera_id} read error: {e}")
                time.sleep(0.5)  # Wait before retry
                continue
            # Check stop flag frequently (5ms instead of 10ms) for faster shutdown
            time.sleep(0.005)

        self.state = "stopped"
        print(f"[CameraWorker] {self.camera_id} stopped")

    def stop(self):
        if not self._running:
            return
        self.state = "stopping"
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        self.state = "stopped"

    def get_latest(self) -> Optional[Tuple[np.ndarray, float]]:
        with self.lock:
            return self._frame

    def is_running(self) -> bool:
        return self._running and self.state == "running"


class CameraManager:
    def __init__(self):
        self.workers: Dict[str, CameraWorker] = {}
        self._lock = threading.Lock()

    def add_simulator(self, camera_id: str, file_path: str):
        """Add a file-based simulator camera."""
        reader = LocalCameraSimulator(file_path=file_path)
        worker = CameraWorker(camera_id, reader)
        with self._lock:
            self.workers[camera_id] = worker
        return worker

    def add_rtsp(self, camera_id: str, rtsp_url: str):
        """Add an RTSP camera source.

        Args:
            camera_id: unique identifier for the camera (e.g., "cam_entrance")
            rtsp_url: RTSP URL (e.g., "rtsp://user:pass@192.168.1.100:554/stream")
        """
        reader = RTSPReader(source=rtsp_url, reopen_on_eof=True)
        worker = CameraWorker(camera_id, reader)
        with self._lock:
            self.workers[camera_id] = worker
        return worker

    def add_from_config(self, config: dict):
        """Load cameras from a configuration dict.

        Args:
            config: dict with structure:
                {
                    "cameras": {
                        "cam1": {"type": "simulator", "source": "assets/video.mp4"},
                        "cam2": {"type": "rtsp", "source": "rtsp://user:pass@ip:554/stream"}
                    }
                }
        """
        cameras = config.get("cameras", {})
        for cam_id, cam_config in cameras.items():
            cam_type = cam_config.get("type", "simulator")
            source = cam_config.get("source")

            if not source:
                print(f"[WARNING] camera {cam_id}: no source specified, skipping")
                continue

            if cam_type == "rtsp":
                self.add_rtsp(cam_id, source)
                print(f"[INFO] Added RTSP camera: {cam_id} -> {source}")
            elif cam_type == "simulator":
                self.add_simulator(cam_id, source)
                print(f"[INFO] Added simulator camera: {cam_id} -> {source}")
            else:
                print(f"[WARNING] Unknown camera type '{cam_type}' for {cam_id}")

    def start_all(self):
        """Start all registered cameras."""
        with self._lock:
            workers = list(self.workers.values())
        for w in workers:
            w.start()

    def stop_all(self):
        """Stop all running cameras."""
        with self._lock:
            workers = list(self.workers.values())
        for w in workers:
            w.stop()

    def start_camera(self, camera_id: str) -> bool:
        """Start a specific camera by ID.

        Args:
            camera_id: Camera identifier

        Returns:
            True if started successfully, False if camera not found
        """
        with self._lock:
            worker = self.workers.get(camera_id)

        if worker is None:
            print(f"[WARNING] Cannot start camera '{camera_id}': not found")
            return False

        worker.start()
        return True

    def stop_camera(self, camera_id: str) -> bool:
        """Stop a specific camera by ID.

        Args:
            camera_id: Camera identifier

        Returns:
            True if stopped successfully, False if camera not found
        """
        with self._lock:
            worker = self.workers.get(camera_id)

        if worker is None:
            print(f"[WARNING] Cannot stop camera '{camera_id}': not found")
            return False

        worker.stop()
        return True

    def is_camera_running(self, camera_id: str) -> bool:
        """Check if a camera is currently running.

        Args:
            camera_id: Camera identifier

        Returns:
            True if running, False otherwise
        """
        with self._lock:
            worker = self.workers.get(camera_id)

        return worker.is_running() if worker else False

    def get_camera_ids(self) -> list:
        """Get list of all registered camera IDs.

        Returns:
            List of camera IDs
        """
        with self._lock:
            return list(self.workers.keys())

    def get_frames(self) -> Dict[str, Optional[Tuple[np.ndarray, float]]]:
        """Get latest frames from all cameras.

        Returns:
            Dict mapping camera_id to (frame, timestamp) or None
        """
        with self._lock:
            workers = dict(self.workers)
        return {cid: w.get_latest() for cid, w in workers.items()}
