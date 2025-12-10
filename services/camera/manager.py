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

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while self._running:
            frame = self.reader.get_frame()
            ts = time.time()
            with self.lock:
                self._frame = (frame, ts)
            time.sleep(0.01)

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def get_latest(self) -> Optional[Tuple[np.ndarray, float]]:
        with self.lock:
            return self._frame


class CameraManager:
    def __init__(self):
        self.workers: Dict[str, CameraWorker] = {}

    def add_simulator(self, camera_id: str, file_path: str):
        """Add a file-based simulator camera."""
        reader = LocalCameraSimulator(file_path=file_path)
        worker = CameraWorker(camera_id, reader)
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
        for w in self.workers.values():
            w.start()

    def stop_all(self):
        for w in self.workers.values():
            w.stop()

    def get_frames(self) -> Dict[str, Optional[Tuple[np.ndarray, float]]]:
        return {cid: w.get_latest() for cid, w in self.workers.items()}
