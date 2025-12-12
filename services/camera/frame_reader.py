"""Camera frame reader: unified frame fetching from cameras.

Handles the complexity of reading frames from different camera types
in a uniform interface.
"""

import time
import numpy as np
from typing import Optional, Tuple

from services.camera.manager import CameraManager


class CameraFrameReader:
    """Unified camera frame reader.

    Handles fetching frames from CameraManager in a consistent way,
    with timestamp tracking and frame validation.
    """

    def __init__(self, camera_manager: CameraManager):
        """Initialize reader.

        Args:
            camera_manager: CameraManager instance with started workers
        """
        self.camera_manager = camera_manager
        self.last_timestamps = {}  # Track per-camera to avoid duplicate frames

    def get_frame(self, camera_id: str) -> Optional[Tuple[np.ndarray, float]]:
        """Get latest frame from a camera.

        Args:
            camera_id: Camera identifier from config

        Returns:
            (frame, timestamp) tuple or None if no new frame available
        """
        worker = self.camera_manager.workers.get(camera_id)
        if worker is None:
            return None

        latest = worker.get_latest()
        if latest is None:
            return None

        frame, ts = latest

        # Skip duplicate frames (same timestamp as last read)
        last_ts = self.last_timestamps.get(camera_id, -1)
        if ts is not None and ts <= last_ts:
            return None

        self.last_timestamps[camera_id] = ts

        return frame, ts
