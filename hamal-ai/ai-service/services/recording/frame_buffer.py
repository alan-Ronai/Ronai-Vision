"""Frame Buffer - Circular buffer for storing recent frames per camera.

This allows video recordings to include frames from BEFORE the recording was triggered
(pre-buffer), enabling capture of events that led up to the trigger.
"""

import threading
import time
import logging
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BufferedFrame:
    """A frame stored in the buffer with metadata."""
    frame: np.ndarray
    timestamp: float
    camera_id: str


class FrameBuffer:
    """Circular buffer that stores recent frames for each camera.

    Maintains a rolling buffer of the last N seconds of frames per camera.
    When a recording is triggered, we can retrieve frames from before the trigger.
    """

    def __init__(self, buffer_duration: float = 10.0, fps_estimate: float = 15.0):
        """Initialize frame buffer.

        Args:
            buffer_duration: How many seconds of frames to keep (default 10s)
            fps_estimate: Expected FPS for calculating buffer size
        """
        self.buffer_duration = buffer_duration
        self.fps_estimate = fps_estimate

        # Calculate max frames to store per camera
        self.max_frames_per_camera = int(buffer_duration * fps_estimate)

        # Per-camera frame buffers: {camera_id: deque of BufferedFrame}
        self._buffers: Dict[str, deque] = {}
        self._lock = threading.Lock()

        logger.info(f"FrameBuffer initialized: {buffer_duration}s buffer, ~{self.max_frames_per_camera} frames/camera")

    def add_frame(self, camera_id: str, frame: np.ndarray, timestamp: Optional[float] = None):
        """Add a frame to the buffer for a camera.

        Args:
            camera_id: Camera identifier
            frame: Video frame (numpy array, BGR format)
            timestamp: Frame timestamp (defaults to current time)
        """
        if timestamp is None:
            timestamp = time.time()

        buffered = BufferedFrame(
            frame=frame.copy(),  # Copy to avoid reference issues
            timestamp=timestamp,
            camera_id=camera_id
        )

        with self._lock:
            if camera_id not in self._buffers:
                self._buffers[camera_id] = deque(maxlen=self.max_frames_per_camera)

            self._buffers[camera_id].append(buffered)

    def get_frames(
        self,
        camera_id: str,
        seconds: float,
        from_timestamp: Optional[float] = None
    ) -> List[BufferedFrame]:
        """Get frames from the buffer for a camera.

        Args:
            camera_id: Camera identifier
            seconds: How many seconds of frames to retrieve
            from_timestamp: Get frames before this timestamp (defaults to now)

        Returns:
            List of BufferedFrame objects, oldest first
        """
        if from_timestamp is None:
            from_timestamp = time.time()

        cutoff_time = from_timestamp - seconds

        with self._lock:
            if camera_id not in self._buffers:
                return []

            # Filter frames within the time window
            frames = [
                f for f in self._buffers[camera_id]
                if f.timestamp >= cutoff_time and f.timestamp <= from_timestamp
            ]

            return frames

    def get_all_frames(self, camera_id: str) -> List[BufferedFrame]:
        """Get all buffered frames for a camera.

        Args:
            camera_id: Camera identifier

        Returns:
            List of all BufferedFrame objects for the camera
        """
        with self._lock:
            if camera_id not in self._buffers:
                return []
            return list(self._buffers[camera_id])

    def clear_camera(self, camera_id: str):
        """Clear buffer for a specific camera.

        Args:
            camera_id: Camera identifier
        """
        with self._lock:
            if camera_id in self._buffers:
                self._buffers[camera_id].clear()

    def clear_all(self):
        """Clear all buffers."""
        with self._lock:
            self._buffers.clear()

    def get_stats(self) -> Dict:
        """Get buffer statistics.

        Returns:
            Dictionary with buffer stats per camera
        """
        with self._lock:
            stats = {
                "buffer_duration": self.buffer_duration,
                "max_frames_per_camera": self.max_frames_per_camera,
                "cameras": {}
            }

            for camera_id, buffer in self._buffers.items():
                if buffer:
                    oldest = buffer[0].timestamp
                    newest = buffer[-1].timestamp
                    actual_duration = newest - oldest
                else:
                    actual_duration = 0

                stats["cameras"][camera_id] = {
                    "frame_count": len(buffer),
                    "actual_duration_seconds": round(actual_duration, 2)
                }

            return stats


# Global singleton
_frame_buffer: Optional[FrameBuffer] = None


def get_frame_buffer() -> FrameBuffer:
    """Get or create the global frame buffer instance."""
    global _frame_buffer
    if _frame_buffer is None:
        _frame_buffer = FrameBuffer()
    return _frame_buffer
