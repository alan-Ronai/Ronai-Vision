"""Frame buffer for smooth video streaming.

Provides a thread-safe buffer that captures frames from RTSP
at a consistent rate for smooth streaming to clients.
"""

import cv2
import time
import logging
import os
from threading import Thread, Lock
from typing import Optional, Dict
import numpy as np

logger = logging.getLogger(__name__)


class FrameBuffer:
    """Thread-safe frame buffer for smooth streaming."""

    def __init__(
        self,
        camera_id: str,
        rtsp_url: str,
        target_fps: int = 15,
        username: Optional[str] = None,
        password: Optional[str] = None
    ):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.target_fps = target_fps
        self.username = username
        self.password = password

        self._frame: Optional[np.ndarray] = None
        self._frame_lock = Lock()
        self._running = False
        self._thread: Optional[Thread] = None
        self._last_frame_time = 0.0
        self._connected = False
        self._error: Optional[str] = None
        self._consecutive_read_errors = 0

        # Build full URL with credentials
        self._full_url = self._build_url()

        # Stats
        self._stats = {
            "frames_captured": 0,
            "reconnects": 0,
            "errors": 0
        }

    def _build_url(self) -> str:
        """Build RTSP URL with credentials if provided."""
        url = self.rtsp_url
        if self.username and self.password and "@" not in url:
            if "://" in url:
                protocol, rest = url.split("://", 1)
                url = f"{protocol}://{self.username}:{self.password}@{rest}"
        return url

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def error(self) -> Optional[str]:
        return self._error

    def start(self):
        """Start frame capture thread."""
        if self._running:
            return

        self._running = True
        self._thread = Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info(f"FrameBuffer started for {self.camera_id}")

    def stop(self):
        """Stop frame capture."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        logger.info(f"FrameBuffer stopped for {self.camera_id}")

    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame (thread-safe copy)."""
        with self._frame_lock:
            if self._frame is not None:
                return self._frame.copy()
        return None

    def get_stats(self) -> Dict:
        """Get buffer statistics."""
        return {
            "camera_id": self.camera_id,
            "connected": self._connected,
            "running": self._running,
            "has_frame": self._frame is not None,
            "last_frame_time": self._last_frame_time,
            **self._stats
        }

    def _is_frame_corrupted(self, frame: np.ndarray, threshold: float = None) -> bool:
        """Check if frame is obviously corrupted (mostly single color)."""
        if threshold is None:
            threshold = float(os.getenv("RTSP_CORRUPTION_THRESHOLD", "0.5"))

        try:
            # Check if frame is mostly green (common H264 corruption)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Green detection (H: 35-85, S: 50-255, V: 50-255)
            green_mask = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255))
            green_ratio = np.count_nonzero(green_mask) / green_mask.size

            # Black detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            black_ratio = np.count_nonzero(gray < 10) / gray.size

            # If >50% green or >80% black, frame is likely corrupted
            if green_ratio > threshold or black_ratio > 0.8:
                return True

            return False
        except:
            return False

    def _capture_loop(self):
        """Main capture loop with H264 error recovery."""
        cap = None
        reconnect_delay = 2.0
        frame_interval = 1.0 / self.target_fps

        # Get configurable RTSP settings from environment
        probe_size = os.getenv("RTSP_PROBE_SIZE", "5000000")
        analyze_duration = os.getenv("RTSP_ANALYZE_DURATION", "2000000")
        buffer_size = int(os.getenv("RTSP_BUFFER_SIZE", "3"))
        error_concealment = os.getenv("RTSP_ERROR_CONCEALMENT", "true").lower() == "true"
        skip_corrupted = os.getenv("RTSP_SKIP_CORRUPTED_FRAMES", "true").lower() == "true"

        # Set FFmpeg options for H264 error tolerance
        if error_concealment:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                f"rtsp_transport;tcp|"
                f"fflags;+genpts+discardcorrupt|"
                f"err_detect;ignore_err|"
                f"analyzeduration;{analyze_duration}|"
                f"probesize;{probe_size}"
            )
        else:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

        while self._running:
            try:
                # Connect if needed
                if cap is None or not cap.isOpened():
                    self._connected = False
                    logger.info(f"Connecting to RTSP: {self.camera_id}")

                    cap = cv2.VideoCapture(self._full_url, cv2.CAP_FFMPEG)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
                    cap.set(cv2.CAP_PROP_FPS, self.target_fps)

                    if not cap.isOpened():
                        logger.error(f"Failed to open RTSP: {self.camera_id}")
                        self._error = "Failed to connect to RTSP stream"
                        self._stats["errors"] += 1
                        time.sleep(reconnect_delay)
                        continue

                    # Wait for keyframe - read and discard a few frames to stabilize
                    logger.debug(f"Waiting for keyframe: {self.camera_id}")
                    for _ in range(5):
                        cap.read()

                    self._connected = True
                    self._error = None
                    self._stats["reconnects"] += 1
                    logger.info(f"Connected to RTSP: {self.camera_id}")

                # Read frame
                ret, frame = cap.read()

                if not ret or frame is None:
                    self._consecutive_read_errors += 1
                    # Only log on first failure, every 10th failure
                    if self._consecutive_read_errors == 1 or self._consecutive_read_errors % 10 == 0:
                        logger.warning(f"Failed to read frame: {self.camera_id} (errors: {self._consecutive_read_errors})")

                    # Reconnect after 30 consecutive errors
                    if self._consecutive_read_errors > 30:
                        logger.warning(f"Reconnecting {self.camera_id} after {self._consecutive_read_errors} errors")
                        self._connected = False
                        cap.release()
                        cap = None
                    time.sleep(0.5)
                    continue

                # Check for obviously corrupted frame (if enabled)
                if skip_corrupted and self._is_frame_corrupted(frame):
                    self._consecutive_read_errors += 1
                    if self._consecutive_read_errors % 10 == 0:
                        logger.debug(f"Skipping corrupted frame: {self.camera_id}")
                    continue

                # Success - reset error counter
                self._consecutive_read_errors = 0

                # Update frame buffer
                with self._frame_lock:
                    self._frame = frame
                    self._last_frame_time = time.time()

                self._stats["frames_captured"] += 1

                # Rate limiting
                time.sleep(frame_interval)

            except Exception as e:
                logger.error(f"Capture error for {self.camera_id}: {e}")
                self._error = str(e)
                self._stats["errors"] += 1
                self._connected = False
                if cap:
                    cap.release()
                    cap = None
                time.sleep(reconnect_delay)

        # Cleanup
        if cap:
            cap.release()


class FrameBufferManager:
    """Manages multiple frame buffers."""

    def __init__(self):
        self._buffers: Dict[str, FrameBuffer] = {}
        self._lock = Lock()

    def get_or_create(
        self,
        camera_id: str,
        rtsp_url: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        target_fps: int = 15
    ) -> FrameBuffer:
        """Get existing buffer or create new one."""
        with self._lock:
            if camera_id in self._buffers:
                return self._buffers[camera_id]

            buffer = FrameBuffer(
                camera_id=camera_id,
                rtsp_url=rtsp_url,
                target_fps=target_fps,
                username=username,
                password=password
            )
            buffer.start()
            self._buffers[camera_id] = buffer
            return buffer

    def get(self, camera_id: str) -> Optional[FrameBuffer]:
        """Get buffer by camera ID."""
        return self._buffers.get(camera_id)

    def remove(self, camera_id: str):
        """Stop and remove buffer."""
        with self._lock:
            if camera_id in self._buffers:
                self._buffers[camera_id].stop()
                del self._buffers[camera_id]

    def stop_all(self):
        """Stop all buffers."""
        with self._lock:
            for buffer in self._buffers.values():
                buffer.stop()
            self._buffers.clear()

    def get_all_stats(self) -> Dict:
        """Get stats for all buffers."""
        return {
            cam_id: buf.get_stats()
            for cam_id, buf in self._buffers.items()
        }


# Global singleton
_manager: Optional[FrameBufferManager] = None


def get_frame_buffer_manager() -> FrameBufferManager:
    """Get or create frame buffer manager."""
    global _manager
    if _manager is None:
        _manager = FrameBufferManager()
    return _manager
