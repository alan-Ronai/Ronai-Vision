"""FFmpeg-based RTSP reader for stable streaming.

OpenCV's RTSP is unreliable over network - constantly reconnects due to H.264 packet loss.
FFmpeg handles errors much better and uses TCP transport to avoid UDP packet loss.

This provides smooth, stable video streaming for the detection pipeline.
"""

import subprocess
import numpy as np
import logging
import time
import threading
from typing import Optional, Callable, Tuple, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FFmpegConfig:
    """FFmpeg stream configuration."""
    width: int = 1280
    height: int = 720
    fps: int = 15
    tcp: bool = True
    reconnect_delay: float = 2.0
    quality: int = 80


class FFmpegStream:
    """Stable RTSP stream using FFmpeg subprocess."""

    def __init__(
        self,
        camera_id: str,
        rtsp_url: str,
        config: Optional[FFmpegConfig] = None,
        on_frame: Optional[Callable[[np.ndarray], None]] = None
    ):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.config = config or FFmpegConfig()
        self.on_frame = on_frame

        self._process: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Frame buffer - latest frame always available
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._frame_time = 0.0
        self._frame_count = 0

        # Stats
        self._connected = False
        self._frames_read = 0
        self._reconnects = 0
        self._errors = 0

    @property
    def is_connected(self) -> bool:
        return self._connected

    def start(self):
        """Start FFmpeg stream in background."""
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._thread.start()
        logger.info(f"FFmpeg stream started: {self.camera_id}")

    def stop(self):
        """Stop FFmpeg stream."""
        self._stop_event.set()

        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=3)
            except:
                try:
                    self._process.kill()
                except:
                    pass

        if self._thread:
            self._thread.join(timeout=3)

        self._connected = False
        logger.info(f"FFmpeg stream stopped: {self.camera_id}")

    def get_frame(self) -> Tuple[Optional[np.ndarray], float]:
        """Get latest frame and its timestamp.

        Returns:
            (frame, timestamp) or (None, 0) if no frame
        """
        with self._frame_lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy(), self._frame_time
            return None, 0.0

    def _build_ffmpeg_cmd(self) -> list:
        """Build FFmpeg command for RTSP reading with ultra-low latency."""
        cmd = [
            'ffmpeg',
            '-hide_banner',
            '-loglevel', 'error',

            # CRITICAL: Low-latency input options (reduces 2-4 seconds of delay!)
            '-fflags', 'nobuffer+fastseek+flush_packets',  # Disable buffering
            '-flags', 'low_delay',           # Low delay mode
            '-avioflags', 'direct',          # Direct I/O (bypass buffering)
            '-probesize', '32',              # Minimal probe size (faster start)
            '-analyzeduration', '0',         # Don't analyze stream (immediate start)

            # RTSP transport and error handling
            '-rtsp_transport', 'tcp' if self.config.tcp else 'udp',  # TCP more reliable
            '-fflags', '+genpts+discardcorrupt',  # Handle timing issues
            '-max_delay', '100000',          # Reduced from 500000 for lower latency
            '-reorder_queue_size', '0',      # No reordering queue

            '-i', self.rtsp_url,

            # Output options - prioritize speed over quality
            '-vf', f'fps={self.config.fps},scale={self.config.width}:{self.config.height}',
            '-pix_fmt', 'bgr24',
            '-f', 'rawvideo',
            '-vsync', 'drop',                # Drop frames to maintain timing
            '-an',                            # No audio
            '-'
        ]
        return cmd

    def _connect(self) -> bool:
        """Start FFmpeg subprocess."""
        try:
            if self._process:
                try:
                    self._process.terminate()
                    self._process.wait(timeout=2)
                except:
                    pass
                self._process = None

            cmd = self._build_ffmpeg_cmd()
            logger.info(f"Starting FFmpeg for {self.camera_id}...")

            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0  # CRITICAL: No buffering for minimum latency
            )

            # Wait briefly and check if running
            time.sleep(0.5)
            if self._process.poll() is not None:
                stderr = ""
                if self._process.stderr:
                    stderr = self._process.stderr.read().decode()
                logger.error(f"FFmpeg failed for {self.camera_id}: {stderr[:200]}")
                return False

            self._connected = True
            self._reconnects += 1
            logger.info(f"FFmpeg connected: {self.camera_id}")
            return True

        except Exception as e:
            logger.error(f"FFmpeg connection error for {self.camera_id}: {e}")
            self._errors += 1
            return False

    def _stream_loop(self):
        """Main streaming loop."""
        frame_size = self.config.width * self.config.height * 3

        while not self._stop_event.is_set():
            # Connect if needed
            if not self._process or self._process.poll() is not None:
                self._connected = False

                if not self._connect():
                    time.sleep(self.config.reconnect_delay)
                    continue

            # Read frame
            try:
                raw = self._process.stdout.read(frame_size)

                if len(raw) != frame_size:
                    if len(raw) > 0:
                        logger.warning(f"Incomplete frame for {self.camera_id}: {len(raw)}/{frame_size}")
                    self._connected = False
                    try:
                        self._process.terminate()
                    except:
                        pass
                    self._process = None
                    continue

                # Convert to numpy
                frame = np.frombuffer(raw, dtype=np.uint8)
                frame = frame.reshape((self.config.height, self.config.width, 3))

                # Store latest frame
                with self._frame_lock:
                    self._latest_frame = frame
                    self._frame_time = time.time()
                    self._frame_count += 1

                self._frames_read += 1

                # Callback
                if self.on_frame:
                    try:
                        self.on_frame(frame)
                    except Exception as e:
                        logger.error(f"Frame callback error: {e}")

            except Exception as e:
                logger.error(f"FFmpeg read error for {self.camera_id}: {e}")
                self._errors += 1
                self._connected = False

        # Cleanup
        if self._process:
            try:
                self._process.terminate()
            except:
                pass

    def get_stats(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "connected": self._connected,
            "frames_read": self._frames_read,
            "reconnects": self._reconnects,
            "errors": self._errors,
            "frame_age": time.time() - self._frame_time if self._frame_time else None,
            "fps": self.config.fps
        }


class FFmpegStreamManager:
    """Manages multiple FFmpeg streams."""

    def __init__(self):
        self._streams: Dict[str, FFmpegStream] = {}
        self._lock = threading.Lock()

    def add_camera(
        self,
        camera_id: str,
        rtsp_url: str,
        config: Optional[FFmpegConfig] = None,
        on_frame: Optional[Callable] = None
    ) -> FFmpegStream:
        """Add and start a camera stream."""
        with self._lock:
            if camera_id in self._streams:
                self.remove_camera(camera_id)

            stream = FFmpegStream(camera_id, rtsp_url, config, on_frame)
            stream.start()
            self._streams[camera_id] = stream
            return stream

    def remove_camera(self, camera_id: str):
        """Stop and remove a camera stream."""
        with self._lock:
            if camera_id in self._streams:
                self._streams[camera_id].stop()
                del self._streams[camera_id]

    def get_stream(self, camera_id: str) -> Optional[FFmpegStream]:
        """Get stream by camera ID."""
        return self._streams.get(camera_id)

    def get_frame(self, camera_id: str) -> Tuple[Optional[np.ndarray], float]:
        """Get latest frame for a camera."""
        stream = self._streams.get(camera_id)
        if stream:
            return stream.get_frame()
        return None, 0.0

    def get_active_cameras(self) -> list:
        """Get list of active camera IDs."""
        return list(self._streams.keys())

    def stop_all(self):
        """Stop all streams."""
        with self._lock:
            for stream in self._streams.values():
                stream.stop()
            self._streams.clear()

    def get_all_stats(self) -> dict:
        """Get stats for all streams."""
        return {cid: s.get_stats() for cid, s in self._streams.items()}


# Global singleton
_manager: Optional[FFmpegStreamManager] = None


def get_ffmpeg_manager() -> FFmpegStreamManager:
    """Get or create the global FFmpeg stream manager."""
    global _manager
    if _manager is None:
        _manager = FFmpegStreamManager()
    return _manager
