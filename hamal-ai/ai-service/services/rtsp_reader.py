"""Robust RTSP reader using FFmpeg subprocess.

OpenCV's RTSP handling is fragile. FFmpeg is much more robust
for handling network errors and H.264 decoding issues.
"""

import subprocess
import numpy as np
import logging
import time
import threading
import cv2
from pathlib import Path
from typing import Optional, Callable, Dict, List
from queue import Queue, Empty
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RTSPConfig:
    """RTSP reader configuration."""
    width: int = 1280
    height: int = 720
    fps: int = 15  # Increased from 5 to 15 for smoother streaming
    reconnect_delay: float = 3.0
    max_reconnect_attempts: int = -1  # -1 = infinite
    tcp_transport: bool = True
    buffer_size: int = 3
    jpeg_quality: int = 70


class FFmpegRTSPReader:
    """Robust RTSP reader using FFmpeg subprocess."""

    def __init__(
        self,
        camera_id: str,
        rtsp_url: str,
        config: Optional[RTSPConfig] = None,
        on_frame: Optional[Callable[[str, np.ndarray], None]] = None
    ):
        self.camera_id = camera_id

        # Resolve local file paths relative to Ronai-Vision root
        if not rtsp_url.startswith(('rtsp://', 'http://', 'https://', 'rtmp://', 'udp://')):
            # Local file path - resolve relative to Ronai-Vision root
            # Current file is at: Ronai-Vision/hamal-ai/ai-service/services/rtsp_reader.py
            # Root is three levels up: ../../../
            script_dir = Path(__file__).parent
            project_root = script_dir.parent.parent.parent  # Ronai-Vision root

            # Clean up the path (remove leading slash if present for relative paths)
            clean_path = rtsp_url.lstrip('/')

            # Try as relative path first (from project root)
            video_path = project_root / clean_path

            # If doesn't exist as relative, try as absolute
            if not video_path.exists() and Path(rtsp_url).is_absolute():
                video_path = Path(rtsp_url)

            # Convert to absolute path
            resolved_path = str(video_path.absolute())

            # Check if file exists
            if video_path.exists():
                logger.info(f"Camera {camera_id}: ✓ Resolved local video: {resolved_path}")
                rtsp_url = resolved_path
            else:
                logger.error(f"Camera {camera_id}: ✗ Video file not found: {resolved_path}")
                logger.error(f"Camera {camera_id}: Looked in project root: {project_root}")
                # Keep original path and let FFmpeg fail with proper error

        self.rtsp_url = rtsp_url
        self.config = config or RTSPConfig()
        self.on_frame = on_frame

        self._process: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._frame_queue: Queue = Queue(maxsize=self.config.buffer_size)
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()

        self._reconnect_count = 0
        self._is_connected = False
        self._stats = {
            "frames_read": 0,
            "errors": 0,
            "reconnects": 0
        }

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def start(self):
        """Start reading frames in background thread."""
        if self._thread and self._thread.is_alive():
            logger.warning(f"Camera {self.camera_id}: Already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        logger.info(f"Camera {self.camera_id}: Started FFmpeg RTSP reader")

    def stop(self):
        """Stop reading frames."""
        self._stop_event.set()

        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None

        if self._thread:
            self._thread.join(timeout=5)

        self._is_connected = False
        logger.info(f"Camera {self.camera_id}: Stopped FFmpeg RTSP reader")

    def get_frame(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Get latest frame."""
        with self._frame_lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()
        return None

    def _build_ffmpeg_command(self) -> list:
        """Build FFmpeg command with H264 error recovery."""
        # Check if this is an RTSP/network stream vs a local file
        is_network_stream = self.rtsp_url.startswith(('rtsp://', 'http://', 'https://', 'rtmp://', 'udp://'))

        cmd = [
            'ffmpeg',
            '-hide_banner',
            '-loglevel', 'warning',  # Show warnings but reduce spam
        ]

        # Add RTSP-specific options only for network streams
        if is_network_stream:
            cmd.extend([
                # RTSP transport - TCP is more reliable for H264
                '-rtsp_transport', 'tcp' if self.config.tcp_transport else 'udp',
                '-rtsp_flags', 'prefer_tcp',
                '-timeout', '5000000',          # 5 second timeout (in microseconds)

                # Buffer settings - balance between latency and stability
                # NOTE: nobuffer causes H264 decode errors - using genpts+discardcorrupt instead
                '-fflags', '+genpts+discardcorrupt',  # Generate PTS, discard corrupt packets
                '-flags', 'low_delay',

                # Moderate probe/analyze - not too aggressive for camera compatibility
                '-probesize', '2000000',        # 2MB probe (reduced for compatibility)
                '-analyzeduration', '1000000',  # 1s analysis (reduced)

                # H264 error handling - CRITICAL for corrupted streams
                '-err_detect', 'ignore_err',    # Continue despite errors
                '-ec', 'guess_mvs+deblock',     # Error concealment: guess motion vectors + deblock

                # Reasonable buffer to handle network jitter
                '-max_delay', '500000',         # 500ms max delay
                '-reorder_queue_size', '5',     # Reduced from 10 for lower latency
            ])
        else:
            # For local video files - loop infinitely
            cmd.extend([
                '-stream_loop', '-1',           # Loop video infinitely
                '-re',                          # Read at native frame rate (prevents CPU spike)
            ])

        # Input file/stream
        cmd.extend([
            '-i', self.rtsp_url,

            # Output options
            '-vf', f'fps={self.config.fps},scale={self.config.width}:{self.config.height}',
            '-pix_fmt', 'bgr24',
            '-f', 'rawvideo',
            '-fps_mode', 'cfr' if is_network_stream else 'passthrough',  # CFR for streams, passthrough for files
            '-an',  # No audio
            '-'
        ])

        return cmd

    def _connect(self) -> bool:
        """Start FFmpeg subprocess."""
        try:
            if self._process:
                self._process.terminate()
                try:
                    self._process.wait(timeout=2)
                except:
                    self._process.kill()
                self._process = None

            cmd = self._build_ffmpeg_command()
            logger.info(f"Camera {self.camera_id}: Starting FFmpeg...")

            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**7  # 10MB buffer for large frames
            )

            # Wait a bit and check if process is running
            time.sleep(1.0)
            if self._process.poll() is not None:
                stderr = self._process.stderr.read().decode() if self._process.stderr else ""
                logger.error(f"Camera {self.camera_id}: FFmpeg failed: {stderr[:500]}")
                return False

            self._is_connected = True
            self._reconnect_count = 0
            logger.info(f"Camera {self.camera_id}: FFmpeg connected")
            return True

        except Exception as e:
            logger.error(f"Camera {self.camera_id}: Connection error: {e}")
            return False

    def _read_loop(self):
        """Main reading loop."""
        frame_size = self.config.width * self.config.height * 3

        while not self._stop_event.is_set():
            # Connect if needed
            if not self._process or self._process.poll() is not None:
                self._is_connected = False

                if self.config.max_reconnect_attempts >= 0:
                    if self._reconnect_count >= self.config.max_reconnect_attempts:
                        logger.error(f"Camera {self.camera_id}: Max reconnect attempts reached")
                        break

                self._reconnect_count += 1
                self._stats["reconnects"] += 1
                logger.info(f"Camera {self.camera_id}: Reconnecting (attempt {self._reconnect_count})...")

                if not self._connect():
                    time.sleep(self.config.reconnect_delay)
                    continue

            # Read frame
            try:
                raw_frame = self._process.stdout.read(frame_size)

                if len(raw_frame) != frame_size:
                    if len(raw_frame) == 0:
                        logger.warning(f"Camera {self.camera_id}: Stream ended")
                    else:
                        logger.warning(f"Camera {self.camera_id}: Incomplete frame ({len(raw_frame)}/{frame_size})")
                    self._stats["errors"] += 1
                    self._is_connected = False
                    if self._process:
                        self._process.terminate()
                    self._process = None
                    continue

                # Convert to numpy array
                frame = np.frombuffer(raw_frame, dtype=np.uint8)
                frame = frame.reshape((self.config.height, self.config.width, 3))

                self._stats["frames_read"] += 1

                # Store latest frame
                with self._frame_lock:
                    self._latest_frame = frame

                # Call callback
                if self.on_frame:
                    try:
                        self.on_frame(self.camera_id, frame)
                    except Exception as e:
                        logger.error(f"Camera {self.camera_id}: Frame callback error: {e}")

            except Exception as e:
                logger.error(f"Camera {self.camera_id}: Read error: {e}")
                self._stats["errors"] += 1
                self._is_connected = False

        # Cleanup
        if self._process:
            self._process.terminate()

    def get_stats(self) -> dict:
        """Get reader statistics."""
        return {
            **self._stats,
            "camera_id": self.camera_id,
            "connected": self._is_connected,
            "reconnect_count": self._reconnect_count,
            "has_frame": self._latest_frame is not None
        }


class RTSPReaderManager:
    """Manages multiple RTSP readers."""

    def __init__(self):
        self._readers: Dict[str, FFmpegRTSPReader] = {}
        self._frame_callbacks: List[Callable] = []
        self._lock = threading.Lock()

    def add_frame_callback(self, callback: Callable[[str, np.ndarray], None]):
        """Add callback for all frames."""
        self._frame_callbacks.append(callback)

    def _on_frame(self, camera_id: str, frame: np.ndarray):
        """Internal frame handler that calls all callbacks."""
        for callback in self._frame_callbacks:
            try:
                callback(camera_id, frame)
            except Exception as e:
                logger.error(f"Frame callback error: {e}")

    def add_camera(
        self,
        camera_id: str,
        rtsp_url: str,
        config: Optional[RTSPConfig] = None
    ) -> FFmpegRTSPReader:
        """Add and start a camera."""
        with self._lock:
            if camera_id in self._readers:
                self.remove_camera(camera_id)

            reader = FFmpegRTSPReader(
                camera_id=camera_id,
                rtsp_url=rtsp_url,
                config=config,
                on_frame=self._on_frame
            )
            reader.start()
            self._readers[camera_id] = reader
            return reader

    def remove_camera(self, camera_id: str):
        """Stop and remove a camera."""
        with self._lock:
            if camera_id in self._readers:
                self._readers[camera_id].stop()
                del self._readers[camera_id]

    def get_reader(self, camera_id: str) -> Optional[FFmpegRTSPReader]:
        """Get reader by camera ID."""
        return self._readers.get(camera_id)

    def get_frame(self, camera_id: str) -> Optional[np.ndarray]:
        """Get latest frame for camera."""
        reader = self._readers.get(camera_id)
        if reader:
            return reader.get_frame()
        return None

    def get_all_stats(self) -> dict:
        """Get stats for all readers."""
        return {cid: r.get_stats() for cid, r in self._readers.items()}

    def get_active_cameras(self) -> List[str]:
        """Get list of active camera IDs."""
        return list(self._readers.keys())

    def stop_all(self):
        """Stop all readers."""
        with self._lock:
            for reader in self._readers.values():
                reader.stop()
            self._readers.clear()


# Global singleton
_manager: Optional[RTSPReaderManager] = None

def get_rtsp_manager() -> RTSPReaderManager:
    """Get or create RTSP reader manager."""
    global _manager
    if _manager is None:
        _manager = RTSPReaderManager()
    return _manager
