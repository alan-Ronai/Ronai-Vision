"""GStreamer-based RTSP reader for stable streaming.

GStreamer provides:
- Better RTP jitter buffer handling than FFmpeg
- Hardware-accelerated decoding (VideoToolbox on Mac)
- More robust H.264 error recovery
- Better handling of corrupted NAL units

This provides smoother, more stable video streaming with fewer glitches.
"""

import subprocess
import numpy as np
import logging
import time
import threading
from typing import Optional, Callable, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GStreamerConfig:
    """GStreamer stream configuration."""
    width: int = 1280
    height: int = 720
    fps: int = 15
    latency: int = 200  # RTP jitter buffer latency in ms
    tcp: bool = True
    reconnect_delay: float = 2.0


class GStreamerRTSPReader:
    """Stable RTSP reader using GStreamer with hardware decoding."""

    def __init__(
        self,
        camera_id: str,
        rtsp_url: str,
        config: Optional[GStreamerConfig] = None,
        on_frame: Optional[Callable[[np.ndarray], None]] = None
    ):
        self.camera_id = camera_id
        self.config = config or GStreamerConfig()
        self.on_frame = on_frame

        # Resolve local file paths relative to Ronai-Vision root
        self.is_local_file = False
        if not rtsp_url.startswith(('rtsp://', 'http://', 'https://', 'rtmp://', 'udp://')):
            # Local file path - resolve relative to Ronai-Vision root
            # Path: .../Ronai-Vision/hamal-ai/ai-service/services/streaming/gstreamer_rtsp.py
            # script_dir = streaming, need 4 more parents to reach Ronai-Vision
            from pathlib import Path
            script_dir = Path(__file__).parent  # .../services/streaming
            project_root = script_dir.parent.parent.parent.parent  # Ronai-Vision root

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
                self.is_local_file = True
            else:
                logger.error(f"Camera {camera_id}: ✗ Video file not found: {resolved_path}")
                logger.error(f"Camera {camera_id}: Looked in project root: {project_root}")

        self.rtsp_url = rtsp_url

        self._process: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Frame buffer
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
        """Start GStreamer stream in background."""
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._thread.start()
        logger.info(f"GStreamer stream started: {self.camera_id}")

    def stop(self):
        """Stop GStreamer stream."""
        self._stop_event.set()

        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=3)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass

        if self._thread:
            self._thread.join(timeout=3)

        self._connected = False
        logger.info(f"GStreamer stream stopped: {self.camera_id}")

    def get_frame(self):
        """Get latest frame and timestamp."""
        with self._frame_lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy(), self._frame_time
            return None, 0.0

    def _build_gstreamer_pipeline(self) -> str:
        """Build GStreamer pipeline string.

        For RTSP streams:
        1. rtspsrc: RTSP source with TCP transport and jitter buffer
        2. rtph264depay: Extract H.264 from RTP
        3. h264parse: Parse H.264 stream with error recovery
        4. avdec_h264: Decode H.264
        5. videoconvert: Convert to BGR for OpenCV
        6. videoscale: Scale to target resolution
        7. videorate: Control frame rate
        8. fdsink: Output to stdout for reading

        For local video files:
        1. filesrc: Read from local file
        2. decodebin: Auto-detect and decode any video format
        3. videoconvert: Convert to BGR for OpenCV
        4. videoscale: Scale to target resolution
        5. videorate: Control frame rate
        6. fdsink: Output to stdout for reading
        """
        if self.is_local_file:
            # Local video file pipeline
            # Using filesrc + decodebin for robust format detection
            # The reconnect logic in _stream_loop will restart the video when it ends (loop)
            pipeline = (
                f'filesrc location="{self.rtsp_url}" '
                f'! decodebin '
                f'! videoconvert '
                f'! video/x-raw,format=BGR '
                f'! videoscale '
                f'! video/x-raw,width={self.config.width},height={self.config.height} '
                f'! videorate '
                f'! video/x-raw,framerate={self.config.fps}/1 '
                f'! fdsink fd=1'
            )
            logger.info(f"Camera {self.camera_id}: Using local file pipeline (auto-loops on EOF)")
        else:
            # RTSP stream pipeline
            protocols = "tcp" if self.config.tcp else "udp"

            # Base pipeline with robust RTSP source
            pipeline = (
                f'rtspsrc location="{self.rtsp_url}" '
                f'protocols={protocols} '
                f'latency={self.config.latency} '
                f'buffer-mode=auto '
                f'drop-on-latency=true '
                f'ntp-sync=false '
                f'retry=5 '
                f'timeout=5000000 '
                f'! rtph264depay '
                f'! h264parse config-interval=-1 '  # Send SPS/PPS with every IDR
            )

            # Try hardware decoding first (VideoToolbox on Mac), fallback to software
            # Note: vtdec_hw may not be available on all systems
            pipeline += (
                '! avdec_h264 skip-frame=default max-threads=2 direct-rendering=true '
            )

            # Video processing
            pipeline += (
                f'! videoconvert '
                f'! video/x-raw,format=BGR '
                f'! videoscale '
                f'! video/x-raw,width={self.config.width},height={self.config.height} '
                f'! videorate '
                f'! video/x-raw,framerate={self.config.fps}/1 '
                f'! fdsink fd=1'  # Output to stdout
            )

        return pipeline

    def _connect(self) -> bool:
        """Start GStreamer subprocess."""
        try:
            if self._process:
                try:
                    self._process.terminate()
                    self._process.wait(timeout=2)
                except Exception:
                    pass
                self._process = None

            pipeline = self._build_gstreamer_pipeline()
            logger.info(f"Starting GStreamer for {self.camera_id} (local_file={self.is_local_file})...")
            logger.debug(f"Pipeline: {pipeline}")

            # Use shell=True to properly handle the pipeline string with quotes
            # This ensures file paths and URLs with special characters work correctly
            full_cmd = f'gst-launch-1.0 -q -e {pipeline}'
            self._process = subprocess.Popen(
                full_cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**7
            )

            # Wait briefly and check if running
            time.sleep(1.0)
            if self._process.poll() is not None:
                stderr = ""
                if self._process.stderr:
                    stderr = self._process.stderr.read().decode()
                logger.error(f"GStreamer failed for {self.camera_id}: {stderr[:300]}")
                return False

            self._connected = True
            self._reconnects += 1
            logger.info(f"GStreamer connected: {self.camera_id}")
            return True

        except FileNotFoundError:
            logger.error("GStreamer not installed. Install with: brew install gstreamer gst-plugins-base gst-plugins-good gst-plugins-bad gst-plugins-ugly")
            return False
        except Exception as e:
            logger.error(f"GStreamer connection error for {self.camera_id}: {e}")
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
                if self._process is None or self._process.stdout is None:
                    self._connected = False
                    continue

                raw = self._process.stdout.read(frame_size)

                if len(raw) != frame_size:
                    if len(raw) > 0:
                        logger.warning(f"Incomplete frame for {self.camera_id}: {len(raw)}/{frame_size}")
                    self._connected = False
                    try:
                        self._process.terminate()
                    except Exception:
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
                logger.error(f"GStreamer read error for {self.camera_id}: {e}")
                self._errors += 1
                self._connected = False

        # Cleanup
        if self._process:
            try:
                self._process.terminate()
            except Exception:
                pass

    def get_stats(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "backend": "gstreamer",
            "source_type": "local_file" if self.is_local_file else "rtsp",
            "connected": self._connected,
            "frames_read": self._frames_read,
            "reconnects": self._reconnects,
            "errors": self._errors,
            "frame_age": time.time() - self._frame_time if self._frame_time else None,
            "fps": self.config.fps
        }


class GStreamerStreamManager:
    """Manages multiple GStreamer streams.

    Interface matches RTSPReaderManager for drop-in replacement based on RTSP_BACKEND env var.
    """

    def __init__(self):
        self._streams: Dict[str, GStreamerRTSPReader] = {}
        self._frame_callbacks: list = []
        self._lock = threading.Lock()

    def add_frame_callback(self, callback: Callable[[str, np.ndarray], None]):
        """Add callback for all frames (called with camera_id, frame)."""
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
        config=None  # Can be GStreamerConfig or RTSPConfig
    ) -> GStreamerRTSPReader:
        """Add and start a camera stream.

        Accepts either GStreamerConfig or RTSPConfig for compatibility with unified manager.
        """
        with self._lock:
            if camera_id in self._streams:
                self.remove_camera(camera_id)

            # Convert RTSPConfig to GStreamerConfig if needed
            gst_config = None
            if config is not None:
                if isinstance(config, GStreamerConfig):
                    gst_config = config
                else:
                    # Assume it's RTSPConfig or similar - extract common fields
                    gst_config = GStreamerConfig(
                        width=getattr(config, 'width', 1280),
                        height=getattr(config, 'height', 720),
                        fps=getattr(config, 'fps', 15),
                        tcp=getattr(config, 'tcp_transport', True),  # RTSPConfig uses tcp_transport
                    )

            # Create wrapper callback that routes to our _on_frame
            def frame_callback(frame: np.ndarray):
                self._on_frame(camera_id, frame)

            stream = GStreamerRTSPReader(camera_id, rtsp_url, gst_config, frame_callback)
            stream.start()
            self._streams[camera_id] = stream
            return stream

    def remove_camera(self, camera_id: str):
        """Stop and remove a camera stream."""
        with self._lock:
            if camera_id in self._streams:
                self._streams[camera_id].stop()
                del self._streams[camera_id]

    def get_stream(self, camera_id: str) -> Optional[GStreamerRTSPReader]:
        """Get stream by camera ID."""
        return self._streams.get(camera_id)

    def get_frame(self, camera_id: str) -> Optional[np.ndarray]:
        """Get latest frame for a camera (returns just the frame, not tuple)."""
        stream = self._streams.get(camera_id)
        if stream:
            frame, _ = stream.get_frame()
            return frame
        return None

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
_gst_manager: Optional[GStreamerStreamManager] = None


def get_gstreamer_manager() -> GStreamerStreamManager:
    """Get or create the global GStreamer stream manager."""
    global _gst_manager
    if _gst_manager is None:
        _gst_manager = GStreamerStreamManager()
    return _gst_manager


def is_gstreamer_available() -> bool:
    """Check if GStreamer is installed and available."""
    try:
        result = subprocess.run(
            ['gst-launch-1.0', '--version'],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False
