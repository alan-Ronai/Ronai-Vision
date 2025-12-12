"""FFmpeg and GStreamer integration for audio streaming.

Provides utilities to receive audio via FFmpeg or GStreamer pipelines.
Supports RTSP, RTP, HTTP, and file inputs.
"""

import subprocess
import threading
import numpy as np
import logging
from typing import Optional, Callable
import queue

logger = logging.getLogger(__name__)


class FFmpegAudioReceiver:
    """Receive audio using FFmpeg.

    Uses FFmpeg subprocess to decode audio from various sources:
    - RTSP streams
    - RTP streams
    - HTTP streams
    - Files

    Outputs PCM audio to callback for processing.
    """

    def __init__(
        self,
        input_url: str,
        sample_rate: int = 16000,
        channels: int = 1,
        audio_callback: Optional[Callable] = None,
    ):
        """Initialize FFmpeg audio receiver.

        Args:
            input_url: Input URL or file path
            sample_rate: Output sample rate (default 16000)
            channels: Number of audio channels (default 1)
            audio_callback: Callback(audio_chunk, sample_rate) for audio data
        """
        self.input_url = input_url
        self.sample_rate = sample_rate
        self.channels = channels
        self.audio_callback = audio_callback

        # FFmpeg process
        self._process: Optional[subprocess.Popen] = None
        self._read_thread: Optional[threading.Thread] = None

        # State
        self._running = False

        logger.info(f"FFmpegAudioReceiver initialized")
        logger.info(f"  Input: {input_url}")
        logger.info(f"  Sample rate: {sample_rate}Hz")
        logger.info(f"  Channels: {channels}")

    def start(self):
        """Start FFmpeg audio receiver."""
        if self._running:
            logger.warning("FFmpeg receiver already running")
            return

        logger.info("Starting FFmpeg audio receiver...")

        # Build FFmpeg command
        cmd = [
            "ffmpeg",
            "-i",
            self.input_url,
            "-f",
            "s16le",  # PCM signed 16-bit little-endian
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(self.sample_rate),
            "-ac",
            str(self.channels),
            "-",  # Output to stdout
        ]

        try:
            # Start FFmpeg process
            self._process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8
            )

            logger.info(f"FFmpeg process started (PID: {self._process.pid})")

        except Exception as e:
            logger.error(f"Failed to start FFmpeg: {e}")
            raise

        # Start read thread
        self._running = True
        self._read_thread = threading.Thread(
            target=self._read_loop, daemon=True, name="FFmpeg-Read"
        )
        self._read_thread.start()

        logger.info("FFmpeg audio receiver started")

    def stop(self):
        """Stop FFmpeg audio receiver."""
        if not self._running:
            return

        logger.info("Stopping FFmpeg audio receiver...")
        self._running = False

        # Terminate FFmpeg process
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=5.0)
            except:
                self._process.kill()

        # Wait for thread
        if self._read_thread:
            self._read_thread.join(timeout=2.0)

        logger.info("FFmpeg audio receiver stopped")

    def _read_loop(self):
        """Read audio data from FFmpeg stdout."""
        logger.info("FFmpeg read loop started")

        chunk_size = (
            self.sample_rate * self.channels * 2
        )  # 1 second of audio (2 bytes per sample)

        while self._running and self._process:
            try:
                # Read audio chunk
                data = self._process.stdout.read(chunk_size)

                if not data:
                    logger.info("FFmpeg stream ended")
                    break

                # Convert to numpy array
                audio = np.frombuffer(data, dtype=np.int16)

                # Call callback
                if self.audio_callback:
                    try:
                        self.audio_callback(audio, self.sample_rate)
                    except Exception as e:
                        logger.error(f"Audio callback error: {e}")

            except Exception as e:
                if self._running:
                    logger.error(f"FFmpeg read error: {e}", exc_info=True)
                break

        logger.info("FFmpeg read loop stopped")


class GStreamerAudioReceiver:
    """Receive audio using GStreamer.

    Uses GStreamer pipeline to decode audio from various sources.
    Outputs PCM audio to callback for processing.
    """

    def __init__(
        self,
        pipeline_string: Optional[str] = None,
        input_url: Optional[str] = None,
        sample_rate: int = 16000,
        channels: int = 1,
        audio_callback: Optional[Callable] = None,
    ):
        """Initialize GStreamer audio receiver.

        Args:
            pipeline_string: Full GStreamer pipeline string (optional)
            input_url: Input URL (will build pipeline if pipeline_string not provided)
            sample_rate: Output sample rate (default 16000)
            channels: Number of audio channels (default 1)
            audio_callback: Callback(audio_chunk, sample_rate) for audio data
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.audio_callback = audio_callback

        # Build pipeline string if not provided
        if pipeline_string:
            self.pipeline_string = pipeline_string
        elif input_url:
            self.pipeline_string = self._build_pipeline(input_url)
        else:
            raise ValueError("Either pipeline_string or input_url must be provided")

        # GStreamer process
        self._process: Optional[subprocess.Popen] = None
        self._read_thread: Optional[threading.Thread] = None

        # State
        self._running = False

        logger.info(f"GStreamerAudioReceiver initialized")
        logger.info(f"  Pipeline: {self.pipeline_string}")

    def _build_pipeline(self, input_url: str) -> str:
        """Build GStreamer pipeline for input URL.

        Args:
            input_url: Input URL or file path

        Returns:
            GStreamer pipeline string
        """
        # Detect input type
        if input_url.startswith("rtsp://"):
            src = f"rtspsrc location={input_url} ! rtph264depay ! decodebin"
        elif input_url.startswith("rtp://"):
            src = f"udpsrc uri={input_url} ! application/x-rtp ! rtpjitterbuffer ! decodebin"
        elif input_url.startswith("http://") or input_url.startswith("https://"):
            src = f"souphttpsrc location={input_url} ! decodebin"
        else:
            src = f"filesrc location={input_url} ! decodebin"

        # Build full pipeline
        pipeline = (
            f"{src} ! "
            f"audioconvert ! "
            f"audioresample ! "
            f"audio/x-raw,format=S16LE,rate={self.sample_rate},channels={self.channels} ! "
            f"fdsink"
        )

        return pipeline

    def start(self):
        """Start GStreamer audio receiver."""
        if self._running:
            logger.warning("GStreamer receiver already running")
            return

        logger.info("Starting GStreamer audio receiver...")

        # Build gst-launch command
        cmd = ["gst-launch-1.0", self.pipeline_string]

        try:
            # Start GStreamer process
            self._process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8
            )

            logger.info(f"GStreamer process started (PID: {self._process.pid})")

        except Exception as e:
            logger.error(f"Failed to start GStreamer: {e}")
            logger.error(
                "Make sure GStreamer is installed: sudo apt-get install gstreamer1.0-tools"
            )
            raise

        # Start read thread
        self._running = True
        self._read_thread = threading.Thread(
            target=self._read_loop, daemon=True, name="GStreamer-Read"
        )
        self._read_thread.start()

        logger.info("GStreamer audio receiver started")

    def stop(self):
        """Stop GStreamer audio receiver."""
        if not self._running:
            return

        logger.info("Stopping GStreamer audio receiver...")
        self._running = False

        # Terminate GStreamer process
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=5.0)
            except:
                self._process.kill()

        # Wait for thread
        if self._read_thread:
            self._read_thread.join(timeout=2.0)

        logger.info("GStreamer audio receiver stopped")

    def _read_loop(self):
        """Read audio data from GStreamer stdout."""
        logger.info("GStreamer read loop started")

        chunk_size = self.sample_rate * self.channels * 2  # 1 second of audio

        while self._running and self._process:
            try:
                # Read audio chunk
                data = self._process.stdout.read(chunk_size)

                if not data:
                    logger.info("GStreamer stream ended")
                    break

                # Convert to numpy array
                audio = np.frombuffer(data, dtype=np.int16)

                # Call callback
                if self.audio_callback:
                    try:
                        self.audio_callback(audio, self.sample_rate)
                    except Exception as e:
                        logger.error(f"Audio callback error: {e}")

            except Exception as e:
                if self._running:
                    logger.error(f"GStreamer read error: {e}", exc_info=True)
                break

        logger.info("GStreamer read loop stopped")
