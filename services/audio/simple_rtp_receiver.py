"""Simple Raw UDP/RTP receiver for 16-bit PCM audio @ 16kHz.

Receives RTP packets directly on UDP port, extracts raw PCM audio payload
(skipping 12-byte RTP header), and saves to both WAV and PCM files.

Designed for streams sending raw 16-bit PCM at 16000 Hz sample rate.
"""

import socket
import threading
import time
import numpy as np
import logging
import sys
from typing import Optional, Callable
from datetime import datetime
from pathlib import Path

# Handle imports for both module and standalone execution
try:
    # Try relative import first (when running as module)
    from .audio_writer import AudioWriter
except ImportError:
    # Add parent directory to path for standalone execution
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from services.audio.audio_writer import AudioWriter

logger = logging.getLogger(__name__)


class SimpleRTPReceiver:
    """Simple RTP receiver for raw 16-bit PCM audio @ 16kHz.

    Receives RTP packets, extracts raw PCM audio payload (skipping 12-byte RTP header),
    and saves to WAV and PCM files. No decoding needed - handles raw PCM streams.
    """

    def __init__(
        self,
        listen_host: str = "0.0.0.0",
        listen_port: int = 5004,
        target_sample_rate: int = 16000,
        storage_path: str = "audio_storage/recordings",
        audio_callback: Optional[Callable] = None,
        inactivity_timeout: float = 3.0,
    ):
        """Initialize simple RTP receiver.

        Args:
            listen_host: Host to bind to (default "0.0.0.0")
            listen_port: UDP port to listen on (default 5004)
            target_sample_rate: Target sampling rate in Hz (default 16000)
            storage_path: Path to save audio files
            audio_callback: Optional callback(audio_data, sample_rate) for streaming
            inactivity_timeout: Seconds of inactivity before saving files (default 3.0)
        """
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.target_sample_rate = target_sample_rate
        self.storage_path = storage_path
        self.audio_callback = audio_callback
        self.inactivity_timeout = inactivity_timeout

        # RTP socket
        self._socket: Optional[socket.socket] = None
        self._receive_thread: Optional[threading.Thread] = None
        self._monitor_thread: Optional[threading.Thread] = None

        # State
        self._running = False
        self._session_id = None
        self._remote_address = None
        self._audio_writer: Optional[AudioWriter] = None
        self._pcm_file = None
        self._pcm_path = None
        self._last_packet_time = 0.0
        self._files_lock = threading.Lock()

        # Statistics
        self._stats = {
            "packets_received": 0,
            "bytes_received": 0,
            "payload_bytes_written": 0,
            "packets_too_short": 0,
            "start_time": None,
            "last_packet_time": None,
        }

        logger.info(f"SimpleRTPReceiver initialized on {listen_host}:{listen_port}")
        logger.info(f"Target sampling rate: {target_sample_rate} Hz")

    def start(self):
        """Start listening for RTP packets."""
        if self._running:
            logger.warning("Simple RTP receiver already running")
            return

        logger.info(
            f"Starting simple RTP receiver on {self.listen_host}:{self.listen_port}"
        )

        try:
            # Create UDP socket
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._socket.bind((self.listen_host, self.listen_port))
            self._socket.settimeout(1.0)

            logger.info(
                f"Simple RTP receiver listening on {self.listen_host}:{self.listen_port}"
            )

        except Exception as e:
            logger.error(f"Failed to bind socket: {e}")
            raise

        # Initialize session (files will be created on first packet)
        self._session_id = f"simple_rtp_{int(time.time())}"

        # Start threads
        self._running = True
        self._stats["start_time"] = datetime.now()
        self._last_packet_time = time.time()

        self._receive_thread = threading.Thread(
            target=self._receive_loop, daemon=True, name="SimpleRTP-Receive"
        )
        self._receive_thread.start()

        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="SimpleRTP-Monitor"
        )
        self._monitor_thread.start()

        logger.info("Simple RTP receiver started")

    def stop(self):
        """Stop receiving RTP packets."""
        if not self._running:
            return

        logger.info("Stopping simple RTP receiver...")
        self._running = False

        # Close socket
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass

        # Wait for threads
        if self._receive_thread:
            self._receive_thread.join(timeout=2.0)
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)

        # Close files
        self._close_files()

        logger.info("Simple RTP receiver stopped")
        self._print_stats()

    def get_stats(self) -> dict:
        """Get receiver statistics."""
        return {
            **self._stats,
            "running": self._running,
            "session_id": self._session_id,
            "remote_address": self._remote_address,
            "wav_path": self._audio_writer.filepath if self._audio_writer else None,
            "pcm_path": str(self._pcm_path) if self._pcm_path else None,
        }

    def _setup_output_file(self):
        """Setup WAV and PCM file writers for audio data."""
        try:
            # Ensure session_id is set
            if self._session_id is None:
                self._session_id = f"simple_rtp_{int(time.time())}"

            # Create audio writer for WAV output
            self._audio_writer = AudioWriter(
                output_dir=self.storage_path,
                session_id=self._session_id,
                codec="raw_pcm_16bit",
                sample_rate=self.target_sample_rate,
                channels=1,
            )
            self._audio_writer.open()

            # Create PCM file
            pcm_filename = self._audio_writer.filepath.replace(".wav", ".pcm")
            self._pcm_path = Path(pcm_filename)
            self._pcm_file = open(self._pcm_path, "wb")

            logger.info(f"WAV file created: {self._audio_writer.filepath}")
            logger.info(f"PCM file created: {self._pcm_path}")

        except Exception as e:
            logger.error(f"Failed to setup audio files: {e}")

    def _close_files(self):
        """Close and save audio files."""
        with self._files_lock:
            # Close WAV file
            if self._audio_writer:
                self._audio_writer.close()
                logger.info(f"WAV file saved: {self._audio_writer.filepath}")
                self._audio_writer = None

            # Close PCM file
            if self._pcm_file:
                self._pcm_file.close()
                logger.info(f"PCM file saved: {self._pcm_path}")
                self._pcm_file = None

    def _monitor_loop(self):
        """Monitor for inactivity and auto-save files."""
        logger.info("Monitor loop started")

        while self._running:
            try:
                time.sleep(0.5)  # Check every 500ms

                # Check if we have active files and if there's been inactivity
                if self._audio_writer and self._last_packet_time > 0:
                    elapsed = time.time() - self._last_packet_time

                    if elapsed > self.inactivity_timeout:
                        logger.info(f"No packets for {elapsed:.1f}s, saving files...")
                        self._close_files()
                        # Reset for next session (files will be created on next packet)
                        self._session_id = f"simple_rtp_{int(time.time())}"
                        self._last_packet_time = 0.0

            except Exception as e:
                if self._running:
                    logger.error(f"Monitor error: {e}", exc_info=True)

        logger.info("Monitor loop stopped")

    def _receive_loop(self):
        """Main receive loop."""
        logger.info("Receive loop started")

        while self._running:
            try:
                # Check socket is valid
                if self._socket is None:
                    logger.error("Socket is None in receive loop")
                    break

                # Receive packet (buffer size 32000 like Java code)
                data, addr = self._socket.recvfrom(32000)

                # Track remote address
                if self._remote_address is None:
                    self._remote_address = addr
                    logger.info(f"Detected RTP source: {addr}")

                # Update stats and packet time
                self._stats["packets_received"] += 1
                self._stats["bytes_received"] += len(data)
                self._stats["last_packet_time"] = datetime.now()
                self._last_packet_time = time.time()

                # Process packet
                self._process_packet(data)

            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    logger.error(f"Receive error: {e}", exc_info=True)

        logger.info("Receive loop stopped")

    def _process_packet(self, data: bytes):
        """Process received RTP packet.

        Args:
            data: Raw UDP packet data
        """
        length = len(data)

        # Check minimum RTP header size (12 bytes)
        if length < 12:
            logger.warning(f"Received too short packet: {length} bytes")
            self._stats["packets_too_short"] += 1
            return

        try:
            # Parse RTP header (simplified, matching Java code)
            version = (data[0] >> 6) & 0x03
            payload_type = data[1] & 0x7F
            sequence_number = ((data[2] & 0xFF) << 8) | (data[3] & 0xFF)
            timestamp = (
                ((data[4] & 0xFF) << 24)
                | ((data[5] & 0xFF) << 16)
                | ((data[6] & 0xFF) << 8)
                | (data[7] & 0xFF)
            )
            ssrc = (
                ((data[8] & 0xFF) << 24)
                | ((data[9] & 0xFF) << 16)
                | ((data[10] & 0xFF) << 8)
                | (data[11] & 0xFF)
            )

            payload_size = length - 12

            # Log packet info (similar to Java printf)
            if self._stats["packets_received"] % 100 == 0:  # Log every 100 packets
                logger.info(
                    f"RTP Packet: ver={version}, pt={payload_type} (Raw PCM 16kHz), seq={sequence_number}, "
                    f"ts={timestamp}, ssrc={ssrc}, size={payload_size} bytes"
                )

            # Create files on first packet if not already created
            if self._audio_writer is None:
                logger.info("First packet received, creating audio files...")
                self._setup_output_file()

            # Extract payload (skip 12-byte header) - raw 16-bit PCM data
            # Matching Java code: System.arraycopy(data, 12, usA, 0, length-12)
            payload = data[12:length]

            # Write raw PCM data directly (no decoding needed - already raw PCM @ 16kHz)
            with self._files_lock:
                if self._audio_writer:
                    # Convert bytes to numpy array for AudioWriter (WAV)
                    audio_array = np.frombuffer(payload, dtype=np.int16)
                    self._audio_writer.write(audio_array)
                    self._stats["payload_bytes_written"] += len(payload)

                # Write raw PCM data
                if self._pcm_file:
                    self._pcm_file.write(payload)
                    self._pcm_file.flush()

            # Call streaming callback if provided
            if self.audio_callback:
                try:
                    self.audio_callback(payload, self.target_sample_rate)
                except Exception as e:
                    logger.error(f"Audio callback error: {e}")

        except Exception as e:
            logger.error(f"Packet processing error: {e}", exc_info=True)

    def _print_stats(self):
        """Print session statistics."""
        if self._stats["start_time"]:
            duration = (datetime.now() - self._stats["start_time"]).total_seconds()
            logger.info("Session statistics:")
            logger.info(f"  Duration: {duration:.1f}s")
            logger.info(f"  Packets received: {self._stats['packets_received']}")
            logger.info(f"  Bytes received: {self._stats['bytes_received']}")
            logger.info(
                f"  Payload bytes written: {self._stats['payload_bytes_written']}"
            )
            logger.info(f"  Packets too short: {self._stats['packets_too_short']}")


def main():
    """Test simple RTP receiver."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    receiver = SimpleRTPReceiver(
        listen_host="0.0.0.0",
        listen_port=5004,
        target_sample_rate=16000,
    )

    try:
        receiver.start()
        logger.info("Receiver running. Press Ctrl+C to stop...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        receiver.stop()


if __name__ == "__main__":
    main()
