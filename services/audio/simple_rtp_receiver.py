"""Simple Raw UDP/RTP receiver with payload extraction.

Receives RTP packets directly on UDP port, extracts audio payload,
and handles sampling rate conversion similar to PTT protocol.
"""

import socket
import struct
import threading
import time
import numpy as np
import logging
import sys
import audioop
from typing import Optional, Callable
from datetime import datetime
from pathlib import Path

# Handle imports for both module and standalone execution
try:
    from services.audio.audio_writer import AudioWriter
except ImportError:
    # Try relative import if running as part of package
    try:
        from .audio_writer import AudioWriter
    except ImportError:
        # Add parent directory to path for standalone execution
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from services.audio.audio_writer import AudioWriter

logger = logging.getLogger(__name__)


class SimpleRTPReceiver:
    """Simple RTP receiver for raw UDP packets with payload extraction.

    Receives RTP packets, extracts audio payload (skipping 12-byte header),
    and optionally resamples audio based on payload type.
    """

    def __init__(
        self,
        listen_host: str = "0.0.0.0",
        listen_port: int = 5004,
        target_sample_rate: int = 8000,
        storage_path: str = "audio_storage/recordings",
        audio_callback: Optional[Callable] = None,
        inactivity_timeout: float = 3.0,
    ):
        """Initialize simple RTP receiver.

        Args:
            listen_host: Host to bind to (default "0.0.0.0")
            listen_port: UDP port to listen on (default 5004)
            target_sample_rate: Target sampling rate in Hz (default 8000)
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

        # Try to initialize G.729 decoder - try multiple implementations
        self._g729_decoder = None
        self._g729_available = False

        # Try pygn729
        try:
            import pygn729

            self._g729_decoder = pygn729.G729Decoder()
            self._g729_available = True
            logger.info("G.729 decoder initialized (pygn729)")
        except ImportError:
            pass

        # If that didn't work, log that G.729 is not available
        if not self._g729_available:
            logger.warning("G.729 decoder not available")
            logger.info(
                "Note: G.729 is a proprietary codec with limited Python support"
            )
            logger.info("Alternatives: Install pygn729 or use ffmpeg with subprocess")

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
            except:
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
        """Setup both WAV and PCM file writers for audio data."""
        try:
            # Create audio writer for WAV output
            self._audio_writer = AudioWriter(
                output_dir=self.storage_path,
                session_id=self._session_id,
                codec="rtp_pcm",
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
            is_first_or_last = (data[1] & 0x80) == 0x80  # Marker bit
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
                    f"RTP Packet: ver={version}, pt={payload_type}, seq={sequence_number}, "
                    f"ts={timestamp}, ssrc={ssrc}, size={payload_size} bytes"
                )

            # Create files on first packet if not already created
            if self._audio_writer is None:
                logger.info("First packet received, creating audio files...")
                logger.info(
                    f"Payload type: {payload_type}, payload size: {payload_size}"
                )

                # Save first packet raw for analysis
                try:
                    raw_sample_dir = Path(self.storage_path)
                    raw_sample_dir.mkdir(parents=True, exist_ok=True)
                    raw_sample_path = (
                        raw_sample_dir
                        / f"raw_sample_pt{payload_type}_{int(time.time())}.bin"
                    )
                    with open(raw_sample_path, "wb") as f:
                        f.write(data[12:length][: min(1000, payload_size)])
                    logger.info(f"Saved raw payload sample to: {raw_sample_path}")
                except Exception as e:
                    logger.warning(f"Could not save raw sample: {e}")

                self._setup_output_file()

            # Extract payload (skip 12-byte header)
            payload = data[12:length]

            # Decode based on RTP payload type (RFC 3551)
            # 0 = PCMU (G.711 μ-law)
            # 8 = PCMA (G.711 A-law)
            # 18 = G.729
            # 5 = Trying G.729 (custom implementation)

            if payload_type == 0:
                # G.711 μ-law - 8-bit compressed to 16-bit linear PCM
                logger.info("Decoding as G.711 μ-law (PCMU)")
                output_data = audioop.ulaw2lin(payload, 2)
            elif payload_type == 8:
                # G.711 A-law - 8-bit compressed to 16-bit linear PCM
                logger.info("Decoding as G.711 A-law (PCMA)")
                output_data = audioop.alaw2lin(payload, 2)
            elif payload_type == 5:
                # Payload type 5 - could be DVI4 (IMA ADPCM) or custom
                # Since G.729 decoder isn't available, let's try treating as:
                # 1. Raw 16-bit PCM
                # 2. 8-bit unsigned PCM (convert to 16-bit signed)
                logger.info("Payload type 5: analyzing format...")

                # Try as raw 16-bit PCM first
                output_data = payload

                # Alternative: if it sounds wrong, uncomment to try 8-bit conversion:
                # Convert 8-bit unsigned to 16-bit signed PCM
                # audio_8bit = np.frombuffer(payload, dtype=np.uint8)
                # audio_16bit = ((audio_8bit.astype(np.int16) - 128) * 256)
                # output_data = audio_16bit.tobytes()

                logger.info(f"Using raw PCM (payload size: {len(payload)} bytes)")
            elif payload_type == 18 and self._g729_decoder:
                # G.729 compressed audio
                logger.info("Decoding as G.729")
                output_data = self._decode_g729(payload)
            elif payload_type == 4:
                # Payload type 4 - treat as raw PCM
                logger.info(
                    f"Payload type {payload_type}: treating as raw PCM (no decoding)"
                )
                output_data = payload
            else:
                # Unknown payload type - treat as raw 16-bit PCM
                logger.info(f"Payload type {payload_type}: treating as raw 16-bit PCM")
                output_data = payload
                if self._audio_writer:
                    # Convert bytes to numpy array for AudioWriter (WAV)
                    audio_array = np.frombuffer(output_data, dtype=np.int16)
                    self._audio_writer.write(audio_array)
                    self._stats["payload_bytes_written"] += len(output_data)

                # Write raw PCM data
                if self._pcm_file:
                    self._pcm_file.write(output_data)
                    self._pcm_file.flush()

            # Call streaming callback if provided
            if self.audio_callback:
                try:
                    self.audio_callback(output_data, self.target_sample_rate)
                except Exception as e:
                    logger.error(f"Audio callback error: {e}")

        except Exception as e:
            logger.error(f"Packet processing error: {e}", exc_info=True)

    def _decode_g729(self, payload: bytes) -> bytes:
        """Decode G.729 compressed audio to PCM.

        Args:
            payload: G.729 encoded bytes

        Returns:
            16-bit PCM audio bytes
        """
        try:
            if self._g729_decoder:
                # G.729 frames are 10 bytes (8 kbps, 10ms frame)
                pcm_samples = []
                for i in range(0, len(payload), 10):
                    frame = payload[i : i + 10]
                    if len(frame) == 10:
                        decoded = self._g729_decoder.decode(frame)
                        pcm_samples.extend(decoded)

                # Convert to bytes
                pcm_array = np.array(pcm_samples, dtype=np.int16)
                return pcm_array.tobytes()
            else:
                logger.warning("G.729 decoder not available")
                return payload
        except Exception as e:
            logger.error(f"G.729 decode error: {e}")
            return payload

    def _upsample(self, audio_data: bytes, from_rate: int, to_rate: int) -> bytes:
        """Upsample audio data from one sample rate to another.

        Args:
            audio_data: Input audio bytes
            from_rate: Source sample rate
            to_rate: Target sample rate

        Returns:
            Resampled audio bytes
        """
        try:
            # Convert bytes to numpy array (assuming 16-bit PCM)
            samples = np.frombuffer(audio_data, dtype=np.int16)

            # Calculate resampling ratio
            ratio = to_rate / from_rate

            # Simple linear interpolation for upsampling
            num_samples = len(samples)
            new_num_samples = int(num_samples * ratio)

            # Create new sample indices
            old_indices = np.arange(num_samples)
            new_indices = np.linspace(0, num_samples - 1, new_num_samples)

            # Interpolate
            resampled = np.interp(new_indices, old_indices, samples)

            # Convert back to int16
            resampled = resampled.astype(np.int16)

            return resampled.tobytes()

        except Exception as e:
            logger.error(f"Upsampling error: {e}")
            return audio_data  # Return original on error

    def _print_stats(self):
        """Print session statistics."""
        if self._stats["start_time"]:
            duration = (datetime.now() - self._stats["start_time"]).total_seconds()
            logger.info(f"Session statistics:")
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
        target_sample_rate=8000,
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
