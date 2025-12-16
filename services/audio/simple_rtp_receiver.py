"""Simple Raw UDP/RTP receiver with payload extraction and multi-format testing.

Receives RTP packets directly on UDP port, extracts audio payload,
and saves multiple test files with different decoding strategies:

Test files created:
1. raw_payload - No decoding (raw RTP payload)
2. adpcm_decoded - DVI4/IMA ADPCM decoded (PT 5)
3. g711_mulaw - G.711 mu-law decoded (PT 0)
4. g711_alaw - G.711 A-law decoded (PT 8)
5. raw_pcm_16bit_le - Raw 16-bit PCM little-endian
6. raw_pcm_16bit_be - Raw 16-bit PCM big-endian (byte-swapped)
7. raw_pcm_8bit - 8-bit PCM converted to 16-bit

Each test creates both a .wav and .pcm file for comparison.
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

        # ADPCM decoder state (needed for stateful decoding)
        self._adpcm_state = None

        # Multiple test output files
        self._test_writers = {}
        self._test_files = {}

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
        test_files_info = {}
        for config_name, writer in self._test_writers.items():
            if writer:
                test_files_info[config_name] = writer.filepath

        return {
            **self._stats,
            "running": self._running,
            "session_id": self._session_id,
            "remote_address": self._remote_address,
            "test_files": test_files_info,
        }

    def _setup_output_file(self):
        """Setup multiple test WAV and PCM files for different decoding strategies."""
        try:
            # Create base output directory
            base_dir = Path(self.storage_path)
            base_dir.mkdir(parents=True, exist_ok=True)

            # Define all test configurations
            test_configs = [
                ("raw_payload", "Raw payload (no decoding)"),
                ("adpcm_decoded", "ADPCM decoded (DVI4)"),
                ("g711_mulaw", "G.711 mu-law decoded"),
                ("g711_alaw", "G.711 A-law decoded"),
                ("raw_pcm_16bit_le", "Raw 16-bit PCM little-endian"),
                ("raw_pcm_16bit_be", "Raw 16-bit PCM big-endian"),
                ("raw_pcm_8bit", "Raw 8-bit PCM"),
            ]

            logger.info(f"Creating {len(test_configs)} test output files...")

            for config_name, description in test_configs:
                # Create AudioWriter for WAV
                writer = AudioWriter(
                    output_dir=self.storage_path,
                    session_id=f"{self._session_id}_{config_name}",
                    codec="rtp_pcm",
                    sample_rate=self.target_sample_rate,
                    channels=1,
                )
                writer.open()
                self._test_writers[config_name] = writer

                # Create PCM file
                pcm_filename = writer.filepath.replace(".wav", ".pcm")
                pcm_path = Path(pcm_filename)
                self._test_files[config_name] = open(pcm_path, "wb")

                logger.info(f"  [{config_name}] {description}")
                logger.info(f"    WAV: {writer.filepath}")
                logger.info(f"    PCM: {pcm_path}")

        except Exception as e:
            logger.error(f"Failed to setup audio files: {e}")

    def _close_files(self):
        """Close and save all test audio files."""
        with self._files_lock:
            logger.info("Closing and saving test audio files...")

            # Close all test WAV writers
            for config_name, writer in self._test_writers.items():
                if writer:
                    writer.close()
                    logger.info(f"  [{config_name}] WAV saved: {writer.filepath}")

            # Close all test PCM files
            for config_name, pcm_file in self._test_files.items():
                if pcm_file:
                    pcm_file.close()
                    logger.info(f"  [{config_name}] PCM saved")

            # Clear dictionaries
            self._test_writers = {}
            self._test_files = {}

    def _monitor_loop(self):
        """Monitor for inactivity and auto-save files."""
        logger.info("Monitor loop started")

        while self._running:
            try:
                time.sleep(0.5)  # Check every 500ms

                # Check if we have active files and if there's been inactivity
                if len(self._test_writers) > 0 and self._last_packet_time > 0:
                    elapsed = time.time() - self._last_packet_time

                    if elapsed > self.inactivity_timeout:
                        logger.info(f"No packets for {elapsed:.1f}s, saving files...")
                        self._close_files()
                        # Reset for next session (files will be created on next packet)
                        self._session_id = f"simple_rtp_{int(time.time())}"
                        self._last_packet_time = 0.0
                        self._adpcm_state = None  # Reset decoder state

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
                codec_name = {
                    0: "PCMU (G.711 mu-law)",
                    5: "DVI4 (IMA ADPCM)",
                    8: "PCMA (G.711 A-law)",
                }.get(payload_type, "Unknown/Raw PCM")
                logger.info(
                    f"RTP Packet: ver={version}, pt={payload_type} ({codec_name}), seq={sequence_number}, "
                    f"ts={timestamp}, ssrc={ssrc}, size={payload_size} bytes"
                )

            # Create files on first packet if not already created
            if len(self._test_writers) == 0:
                logger.info("First packet received, creating audio files...")
                self._setup_output_file()

            # Extract payload (skip 12-byte header)
            # Matching Java code: System.arraycopy(data, 12, usA, 0, length-12)
            payload = data[12:length]

            # Prepare all decoding variants for testing
            test_outputs = {}

            # 1. Raw payload (no decoding at all)
            test_outputs["raw_payload"] = payload

            # 2. ADPCM decoded (DVI4) - PT 5
            try:
                decoded_adpcm, self._adpcm_state = audioop.adpcm2lin(
                    payload, 2, self._adpcm_state
                )
                test_outputs["adpcm_decoded"] = decoded_adpcm
            except Exception as e:
                logger.warning(f"ADPCM decode failed: {e}")
                test_outputs["adpcm_decoded"] = payload

            # 3. G.711 mu-law decoded - PT 0
            try:
                decoded_mulaw = audioop.ulaw2lin(payload, 2)
                test_outputs["g711_mulaw"] = decoded_mulaw
            except Exception as e:
                logger.warning(f"G.711 mu-law decode failed: {e}")
                test_outputs["g711_mulaw"] = payload

            # 4. G.711 A-law decoded - PT 8
            try:
                decoded_alaw = audioop.alaw2lin(payload, 2)
                test_outputs["g711_alaw"] = decoded_alaw
            except Exception as e:
                logger.warning(f"G.711 A-law decode failed: {e}")
                test_outputs["g711_alaw"] = payload

            # 5. Raw 16-bit PCM little-endian (assume it's already PCM)
            test_outputs["raw_pcm_16bit_le"] = payload

            # 6. Raw 16-bit PCM big-endian (swap byte order)
            try:
                # Convert to numpy array and swap byte order
                pcm_array = np.frombuffer(payload, dtype=np.int16)
                pcm_array_be = pcm_array.byteswap()
                test_outputs["raw_pcm_16bit_be"] = pcm_array_be.tobytes()
            except Exception as e:
                logger.warning(f"Big-endian conversion failed: {e}")
                test_outputs["raw_pcm_16bit_be"] = payload

            # 7. Raw 8-bit PCM (convert to 16-bit by expanding)
            try:
                # Treat as 8-bit unsigned, convert to 16-bit signed
                pcm_8bit = np.frombuffer(payload, dtype=np.uint8)
                # Convert 0-255 to -32768 to 32767
                pcm_16bit = ((pcm_8bit.astype(np.int32) - 128) * 256).astype(np.int16)
                test_outputs["raw_pcm_8bit"] = pcm_16bit.tobytes()
            except Exception as e:
                logger.warning(f"8-bit PCM conversion failed: {e}")
                test_outputs["raw_pcm_8bit"] = payload

            # Write all variants to their respective files
            with self._files_lock:
                for config_name, audio_data in test_outputs.items():
                    try:
                        writer = self._test_writers.get(config_name)
                        pcm_file = self._test_files.get(config_name)

                        if writer and pcm_file:
                            # Ensure data is 16-bit PCM for WAV writer
                            try:
                                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                            except Exception:
                                # If conversion fails, pad or truncate to make it valid
                                if len(audio_data) % 2 != 0:
                                    audio_data = audio_data + b'\x00'
                                audio_array = np.frombuffer(audio_data, dtype=np.int16)

                            writer.write(audio_array)
                            pcm_file.write(audio_data)
                            pcm_file.flush()

                    except Exception as e:
                        logger.error(f"Error writing {config_name}: {e}")

                self._stats["payload_bytes_written"] += len(payload)

            # Call streaming callback with ADPCM decoded version (most likely correct)
            if self.audio_callback:
                try:
                    self.audio_callback(
                        test_outputs["adpcm_decoded"], self.target_sample_rate
                    )
                except Exception as e:
                    logger.error(f"Audio callback error: {e}")

        except Exception as e:
            logger.error(f"Packet processing error: {e}", exc_info=True)

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
