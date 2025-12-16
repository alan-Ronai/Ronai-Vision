#!/usr/bin/env python3
"""RTP TCP client - runs on your local machine.

Connects to EC2 RTP relay server and receives RTP packets over TCP.
Saves to audio files locally. No router configuration needed!
"""

import socket
import threading
import time
import logging
import struct
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class RTPTCPClient:
    """Connects to RTP relay server and saves audio locally."""

    def __init__(
        self,
        server_host: str,
        server_port: int = 5005,
        sample_rate: int = 16000,
        storage_path: str = "audio_storage/recordings",
        inactivity_timeout: float = 3.0,
    ):
        """Initialize RTP TCP client.

        Args:
            server_host: EC2 server hostname/IP
            server_port: TCP port to connect to (default 5005)
            sample_rate: Audio sample rate (default 16000)
            storage_path: Path to save audio files
            inactivity_timeout: Seconds of inactivity before saving files
        """
        self.server_host = server_host
        self.server_port = server_port
        self.sample_rate = sample_rate
        self.storage_path = storage_path
        self.inactivity_timeout = inactivity_timeout

        # Socket
        self._socket: Optional[socket.socket] = None

        # Threads
        self._receive_thread: Optional[threading.Thread] = None
        self._monitor_thread: Optional[threading.Thread] = None

        # State
        self._running = False
        self._session_id = None
        self._last_packet_time = 0.0
        self._files_lock = threading.Lock()

        # Audio files
        self._wav_file = None
        self._pcm_file = None
        self._wav_path = None
        self._pcm_path = None

        # Statistics
        self._stats = {
            "packets_received": 0,
            "bytes_received": 0,
            "audio_bytes_written": 0,
            "start_time": None,
        }

        logger.info("RTPTCPClient initialized")
        logger.info(f"  Server: {server_host}:{server_port}")
        logger.info(f"  Sample rate: {sample_rate} Hz")

    def start(self):
        """Connect to server and start receiving."""
        if self._running:
            logger.warning("Client already running")
            return

        logger.info(f"Connecting to {self.server_host}:{self.server_port}...")

        try:
            # Connect to TCP server
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.connect((self.server_host, self.server_port))
            self._socket.settimeout(1.0)
            logger.info("Connected to RTP relay server!")

        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            raise

        # Initialize session
        self._session_id = f"rtp_tcp_{int(time.time())}"
        self._running = True
        self._stats["start_time"] = datetime.now()
        self._last_packet_time = time.time()

        # Start threads
        self._receive_thread = threading.Thread(
            target=self._receive_loop, daemon=True, name="RTP-Receive"
        )
        self._receive_thread.start()

        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="Monitor"
        )
        self._monitor_thread.start()

        logger.info("Client started, receiving RTP packets...")

    def stop(self):
        """Disconnect and stop receiving."""
        if not self._running:
            return

        logger.info("Stopping client...")
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

        logger.info("Client stopped")
        self._print_stats()

    def _setup_output_files(self):
        """Setup WAV and PCM output files."""
        try:
            if self._session_id is None:
                self._session_id = f"rtp_tcp_{int(time.time())}"

            # Create output directory
            output_dir = Path(self.storage_path)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Generate filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"{timestamp}_{self._session_id}"

            self._wav_path = output_dir / f"{base_name}.wav"
            self._pcm_path = output_dir / f"{base_name}.pcm"

            # Open WAV file
            import wave
            self._wav_file = wave.open(str(self._wav_path), 'wb')
            self._wav_file.setnchannels(1)  # Mono
            self._wav_file.setsampwidth(2)  # 16-bit
            self._wav_file.setframerate(self.sample_rate)

            # Open PCM file
            self._pcm_file = open(self._pcm_path, 'wb')

            logger.info(f"WAV file created: {self._wav_path}")
            logger.info(f"PCM file created: {self._pcm_path}")

        except Exception as e:
            logger.error(f"Failed to setup output files: {e}")

    def _close_files(self):
        """Close and save audio files."""
        with self._files_lock:
            if self._wav_file:
                self._wav_file.close()
                logger.info(f"WAV file saved: {self._wav_path}")
                self._wav_file = None

            if self._pcm_file:
                self._pcm_file.close()
                logger.info(f"PCM file saved: {self._pcm_path}")
                self._pcm_file = None

    def _receive_loop(self):
        """Receive RTP packets from TCP connection."""
        logger.info("Receive loop started")

        while self._running:
            try:
                if self._socket is None:
                    break

                # Read packet length (4 bytes, network byte order)
                length_bytes = self._recv_exact(4)
                if not length_bytes:
                    logger.warning("Connection closed by server")
                    break

                packet_length = struct.unpack("!I", length_bytes)[0]

                # Read RTP packet data
                rtp_data = self._recv_exact(packet_length)
                if not rtp_data:
                    logger.warning("Incomplete packet received")
                    break

                self._stats["packets_received"] += 1
                self._stats["bytes_received"] += len(rtp_data)
                self._last_packet_time = time.time()

                # Process RTP packet
                self._process_rtp_packet(rtp_data)

            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    logger.error(f"Receive error: {e}", exc_info=True)
                break

        logger.info("Receive loop stopped")
        self._running = False

    def _recv_exact(self, n: int) -> Optional[bytes]:
        """Receive exactly n bytes from socket."""
        data = b''
        while len(data) < n:
            try:
                chunk = self._socket.recv(n - len(data))
                if not chunk:
                    return None
                data += chunk
            except socket.timeout:
                if not self._running:
                    return None
                continue
        return data

    def _process_rtp_packet(self, rtp_data: bytes):
        """Process received RTP packet and extract audio."""
        # Create files on first packet
        if self._wav_file is None:
            logger.info("First packet received, creating audio files...")
            self._setup_output_files()

        # Extract payload (skip 12-byte RTP header)
        if len(rtp_data) < 12:
            return

        payload = rtp_data[12:]

        # Write to files
        with self._files_lock:
            if self._wav_file:
                # Convert to numpy array and write to WAV
                try:
                    audio_array = np.frombuffer(payload, dtype=np.int16)
                    pcm_bytes = audio_array.astype(np.int16).tobytes()
                    self._wav_file.writeframes(pcm_bytes)
                    self._stats["audio_bytes_written"] += len(payload)
                except Exception as e:
                    logger.error(f"WAV write error: {e}")

            if self._pcm_file:
                self._pcm_file.write(payload)
                self._pcm_file.flush()

    def _monitor_loop(self):
        """Monitor for inactivity and auto-save files."""
        logger.info("Monitor loop started")

        while self._running:
            try:
                time.sleep(0.5)

                # Check for inactivity
                if self._wav_file and self._last_packet_time > 0:
                    elapsed = time.time() - self._last_packet_time

                    if elapsed > self.inactivity_timeout:
                        logger.info(f"No packets for {elapsed:.1f}s, saving files...")
                        self._close_files()
                        # Reset for next session
                        self._session_id = f"rtp_tcp_{int(time.time())}"
                        self._last_packet_time = 0.0

            except Exception as e:
                if self._running:
                    logger.error(f"Monitor error: {e}", exc_info=True)

        logger.info("Monitor loop stopped")

    def _print_stats(self):
        """Print session statistics."""
        if self._stats["start_time"]:
            duration = (datetime.now() - self._stats["start_time"]).total_seconds()
            logger.info("Session statistics:")
            logger.info(f"  Duration: {duration:.1f}s")
            logger.info(f"  Packets received: {self._stats['packets_received']}")
            logger.info(f"  Bytes received: {self._stats['bytes_received']}")
            logger.info(f"  Audio bytes written: {self._stats['audio_bytes_written']}")


def main():
    """Run RTP TCP client."""
    import argparse

    parser = argparse.ArgumentParser(
        description="RTP TCP Client - Connect to EC2 relay and save audio locally"
    )
    parser.add_argument(
        "--server",
        required=True,
        help="EC2 server hostname or IP address",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5005,
        help="TCP port to connect to (default: 5005)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Audio sample rate in Hz (default: 16000)",
    )
    parser.add_argument(
        "--output-dir",
        default="audio_storage/recordings",
        help="Directory to save audio files (default: audio_storage/recordings)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    client = RTPTCPClient(
        server_host=args.server,
        server_port=args.port,
        sample_rate=args.sample_rate,
        storage_path=args.output_dir,
    )

    try:
        client.start()
        logger.info("Client running. Press Ctrl+C to stop...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        client.stop()


if __name__ == "__main__":
    main()
