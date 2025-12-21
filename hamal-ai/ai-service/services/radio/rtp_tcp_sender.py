"""RTP Audio Sender - Sends audio to EC2 relay server for radio transmission.

Architecture:
    - PTT Commands: TCP socket on port 12345 (persistent connection)
    - Audio Data: UDP socket on port 12347 (RTP packets)

Commands:
    - /ptt: Start transmission (key up radio)
    - /pts: Stop transmission (key down radio)

Environment Variables:
    - EC2_RTP_HOST: EC2 server hostname/IP
    - EC2_CMD_PORT: TCP port for PTT commands (default: 12345)
    - EC2_AUDIO_PORT: UDP port for RTP audio (default: 12347)
"""

import socket
import struct
import time
import logging
import threading
from threading import Lock
from typing import Optional
import numpy as np
import os

logger = logging.getLogger(__name__)


class RTPAudioSender:
    """Sends audio to EC2 relay server with PTT signaling.

    Uses two sockets:
    - Persistent TCP connection for PTT commands (/ptt, /pts)
    - UDP socket for RTP audio packets
    """

    # RTP constants
    RTP_VERSION = 2
    PAYLOAD_TYPE_8KHZ = 4   # PT 4 = 8000Hz sample rate
    PAYLOAD_TYPE_16KHZ = 5  # PT 5 = 16000Hz sample rate
    SSRC = 0x12345678       # Synchronization source identifier

    def __init__(
        self,
        host: str = "localhost",
        cmd_port: int = 12345,
        audio_port: int = 12347,
        sample_rate: int = 16000,
        auto_connect: bool = True,
    ):
        """Initialize RTP Audio Sender.

        Args:
            host: EC2 relay server hostname/IP
            cmd_port: TCP port for PTT commands (default 12345)
            audio_port: UDP port for RTP audio (default 12347)
            sample_rate: Audio sample rate (default 16000Hz)
            auto_connect: If True, connect sockets on init
        """
        self.host = host
        self.cmd_port = cmd_port
        self.audio_port = audio_port
        self.sample_rate = sample_rate

        # Sockets
        self._cmd_socket: Optional[socket.socket] = None  # TCP for commands
        self._audio_socket: Optional[socket.socket] = None  # UDP for audio
        self._cmd_connected = False
        self._audio_ready = False
        self._lock = Lock()

        # RTP sequence and timestamp tracking
        self._sequence_number = 0
        self._timestamp = 0
        self._samples_per_packet = 320  # 20ms at 16kHz

        # Stats
        self._stats = {
            "packets_sent": 0,
            "bytes_sent": 0,
            "ptt_signals": 0,
            "errors": 0,
            "cmd_connected": False,
            "audio_ready": False,
            "last_send_time": None,
        }

        logger.info(
            f"RTPAudioSender initialized: CMD={host}:{cmd_port} (TCP), "
            f"AUDIO={host}:{audio_port} (UDP) @ {sample_rate}Hz"
        )

        if auto_connect:
            self.connect()

    def connect(self) -> bool:
        """Connect both sockets (TCP for commands, UDP for audio).

        Returns:
            True if both connections successful
        """
        cmd_ok = self._connect_command_socket()
        audio_ok = self._setup_audio_socket()
        return cmd_ok and audio_ok

    def _connect_command_socket(self) -> bool:
        """Establish persistent TCP connection for PTT commands."""
        with self._lock:
            if self._cmd_connected:
                return True

            try:
                logger.info(f"Connecting to command server {self.host}:{self.cmd_port} (TCP)...")
                self._cmd_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._cmd_socket.settimeout(10.0)
                self._cmd_socket.connect((self.host, self.cmd_port))
                self._cmd_connected = True
                self._stats["cmd_connected"] = True
                logger.info(f"Connected to command server {self.host}:{self.cmd_port}")
                return True
            except socket.timeout:
                logger.error(f"Command connection timed out to {self.host}:{self.cmd_port}")
                self._stats["errors"] += 1
                return False
            except ConnectionRefusedError:
                logger.error(f"Command connection refused to {self.host}:{self.cmd_port}")
                self._stats["errors"] += 1
                return False
            except Exception as e:
                logger.error(f"Command connection error: {e}")
                self._stats["errors"] += 1
                return False

    def _setup_audio_socket(self) -> bool:
        """Setup UDP socket for audio transmission."""
        with self._lock:
            if self._audio_ready:
                return True

            try:
                logger.info(f"Setting up audio socket for {self.host}:{self.audio_port} (UDP)...")
                self._audio_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self._audio_ready = True
                self._stats["audio_ready"] = True
                logger.info(f"Audio socket ready for {self.host}:{self.audio_port}")
                return True
            except Exception as e:
                logger.error(f"Audio socket setup error: {e}")
                self._stats["errors"] += 1
                return False

    def disconnect(self):
        """Close both sockets."""
        with self._lock:
            if self._cmd_socket:
                try:
                    self._cmd_socket.close()
                except Exception:
                    pass
                self._cmd_socket = None
            self._cmd_connected = False
            self._stats["cmd_connected"] = False

            if self._audio_socket:
                try:
                    self._audio_socket.close()
                except Exception:
                    pass
                self._audio_socket = None
            self._audio_ready = False
            self._stats["audio_ready"] = False

            logger.info("Disconnected all sockets")

    def _send_command(self, command: str) -> bool:
        """Send a command over TCP connection.

        Args:
            command: Command to send (e.g., '/ptt', '/pts')

        Returns:
            True if sent successfully
        """
        if not self._cmd_connected:
            if not self._connect_command_socket():
                return False

        try:
            with self._lock:
                logger.info(f"Sending command: {command}")
                self._cmd_socket.sendall(f"{command}\n".encode('utf-8'))
                self._stats["ptt_signals"] += 1
                logger.info(f"Command '{command}' sent successfully")
                return True
        except Exception as e:
            logger.error(f"Command send error: {e}")
            self._stats["errors"] += 1
            self._cmd_connected = False
            self._stats["cmd_connected"] = False
            return False

    def send_ptt_start(self) -> bool:
        """Send PTT start signal (/ptt) to key up the radio."""
        return self._send_command("/ptt")

    def send_ptt_stop(self) -> bool:
        """Send PTT stop signal (/pts) to key down the radio."""
        return self._send_command("/pts")

    def _create_rtp_packet(self, audio_data: bytes) -> bytes:
        """Create an RTP packet with the given audio payload.

        RTP Header format (12 bytes):
        - Byte 0: V=2, P=0, X=0, CC=0 -> 0x80
        - Byte 1: M=0, PT=payload_type (4=8kHz, 5=16kHz)
        - Bytes 2-3: Sequence number (big-endian)
        - Bytes 4-7: Timestamp (big-endian)
        - Bytes 8-11: SSRC (big-endian)
        """
        # Select payload type based on sample rate
        if self.sample_rate == 8000:
            payload_type = self.PAYLOAD_TYPE_8KHZ
        else:
            payload_type = self.PAYLOAD_TYPE_16KHZ

        # Build header
        header = struct.pack(
            ">BBHII",
            0x80,  # V=2, P=0, X=0, CC=0
            payload_type,  # M=0, PT=4 or 5
            self._sequence_number & 0xFFFF,
            self._timestamp & 0xFFFFFFFF,
            self.SSRC,
        )

        # Increment sequence and timestamp
        self._sequence_number = (self._sequence_number + 1) & 0xFFFF
        num_samples = len(audio_data) // 2  # 16-bit samples
        self._timestamp = (self._timestamp + num_samples) & 0xFFFFFFFF

        return header + audio_data

    def send_audio(self, audio_data: bytes) -> bool:
        """Send audio data as RTP packets over UDP.

        Args:
            audio_data: Raw 16-bit PCM audio bytes

        Returns:
            True if sent successfully
        """
        if not self._audio_ready:
            if not self._setup_audio_socket():
                return False

        try:
            # Split audio into chunks (20ms at sample rate)
            chunk_size = self._samples_per_packet * 2  # bytes per chunk
            total_packets = (len(audio_data) + chunk_size - 1) // chunk_size
            duration_ms = (len(audio_data) / 2 / self.sample_rate) * 1000

            logger.info(
                f"📤 Starting RTP audio send: {len(audio_data)} bytes, "
                f"~{total_packets} packets, {duration_ms:.0f}ms @ {self.sample_rate}Hz"
            )

            packets_sent_before = self._stats["packets_sent"]

            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                if len(chunk) == 0:
                    continue

                # Create RTP packet
                rtp_packet = self._create_rtp_packet(chunk)

                # Send UDP packet
                self._audio_socket.sendto(rtp_packet, (self.host, self.audio_port))

                self._stats["packets_sent"] += 1
                self._stats["bytes_sent"] += len(rtp_packet)

                # Small delay for real-time pacing (~20ms per packet)
                time.sleep(0.018)

            self._stats["last_send_time"] = time.time()
            packets_sent = self._stats["packets_sent"] - packets_sent_before

            logger.info(
                f"📤 RTP audio send complete: {packets_sent} packets sent to "
                f"{self.host}:{self.audio_port}"
            )

            return True

        except Exception as e:
            logger.error(f"Audio send error: {e}")
            self._stats["errors"] += 1
            return False

    def send_audio_with_ptt(self, audio_data: bytes, ptt_delay: float = 0.1) -> bool:
        """Send audio with automatic PTT signaling.

        Args:
            audio_data: Raw 16-bit PCM audio bytes
            ptt_delay: Delay after PTT start before sending audio

        Returns:
            True if transmission completed successfully
        """
        # Send PTT start
        if not self.send_ptt_start():
            logger.error("Failed to send PTT start")
            return False

        # Wait for radio to key up
        time.sleep(ptt_delay)

        success = False
        try:
            # Send audio
            success = self.send_audio(audio_data)
        finally:
            # Always send PTT stop
            time.sleep(0.1)  # Small delay before releasing PTT
            self.send_ptt_stop()

        return success

    def resample_audio(
        self,
        audio_data: bytes,
        from_rate: int,
        to_rate: Optional[int] = None
    ) -> bytes:
        """Resample audio to target sample rate.

        Args:
            audio_data: Input audio bytes (16-bit PCM)
            from_rate: Source sample rate
            to_rate: Target sample rate (defaults to self.sample_rate)

        Returns:
            Resampled audio bytes
        """
        if to_rate is None:
            to_rate = self.sample_rate

        if from_rate == to_rate:
            return audio_data

        try:
            samples = np.frombuffer(audio_data, dtype=np.int16)
            ratio = to_rate / from_rate
            new_num_samples = int(len(samples) * ratio)

            old_indices = np.arange(len(samples))
            new_indices = np.linspace(0, len(samples) - 1, new_num_samples)
            resampled = np.interp(new_indices, old_indices, samples).astype(np.int16)

            return resampled.tobytes()

        except Exception as e:
            logger.error(f"Resampling error: {e}")
            return audio_data

    def get_stats(self) -> dict:
        """Get sender statistics."""
        return {
            "host": self.host,
            "cmd_port": self.cmd_port,
            "audio_port": self.audio_port,
            "sample_rate": self.sample_rate,
            **self._stats,
        }


# Singleton instance
_sender_instance: Optional[RTPAudioSender] = None
_sender_lock = Lock()


def get_sender(
    host: Optional[str] = None,
    cmd_port: Optional[int] = None,
    audio_port: Optional[int] = None,
    **kwargs
) -> RTPAudioSender:
    """Get or create the singleton RTP sender instance.

    Args:
        host: EC2 host (uses EC2_RTP_HOST env var if not specified)
        cmd_port: TCP command port (uses EC2_CMD_PORT env var, defaults to 12345)
        audio_port: UDP audio port (uses EC2_AUDIO_PORT env var, defaults to 12347)
        **kwargs: Additional arguments for RTPAudioSender

    Returns:
        RTPAudioSender instance
    """
    global _sender_instance

    with _sender_lock:
        if _sender_instance is None:
            _host = host or os.getenv("EC2_RTP_HOST", "localhost")
            _cmd_port = cmd_port or int(os.getenv("EC2_CMD_PORT", "12345"))
            _audio_port = audio_port or int(os.getenv("EC2_AUDIO_PORT", "12347"))

            _sender_instance = RTPAudioSender(
                host=_host,
                cmd_port=_cmd_port,
                audio_port=_audio_port,
                **kwargs
            )

        return _sender_instance


def send_audio_to_radio(
    audio_data: bytes,
    sample_rate: int = 16000,
    auto_ptt: bool = True
) -> bool:
    """Convenience function to send audio to radio.

    Args:
        audio_data: Raw 16-bit PCM audio bytes
        sample_rate: Sample rate of input audio
        auto_ptt: If True, automatically send PTT signals

    Returns:
        True if transmission successful
    """
    sender = get_sender()

    # Resample if needed
    if sample_rate != sender.sample_rate:
        audio_data = sender.resample_audio(audio_data, sample_rate, sender.sample_rate)

    if auto_ptt:
        return sender.send_audio_with_ptt(audio_data)
    else:
        return sender.send_audio(audio_data)
