"""RTP TCP Client - Connects to EC2 relay server for RTP audio.

Instead of receiving RTP directly via UDP (which requires port forwarding),
this client connects to an EC2 relay server via TCP that forwards the RTP packets.

Architecture:
    Radio → RTP/UDP → EC2 Relay → TCP:5005 → This Client → Gemini → UI
"""

import socket
import struct
import time
import logging
from threading import Thread, Event
from typing import Optional, Callable
import numpy as np

logger = logging.getLogger(__name__)


class RTPTCPClient:
    """TCP client that receives RTP packets from EC2 relay server."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5005,
        target_sample_rate: int = 16000,
        audio_callback: Optional[Callable[[bytes, int], None]] = None,
        reconnect_delay: float = 5.0,
    ):
        """Initialize TCP client.

        Args:
            host: EC2 relay server hostname/IP
            port: TCP port to connect to
            target_sample_rate: Target audio sample rate
            audio_callback: Called with (audio_bytes, sample_rate) when audio received
            reconnect_delay: Seconds to wait before reconnecting
        """
        self.host = host
        self.port = port
        self.target_sample_rate = target_sample_rate
        self.audio_callback = audio_callback
        self.reconnect_delay = reconnect_delay

        self._socket: Optional[socket.socket] = None
        self._running = False
        self._thread: Optional[Thread] = None
        self._stop_event = Event()

        # Stats
        self._stats = {
            "packets_received": 0,
            "bytes_received": 0,
            "audio_callbacks": 0,
            "audio_bytes_sent": 0,
            "reconnects": 0,
            "errors": 0,
            "connected": False,
            "last_packet_time": None,
            "connection_uptime": 0.0,
        }

        # Connection monitoring
        self._last_packet_time = 0.0
        self._connection_start_time = 0.0

        logger.info(
            f"RTPTCPClient initialized for {host}:{port} @ {target_sample_rate}Hz"
        )

    def start(self):
        """Start the TCP client."""
        if self._running:
            return

        self._running = True
        self._stop_event.clear()
        self._thread = Thread(target=self._receive_loop, daemon=True)
        self._thread.start()
        logger.info(f"RTPTCPClient started, connecting to {self.host}:{self.port}")

    def stop(self):
        """Stop the TCP client."""
        self._running = False
        self._stop_event.set()

        if self._socket:
            try:
                self._socket.close()
            except Exception as e:
                logger.debug(f"Error closing socket during stop: {e}")

        if self._thread:
            self._thread.join(timeout=2)

        logger.info("RTPTCPClient stopped")

    def _connect(self) -> bool:
        """Establish TCP connection to relay server."""
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(10.0)
            self._socket.connect((self.host, self.port))
            self._stats["connected"] = True
            self._stats["reconnects"] += 1
            self._connection_start_time = time.time()
            logger.info(
                f"✅ Connected to EC2 relay at {self.host}:{self.port} (attempt #{self._stats['reconnects']})"
            )
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to {self.host}:{self.port}: {e}")
            self._stats["errors"] += 1
            self._stats["connected"] = False
            return False

    def _receive_loop(self):
        """Main receive loop running in background thread."""
        while self._running and not self._stop_event.is_set():
            # Connect if needed
            if not self._socket or not self._stats["connected"]:
                if not self._connect():
                    time.sleep(self.reconnect_delay)
                    continue

            try:
                # Read packet length (4 bytes, big-endian)
                length_data = self._recv_exact(4)
                if not length_data:
                    raise ConnectionError("Connection closed")

                packet_length = struct.unpack(">I", length_data)[0]

                # Sanity check
                if packet_length > 65535:
                    logger.warning(f"Invalid packet length: {packet_length}")
                    continue

                # Read the RTP packet
                packet_data = self._recv_exact(packet_length)
                if not packet_data:
                    raise ConnectionError("Connection closed")

                self._stats["packets_received"] += 1
                self._stats["bytes_received"] += len(packet_data)
                self._last_packet_time = time.time()
                self._stats["last_packet_time"] = time.time()
                self._stats["connection_uptime"] = (
                    time.time() - self._connection_start_time
                )

                # Log connection health every 100 packets
                if self._stats["packets_received"] % 100 == 0:
                    uptime_mins = self._stats["connection_uptime"] / 60
                    logger.info(
                        f"📶 Connection healthy: {self._stats['packets_received']} packets, "
                        f"{self._stats['bytes_received'] / 1024:.1f} KB, "
                        f"uptime {uptime_mins:.1f}m"
                    )

                # Parse RTP and extract audio
                self._process_rtp_packet(packet_data)

            except socket.timeout:
                # Check if we haven't received packets in a while
                if self._last_packet_time > 0:
                    silence_duration = time.time() - self._last_packet_time
                    if silence_duration > 30:  # 30 seconds of silence
                        logger.warning(
                            f"⚠️  No packets received for {silence_duration:.1f}s - "
                            f"connection may be dead"
                        )
                continue
            except ConnectionError as e:
                logger.error(f"🔌 Connection lost: {e}")
                self._stats["errors"] += 1
                self._stats["connected"] = False
                if self._socket:
                    try:
                        self._socket.close()
                    except Exception as close_err:
                        logger.debug(f"Error closing socket: {close_err}")
                    self._socket = None
                logger.info(f"Will retry connection in {self.reconnect_delay}s...")
                time.sleep(self.reconnect_delay)
            except Exception as e:
                logger.error(f"Receive error: {e}", exc_info=True)
                self._stats["errors"] += 1
                self._stats["connected"] = False
                if self._socket:
                    try:
                        self._socket.close()
                    except Exception as close_err:
                        logger.debug(f"Error closing socket: {close_err}")
                    self._socket = None
                time.sleep(1)

    def _recv_exact(self, num_bytes: int) -> Optional[bytes]:
        """Receive exactly num_bytes from socket."""
        data = b""
        while len(data) < num_bytes:
            try:
                chunk = self._socket.recv(num_bytes - len(data))
                if not chunk:
                    return None
                data += chunk
            except socket.timeout:
                if not self._running:
                    return None
                continue
        return data

    def _process_rtp_packet(self, packet: bytes):
        """Process an RTP packet and extract audio.

        Args:
            packet: Raw RTP packet bytes
        """
        length = len(packet)

        # Check minimum RTP header size (12 bytes)
        if length < 12:
            logger.warning(f"Received too short packet: {length} bytes")
            self._stats["errors"] += 1
            return

        try:
            # Parse RTP header (matching Java/simple_rtp_receiver code)
            version = (packet[0] >> 6) & 0x03
            payload_type = packet[1] & 0x7F
            sequence_number = ((packet[2] & 0xFF) << 8) | (packet[3] & 0xFF)
            timestamp = (
                ((packet[4] & 0xFF) << 24)
                | ((packet[5] & 0xFF) << 16)
                | ((packet[6] & 0xFF) << 8)
                | (packet[7] & 0xFF)
            )
            ssrc = (
                ((packet[8] & 0xFF) << 24)
                | ((packet[9] & 0xFF) << 16)
                | ((packet[10] & 0xFF) << 8)
                | (packet[11] & 0xFF)
            )

            payload_size = length - 12

            # Log packet info every 100 packets
            if self._stats["packets_received"] % 100 == 0:
                logger.debug(
                    f"RTP Packet: ver={version}, pt={payload_type}, seq={sequence_number}, "
                    f"ts={timestamp}, ssrc={ssrc:08X}, size={payload_size} bytes"
                )

            # Extract payload (skip 12-byte header) - raw PCM data
            payload = packet[12:length]

            # Determine remote sampling rate based on payload type
            # Matching Java: payloadType == 4 ? AUDIO_SAMPLING_RATE_LOW : AUDIO_SAMPLING_RATE
            if payload_type == 4 or payload_type == 0:  # PCMU
                # logger.warning("payload type 0/4 (8000) detected")
                remote_sample_rate = 8000  # Low sample rate
            else:
                # Raw PCM or other format - assume target rate
                # logger.warning(f"payload type {payload_type} ({self.target_sample_rate}) detected")
                remote_sample_rate = self.target_sample_rate

            # Handle sampling rate conversion if needed
            if self.target_sample_rate == remote_sample_rate:
                # No conversion needed
                output_data = payload
            else:
                # Upsample if needed
                output_data = self._upsample(
                    payload, remote_sample_rate, self.target_sample_rate
                )

            # Call audio callback
            if self.audio_callback and output_data:
                self._stats["audio_callbacks"] += 1
                self._stats["audio_bytes_sent"] += len(output_data)

                # Log every 50th callback to confirm audio flow
                if self._stats["audio_callbacks"] % 50 == 1:
                    duration_ms = (
                        len(output_data) / 2 / self.target_sample_rate
                    ) * 1000
                    logger.info(
                        f"📡 RTP audio decoded: {len(output_data)} bytes, "
                        f"{duration_ms:.1f}ms @ {self.target_sample_rate}Hz, PT={payload_type}, "
                        f"seq={sequence_number} (callback #{self._stats['audio_callbacks']})"
                    )

                self.audio_callback(output_data, self.target_sample_rate)

        except Exception as e:
            logger.error(f"RTP packet processing error: {e}", exc_info=True)
            self._stats["errors"] += 1

    def _decode_ulaw(self, data: bytes) -> bytes:
        """Decode μ-law audio to 16-bit PCM."""
        try:
            import audioop

            return audioop.ulaw2lin(data, 2)
        except ImportError:
            # Manual μ-law decode
            ulaw_table = self._build_ulaw_table()
            samples = np.array([ulaw_table[b] for b in data], dtype=np.int16)
            return samples.tobytes()

    def _decode_alaw(self, data: bytes) -> bytes:
        """Decode A-law audio to 16-bit PCM."""
        try:
            import audioop

            return audioop.alaw2lin(data, 2)
        except ImportError:
            logger.warning("audioop not available, A-law decoding disabled")
            # Fallback - return as-is (will sound wrong but won't crash)
            return data
        except Exception as e:
            logger.error(f"A-law decode error: {e}")
            return data

    def _build_ulaw_table(self) -> list:
        """Build μ-law to linear conversion table."""
        table = []
        for i in range(256):
            val = ~i
            sign = val & 0x80
            exponent = (val >> 4) & 0x07
            mantissa = val & 0x0F
            sample = (mantissa << 3) + 0x84
            sample <<= exponent
            sample -= 0x84
            if sign:
                sample = -sample
            table.append(sample)
        return table

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
            # logger.warning(f"Upsampling from {from_rate}Hz to {to_rate}Hz")
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

    def get_stats(self) -> dict:
        """Get client statistics."""
        return {
            "host": self.host,
            "port": self.port,
            "running": self._running,
            **self._stats,
        }
