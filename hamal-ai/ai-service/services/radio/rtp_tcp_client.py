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
        reconnect_delay: float = 5.0
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
            "reconnects": 0,
            "errors": 0,
            "connected": False
        }

        logger.info(f"RTPTCPClient initialized for {host}:{port}")

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
            except:
                pass

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
            logger.info(f"Connected to EC2 relay at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to {self.host}:{self.port}: {e}")
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

                # Parse RTP and extract audio
                self._process_rtp_packet(packet_data)

            except socket.timeout:
                continue
            except Exception as e:
                logger.error(f"Receive error: {e}")
                self._stats["errors"] += 1
                self._stats["connected"] = False
                if self._socket:
                    try:
                        self._socket.close()
                    except:
                        pass
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
        if len(packet) < 12:
            return

        # RTP header is 12 bytes minimum
        # Skip header and get payload
        header_length = 12

        # Check for header extensions
        first_byte = packet[0]
        has_extension = (first_byte & 0x10) != 0

        if has_extension and len(packet) > 16:
            # Extension header: 2 bytes profile + 2 bytes length
            ext_length = struct.unpack(">H", packet[14:16])[0]
            header_length = 16 + (ext_length * 4)

        if len(packet) <= header_length:
            return

        # Extract audio payload
        audio_payload = packet[header_length:]

        # Decode based on payload type (usually G.711 μ-law or PCM)
        payload_type = packet[1] & 0x7F

        if payload_type == 0:
            # G.711 μ-law - decode to PCM
            audio_data = self._decode_ulaw(audio_payload)
        elif payload_type == 8:
            # G.711 A-law - decode to PCM
            audio_data = self._decode_alaw(audio_payload)
        else:
            # Assume raw PCM 16-bit
            audio_data = audio_payload

        if self.audio_callback and audio_data:
            self.audio_callback(audio_data, self.target_sample_rate)

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
            # Fallback - return as-is
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

    def get_stats(self) -> dict:
        """Get client statistics."""
        return {
            "host": self.host,
            "port": self.port,
            "running": self._running,
            **self._stats
        }
