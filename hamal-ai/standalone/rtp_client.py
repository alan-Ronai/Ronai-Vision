"""RTP/TCP Client for receiving audio from EC2 relay.

Connects to EC2 relay server via TCP and decodes RTP audio packets.
"""

import socket
import struct
import threading
import time
import logging
from typing import Optional, Callable
from datetime import datetime

logger = logging.getLogger(__name__)


class RTPTCPClient:
    """TCP client that receives RTP audio packets from EC2 relay."""

    def __init__(
        self,
        host: str,
        port: int,
        target_sample_rate: int = 16000,
        audio_callback: Optional[Callable[[bytes, int], None]] = None,
    ):
        """Initialize RTP/TCP client.

        Args:
            host: EC2 relay hostname/IP
            port: EC2 relay TCP port
            target_sample_rate: Target sample rate for audio
            audio_callback: Callback for decoded audio (audio_bytes, sample_rate)
        """
        self.host = host
        self.port = port
        self.target_sample_rate = target_sample_rate
        self.audio_callback = audio_callback

        self._socket: Optional[socket.socket] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._reconnect_delay = 5.0
        self._max_reconnect_delay = 60.0

        # Statistics
        self._stats = {
            "packets_received": 0,
            "bytes_received": 0,
            "connection_attempts": 0,
            "last_packet_time": None,
            "connected_at": None,
        }

        logger.info(f"RTPTCPClient initialized for {host}:{port}")

    def start(self):
        """Start the client in a background thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info(f"RTPTCPClient started, connecting to {self.host}:{self.port}")

    def stop(self):
        """Stop the client."""
        self._running = False
        if self._socket:
            try:
                self._socket.close()
            except:
                pass
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("RTPTCPClient stopped")

    def _run_loop(self):
        """Main connection/receive loop."""
        reconnect_delay = self._reconnect_delay

        while self._running:
            try:
                self._connect()
                reconnect_delay = self._reconnect_delay  # Reset on success
                self._receive_loop()
            except Exception as e:
                if self._running:
                    logger.error(f"Connection error: {e}")
                    logger.info(f"Reconnecting in {reconnect_delay:.1f}s...")
                    time.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 1.5, self._max_reconnect_delay)

    def _connect(self):
        """Establish TCP connection to relay."""
        self._stats["connection_attempts"] += 1

        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.settimeout(30.0)
        self._socket.connect((self.host, self.port))
        self._socket.settimeout(None)

        self._stats["connected_at"] = datetime.now()
        logger.info(f"Connected to EC2 relay at {self.host}:{self.port}")

    def _receive_loop(self):
        """Receive and decode RTP packets."""
        buffer = b""

        while self._running and self._socket:
            try:
                data = self._socket.recv(4096)
                if not data:
                    raise ConnectionError("Connection closed by server")

                buffer += data

                # Process complete RTP packets from buffer
                while len(buffer) >= 4:
                    # Read packet length (4-byte header)
                    packet_len = struct.unpack("!I", buffer[:4])[0]

                    if packet_len > 65535:
                        logger.warning(f"Invalid packet length: {packet_len}")
                        buffer = buffer[1:]  # Skip byte and retry
                        continue

                    if len(buffer) < 4 + packet_len:
                        break  # Wait for more data

                    # Extract RTP packet
                    rtp_packet = buffer[4:4 + packet_len]
                    buffer = buffer[4 + packet_len:]

                    # Decode RTP
                    audio_data = self._decode_rtp(rtp_packet)
                    if audio_data and self.audio_callback:
                        self.audio_callback(audio_data, self.target_sample_rate)

            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    raise

    def _decode_rtp(self, packet: bytes) -> Optional[bytes]:
        """Decode RTP packet and extract audio payload.

        Args:
            packet: Raw RTP packet

        Returns:
            Audio payload bytes or None
        """
        if len(packet) < 12:
            return None

        # RTP header (12 bytes minimum)
        # byte 0: V(2) P(1) X(1) CC(4)
        # byte 1: M(1) PT(7)
        # bytes 2-3: sequence number
        # bytes 4-7: timestamp
        # bytes 8-11: SSRC

        first_byte = packet[0]
        version = (first_byte >> 6) & 0x03
        padding = (first_byte >> 5) & 0x01
        extension = (first_byte >> 4) & 0x01
        cc = first_byte & 0x0F

        if version != 2:
            return None

        header_len = 12 + (cc * 4)

        # Handle extension header
        if extension and len(packet) > header_len + 4:
            ext_len = struct.unpack("!H", packet[header_len + 2:header_len + 4])[0]
            header_len += 4 + (ext_len * 4)

        if len(packet) <= header_len:
            return None

        payload = packet[header_len:]

        # Handle padding
        if padding and payload:
            pad_len = payload[-1]
            if pad_len < len(payload):
                payload = payload[:-pad_len]

        # Update stats
        self._stats["packets_received"] += 1
        self._stats["bytes_received"] += len(payload)
        self._stats["last_packet_time"] = datetime.now()

        # Log periodically
        if self._stats["packets_received"] % 100 == 1:
            logger.info(
                f"RTP: {self._stats['packets_received']} packets, "
                f"{self._stats['bytes_received'] / 1024:.1f} KB received"
            )

        return payload

    def get_stats(self) -> dict:
        """Get client statistics."""
        return {
            **self._stats,
            "connected": self._socket is not None and self._running,
            "host": self.host,
            "port": self.port,
        }
