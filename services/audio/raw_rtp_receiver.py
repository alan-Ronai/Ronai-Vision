"""Raw UDP/RTP receiver without RTSP signaling.

Listens directly on UDP port for RTP packets, bypassing RTSP setup.
Useful for simple devices that send raw RTP streams.
"""

import socket
import struct
import threading
import time
import numpy as np
import logging
from typing import Optional, Callable
from datetime import datetime

from .jitter_buffer import JitterBuffer, RTPPacket
from .audio_decoders import get_decoder, G711Decoder
from .audio_writer import AudioWriter

logger = logging.getLogger(__name__)


class RawRTPReceiver:
    """Raw RTP receiver that listens on UDP port without RTSP.

    Accepts RTP packets directly, detects codec from payload type,
    decodes audio, and optionally saves to file or streams to callback.
    """

    def __init__(
        self,
        listen_host: str = "0.0.0.0",
        listen_port: int = 5004,
        storage_path: str = "audio_storage/recordings",
        jitter_buffer_ms: int = 100,
        auto_detect_codec: bool = True,
        default_codec: str = "g711_ulaw",
        audio_callback: Optional[Callable] = None,
    ):
        """Initialize raw RTP receiver.

        Args:
            listen_host: Host to bind to (default "0.0.0.0")
            listen_port: UDP port to listen on (default 5004)
            storage_path: Path to save audio files
            jitter_buffer_ms: Jitter buffer depth in milliseconds
            auto_detect_codec: Auto-detect codec from RTP payload type
            default_codec: Default codec if auto-detection fails
            audio_callback: Optional callback(audio_chunk, sample_rate) for streaming
        """
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.storage_path = storage_path
        self.jitter_buffer_ms = jitter_buffer_ms
        self.auto_detect_codec = auto_detect_codec
        self.default_codec = default_codec
        self.audio_callback = audio_callback

        # RTP socket
        self._socket: Optional[socket.socket] = None
        self._receive_thread: Optional[threading.Thread] = None
        self._decode_thread: Optional[threading.Thread] = None

        # State
        self._running = False
        self._decoder = None
        self._audio_writer = None
        self._jitter_buffer = None
        self._session_id = None
        self._remote_address = None

        # Statistics
        self._stats = {
            "packets_received": 0,
            "bytes_received": 0,
            "packets_lost": 0,
            "packets_out_of_order": 0,
            "start_time": None,
            "last_packet_time": None,
        }

        logger.info(f"RawRTPReceiver initialized on {listen_host}:{listen_port}")

    def start(self):
        """Start listening for RTP packets."""
        if self._running:
            logger.warning("Raw RTP receiver already running")
            return

        logger.info(
            f"Starting raw RTP receiver on {self.listen_host}:{self.listen_port}"
        )

        try:
            # Create UDP socket
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._socket.bind((self.listen_host, self.listen_port))
            self._socket.settimeout(1.0)

            logger.info(
                f"Raw RTP receiver listening on {self.listen_host}:{self.listen_port}"
            )

        except Exception as e:
            logger.error(f"Failed to bind socket: {e}")
            raise

        # Initialize jitter buffer
        self._jitter_buffer = JitterBuffer(buffer_ms=self.jitter_buffer_ms)

        # Start threads
        self._running = True
        self._stats["start_time"] = datetime.now()

        self._receive_thread = threading.Thread(
            target=self._receive_loop, daemon=True, name="RawRTP-Receive"
        )
        self._receive_thread.start()

        self._decode_thread = threading.Thread(
            target=self._decode_loop, daemon=True, name="RawRTP-Decode"
        )
        self._decode_thread.start()

        logger.info("Raw RTP receiver started")

    def stop(self):
        """Stop receiving RTP packets."""
        if not self._running:
            return

        logger.info("Stopping raw RTP receiver...")
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
        if self._decode_thread:
            self._decode_thread.join(timeout=2.0)

        # Close audio writer
        if self._audio_writer:
            self._audio_writer.close()
            self._audio_writer = None

        logger.info("Raw RTP receiver stopped")
        self._print_stats()

    def get_stats(self) -> dict:
        """Get receiver statistics."""
        return {
            **self._stats,
            "running": self._running,
            "session_id": self._session_id,
            "remote_address": self._remote_address,
            "codec": self._decoder.__class__.__name__ if self._decoder else None,
        }

    def _receive_loop(self):
        """Main receive loop."""
        logger.info("Receive loop started")

        while self._running:
            try:
                # Receive packet
                data, addr = self._socket.recvfrom(2048)

                # Track remote address
                if self._remote_address is None:
                    self._remote_address = addr
                    logger.info(f"Detected RTP source: {addr}")

                # Parse RTP packet
                packet = self._parse_rtp_packet(data)
                if packet is None:
                    continue

                # Auto-detect codec on first packet
                if self._decoder is None and self.auto_detect_codec:
                    self._setup_decoder(packet.payload_type)

                # Initialize session ID if needed
                if self._session_id is None:
                    self._session_id = f"raw_{int(time.time())}"

                # Add to jitter buffer
                self._jitter_buffer.push(packet)

                # Update stats
                self._stats["packets_received"] += 1
                self._stats["bytes_received"] += len(data)
                self._stats["last_packet_time"] = datetime.now()

            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    logger.error(f"Receive error: {e}", exc_info=True)

        logger.info("Receive loop stopped")

    def _decode_loop(self):
        """Decode and process audio."""
        logger.info("Decode loop started")

        while self._running:
            try:
                # Pop packet from jitter buffer
                packet = self._jitter_buffer.pop()
                if packet is None:
                    time.sleep(0.01)
                    continue

                # Ensure decoder is initialized
                if self._decoder is None:
                    logger.warning("No decoder initialized yet")
                    time.sleep(0.1)
                    continue

                # Decode packet
                pcm_samples = self._decoder.decode(packet.payload)

                # Write to file if audio writer is set up
                if self._audio_writer:
                    self._audio_writer.write(pcm_samples)

                # Call streaming callback if provided
                if self.audio_callback:
                    try:
                        self.audio_callback(
                            pcm_samples, self._decoder.get_sample_rate()
                        )
                    except Exception as e:
                        logger.error(f"Audio callback error: {e}")

            except Exception as e:
                if self._running:
                    logger.error(f"Decode error: {e}", exc_info=True)

        logger.info("Decode loop stopped")

    def _setup_decoder(self, payload_type: int):
        """Setup audio decoder based on RTP payload type.

        Args:
            payload_type: RTP payload type number
        """
        # Standard RTP payload types (RFC 3551)
        codec_map = {
            0: ("g711_ulaw", "G.711 μ-law"),
            8: ("g711_alaw", "G.711 A-law"),
            # Add more as needed
        }

        # Handle dynamic payload types (96-127) - assume default codec
        if 96 <= payload_type <= 127:
            codec_name = self.default_codec
            codec_desc = f"Dynamic PT {payload_type} (using {codec_name})"
            logger.warning(
                f"Dynamic payload type {payload_type} detected. "
                f"Assuming {codec_name}. Specify correct codec if different."
            )
        else:
            codec_name, codec_desc = codec_map.get(
                payload_type, (self.default_codec, "Default")
            )

        logger.info(f"Detected RTP payload type {payload_type}: {codec_desc}")

        try:
            self._decoder = get_decoder(codec_name)
            logger.info(f"Initialized {codec_name} decoder")

            # Setup audio writer
            self._setup_audio_writer()

        except Exception as e:
            logger.error(f"Failed to setup decoder: {e}")

    def _setup_audio_writer(self):
        """Setup audio writer for saving to file."""
        if not self._decoder:
            return

        try:
            self._audio_writer = AudioWriter(
                storage_path=self.storage_path,
                session_id=self._session_id,
                codec_name=self._decoder.__class__.__name__.replace(
                    "Decoder", ""
                ).lower(),
                sample_rate=self._decoder.get_sample_rate(),
                channels=self._decoder.get_channels(),
            )
            logger.info(f"Audio writer initialized: {self._audio_writer.wav_path}")

        except Exception as e:
            logger.error(f"Failed to setup audio writer: {e}")

    @staticmethod
    def _parse_rtp_packet(data: bytes) -> Optional[RTPPacket]:
        """Parse RTP packet.

        Args:
            data: Raw UDP packet data

        Returns:
            RTPPacket or None if invalid
        """
        if len(data) < 12:
            return None

        try:
            # Parse RTP header (RFC 3550)
            byte0, byte1, seq, timestamp, ssrc = struct.unpack("!BBHII", data[:12])

            version = (byte0 >> 6) & 0x03
            if version != 2:
                return None

            padding = (byte0 >> 5) & 0x01
            extension = (byte0 >> 4) & 0x01
            csrc_count = byte0 & 0x0F

            marker = (byte1 >> 7) & 0x01
            payload_type = byte1 & 0x7F

            # Skip CSRC identifiers
            header_len = 12 + (csrc_count * 4)

            # Skip extension if present
            if extension and len(data) >= header_len + 4:
                ext_len = struct.unpack("!H", data[header_len + 2 : header_len + 4])[0]
                header_len += 4 + (ext_len * 4)

            # Extract payload
            payload = data[header_len:]

            # Remove padding if present
            if padding and len(payload) > 0:
                padding_len = payload[-1]
                payload = payload[:-padding_len]

            return RTPPacket(
                sequence_number=seq,
                timestamp=timestamp,
                payload_type=payload_type,
                payload=payload,
                received_at=time.time(),
            )

        except Exception as e:
            logger.error(f"Parse error: {e}")
            return None

    def _print_stats(self):
        """Print session statistics."""
        if self._stats["start_time"]:
            duration = (datetime.now() - self._stats["start_time"]).total_seconds()
            logger.info(f"Session statistics:")
            logger.info(f"  Duration: {duration:.1f}s")
            logger.info(f"  Packets received: {self._stats['packets_received']}")
            logger.info(f"  Bytes received: {self._stats['bytes_received']}")
            logger.info(f"  Packets lost: {self._stats['packets_lost']}")
