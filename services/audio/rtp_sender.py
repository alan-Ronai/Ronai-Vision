"""RTP audio sender for bidirectional communication.

Sends audio packets to field devices via RTP protocol.
"""

import socket
import struct
import time
import numpy as np
import audioop
import logging
from typing import Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RTPSenderConfig:
    """RTP sender configuration."""

    payload_type: int = 0  # 0 for G.711 μ-law
    ssrc: int = 0x12345678  # Synchronization source identifier
    chunk_size: int = 160  # Samples per packet (20ms at 8kHz)
    sample_rate: int = 8000  # G.711 sample rate
    codec: str = "g711_ulaw"  # Audio codec


class RTPAudioSender:
    """Sends audio via RTP to field devices.

    Encodes PCM audio to G.711 (or other codecs) and sends as RTP packets.
    """

    def __init__(
        self,
        destination_host: str,
        destination_port: int,
        config: Optional[RTPSenderConfig] = None,
    ):
        """Initialize RTP audio sender.

        Args:
            destination_host: Destination IP address
            destination_port: Destination UDP port
            config: RTP sender configuration (optional)
        """
        self.destination_host = destination_host
        self.destination_port = destination_port
        self.config = config or RTPSenderConfig()

        # RTP state
        self._sequence_number = 0
        self._timestamp = 0
        self._socket: Optional[socket.socket] = None

        logger.info(f"RTPAudioSender initialized")
        logger.info(f"  Destination: {destination_host}:{destination_port}")
        logger.info(f"  Codec: {self.config.codec}")
        logger.info(f"  Payload type: {self.config.payload_type}")

    def connect(self):
        """Create UDP socket for sending."""
        if self._socket is not None:
            return

        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            logger.info(f"RTP sender socket created")
        except Exception as e:
            logger.error(f"Failed to create socket: {e}")
            raise

    def disconnect(self):
        """Close socket."""
        if self._socket:
            try:
                self._socket.close()
            except:
                pass
            self._socket = None
            logger.info("RTP sender socket closed")

    def send_audio(self, audio: np.ndarray, sample_rate: int = 16000) -> int:
        """Send audio as RTP packets.

        Args:
            audio: Audio samples (int16 numpy array)
            sample_rate: Sample rate of input audio (will be resampled if needed)

        Returns:
            Number of packets sent
        """
        if self._socket is None:
            self.connect()

        # Convert to int16 if needed
        if audio.dtype != np.int16:
            if audio.dtype in (np.float32, np.float64):
                audio = (audio * 32767).astype(np.int16)
            else:
                audio = audio.astype(np.int16)

        # Resample to codec sample rate if needed
        if sample_rate != self.config.sample_rate:
            audio = self._resample(audio, sample_rate, self.config.sample_rate)

        # Encode audio based on codec
        if self.config.codec == "g711_ulaw":
            encoded = self._encode_g711_ulaw(audio)
        elif self.config.codec == "g711_alaw":
            encoded = self._encode_g711_alaw(audio)
        elif self.config.codec == "pcm":
            encoded = audio.tobytes()
        else:
            logger.error(f"Unsupported codec: {self.config.codec}")
            return 0

        # Split into chunks and send as RTP packets
        packet_count = 0
        chunk_bytes = self.config.chunk_size  # Bytes per packet

        for i in range(0, len(encoded), chunk_bytes):
            chunk = encoded[i : i + chunk_bytes]

            if len(chunk) == 0:
                continue

            # Create RTP packet
            rtp_packet = self._create_rtp_packet(
                chunk, self._sequence_number, self._timestamp
            )

            # Send packet
            try:
                self._socket.sendto(
                    rtp_packet, (self.destination_host, self.destination_port)
                )
                packet_count += 1

                # Update state
                self._sequence_number = (self._sequence_number + 1) & 0xFFFF
                self._timestamp += self.config.chunk_size  # Samples per packet

                # Small delay to maintain packet rate
                # 20ms packets at 8kHz = 160 samples
                time.sleep(0.020)  # 20ms

            except Exception as e:
                logger.error(f"Failed to send RTP packet: {e}")

        logger.info(f"Sent {packet_count} RTP packets ({len(audio)} samples)")
        return packet_count

    def _create_rtp_packet(
        self, payload: bytes, sequence_number: int, timestamp: int
    ) -> bytes:
        """Create RTP packet with header.

        Args:
            payload: Audio payload bytes
            sequence_number: RTP sequence number
            timestamp: RTP timestamp

        Returns:
            Complete RTP packet bytes
        """
        # RTP header (RFC 3550)
        # V=2, P=0, X=0, CC=0, M=0, PT=payload_type
        byte0 = 2 << 6  # Version 2
        byte1 = self.config.payload_type & 0x7F

        # Pack header
        header = struct.pack(
            "!BBHII", byte0, byte1, sequence_number, timestamp, self.config.ssrc
        )

        return header + payload

    def _encode_g711_ulaw(self, audio: np.ndarray) -> bytes:
        """Encode PCM to G.711 μ-law.

        Args:
            audio: PCM audio samples (int16)

        Returns:
            Encoded bytes
        """
        pcm_bytes = audio.tobytes()
        return audioop.lin2ulaw(pcm_bytes, 2)

    def _encode_g711_alaw(self, audio: np.ndarray) -> bytes:
        """Encode PCM to G.711 A-law.

        Args:
            audio: PCM audio samples (int16)

        Returns:
            Encoded bytes
        """
        pcm_bytes = audio.tobytes()
        return audioop.lin2alaw(pcm_bytes, 2)

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate.

        Args:
            audio: Audio samples
            orig_sr: Original sample rate
            target_sr: Target sample rate

        Returns:
            Resampled audio
        """
        try:
            import librosa

            # librosa expects float32
            audio_float = audio.astype(np.float32) / 32768.0
            resampled = librosa.resample(
                audio_float, orig_sr=orig_sr, target_sr=target_sr
            )
            return (resampled * 32767).astype(np.int16)
        except ImportError:
            # Fallback: simple linear interpolation
            from scipy import signal

            num_samples = int(len(audio) * target_sr / orig_sr)
            return signal.resample(audio, num_samples).astype(np.int16)

    def reset_state(self):
        """Reset RTP sequence number and timestamp."""
        self._sequence_number = 0
        self._timestamp = 0
        logger.debug("RTP sender state reset")
