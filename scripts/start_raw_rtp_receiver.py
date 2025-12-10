#!/usr/bin/env python3
"""Standalone RTP receiver (no RTSP handshake required).

Accepts raw RTP packets on UDP port 5004 without any session setup.
Useful for simple streaming or military equipment that doesn't use RTSP.
"""

import sys
import os
import signal
import socket
import struct
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.audio.jitter_buffer import JitterBuffer, RTPPacket
from services.audio.audio_decoders import G711Decoder
from services.audio.audio_writer import AudioWriter


class RawRTPReceiver:
    """Simple RTP receiver without RTSP."""

    def __init__(self, port=5004, storage_path="audio_storage/recordings"):
        self.port = port
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.socket = None
        self.running = False
        self.jitter_buffer = JitterBuffer(buffer_ms=100)
        self.decoder = G711Decoder(law='ulaw')  # G.711 μ-law
        self.writer = None

    def start(self):
        """Start receiving RTP packets."""
        # Create UDP socket
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(('0.0.0.0', self.port))
        self.socket.settimeout(1.0)  # 1 second timeout for checking stop

        print(f"[RTP] Listening on UDP port {self.port}")
        print(f"[RTP] Waiting for packets...")
        print(f"[RTP] Storage: {self.storage_path}")
        print()

        self.running = True
        packet_count = 0
        session_id = f"raw_rtp_{int(time.time())}"

        # Create audio writer
        self.writer = AudioWriter(
            output_dir=str(self.storage_path),
            session_id=session_id,
            codec="g711_ulaw",
            sample_rate=8000,
            channels=1
        )
        self.writer.open()

        try:
            while self.running:
                try:
                    data, addr = self.socket.recvfrom(2048)

                    # First packet - log sender
                    if packet_count == 0:
                        print(f"[RTP] Receiving from {addr[0]}:{addr[1]}")

                    # Parse RTP header
                    packet = self._parse_rtp_packet(data)
                    if packet:
                        # Add to jitter buffer
                        self.jitter_buffer.insert(packet)
                        packet_count += 1

                        # Progress indicator
                        if packet_count % 50 == 0:
                            stats = self.jitter_buffer.get_stats()
                            print(f"[RTP] Received {packet_count} packets "
                                  f"(dropped: {stats['packets_dropped']}, "
                                  f"reordered: {stats['packets_reordered']})")

                        # Process buffered packets
                        while True:
                            buffered_packet = self.jitter_buffer.pop()
                            if not buffered_packet:
                                break

                            # Decode audio
                            pcm_samples = self.decoder.decode(buffered_packet.payload)

                            # Write to file
                            self.writer.write(pcm_samples)

                except socket.timeout:
                    # Timeout - check if we should stop
                    continue
                except Exception as e:
                    print(f"[RTP] Error: {e}")
                    import traceback
                    traceback.print_exc()

        finally:
            print(f"\n[RTP] Received total {packet_count} packets")

            # Close writer
            if self.writer:
                self.writer.close()
                print(f"[RTP] Saved: {self.writer.filepath}")

            # Close socket
            if self.socket:
                self.socket.close()

    def stop(self):
        """Stop receiving."""
        self.running = False

    def _parse_rtp_packet(self, data):
        """Parse RTP packet header."""
        if len(data) < 12:
            return None

        # Parse header (12 bytes minimum)
        header = struct.unpack('!BBHII', data[:12])

        byte0 = header[0]
        version = (byte0 >> 6) & 0x03
        padding = (byte0 >> 5) & 0x01
        extension = (byte0 >> 4) & 0x01
        csrc_count = byte0 & 0x0F

        byte1 = header[1]
        marker = (byte1 >> 7) & 0x01
        payload_type = byte1 & 0x7F

        sequence_number = header[2]
        timestamp = header[3]
        ssrc = header[4]

        # Validate version
        if version != 2:
            return None

        # Calculate header length
        header_length = 12 + (csrc_count * 4)
        if extension:
            if len(data) < header_length + 4:
                return None
            ext_header = struct.unpack('!HH', data[header_length:header_length + 4])
            ext_length = ext_header[1] * 4
            header_length += 4 + ext_length

        # Extract payload
        payload = data[header_length:]

        # Remove padding if present
        if padding and len(payload) > 0:
            padding_length = payload[-1]
            payload = payload[:-padding_length]

        return RTPPacket(
            sequence_number=sequence_number,
            timestamp=timestamp,
            payload_type=payload_type,
            payload=payload,
            received_at=time.time()
        )


def main():
    """Start raw RTP receiver."""
    print("=" * 60)
    print("Raw RTP Receiver (No RTSP)")
    print("=" * 60)
    print()

    receiver = RawRTPReceiver(port=5004)

    # Handle Ctrl+C
    def signal_handler(sig, frame):
        print("\n[RTP] Shutting down...")
        receiver.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        receiver.start()
    except Exception as e:
        print(f"\n[RTP] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
