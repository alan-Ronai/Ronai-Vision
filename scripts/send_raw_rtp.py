#!/usr/bin/env python3
"""Send raw RTP packets without RTSP handshake.

Simple client that blasts RTP packets directly to a UDP port.
No session setup required.
"""

import socket
import struct
import time
import argparse
import audioop
import numpy as np


def generate_tone(frequency=440, sample_rate=8000, duration=0.02):
    """Generate sine wave tone."""
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples, False)
    tone = np.sin(frequency * 2 * np.pi * t)
    pcm = (tone * 32767).astype(np.int16)
    return pcm


def encode_g711_ulaw(pcm_samples):
    """Encode PCM to G.711 μ-law."""
    pcm_bytes = pcm_samples.tobytes()
    return audioop.lin2ulaw(pcm_bytes, 2)


def create_rtp_packet(payload, sequence_number, timestamp, payload_type=0, ssrc=0x12345678):
    """Create RTP packet."""
    version = 2
    padding = 0
    extension = 0
    csrc_count = 0
    marker = 0

    byte0 = (version << 6) | (padding << 5) | (extension << 4) | csrc_count
    byte1 = (marker << 7) | payload_type

    header = struct.pack('!BBHII',
                         byte0,
                         byte1,
                         sequence_number & 0xFFFF,
                         timestamp & 0xFFFFFFFF,
                         ssrc)

    return header + payload


def send_raw_rtp(server_host="127.0.0.1", server_port=5004, duration=10):
    """Send raw RTP packets."""
    print(f"[Client] Sending raw RTP to {server_host}:{server_port}")
    print(f"[Client] Duration: {duration} seconds")
    print(f"[Client] Codec: G.711 μ-law")
    print()

    # Create UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Audio parameters
    sample_rate = 8000
    packet_duration = 0.02  # 20ms packets
    samples_per_packet = int(sample_rate * packet_duration)

    # Timing
    start_time = time.time()
    sequence_number = 0
    timestamp = 0
    packet_count = 0

    print("[Client] Sending 440 Hz test tone...")

    try:
        while time.time() - start_time < duration:
            # Generate tone
            pcm_samples = generate_tone(frequency=440, sample_rate=sample_rate, duration=packet_duration)

            # Encode to G.711 μ-law
            g711_payload = encode_g711_ulaw(pcm_samples)

            # Create RTP packet
            rtp_packet = create_rtp_packet(
                payload=g711_payload,
                sequence_number=sequence_number,
                timestamp=timestamp,
                payload_type=0  # G.711 μ-law
            )

            # Send packet
            sock.sendto(rtp_packet, (server_host, server_port))

            # Update counters
            sequence_number = (sequence_number + 1) & 0xFFFF
            timestamp += samples_per_packet
            packet_count += 1

            # Progress indicator
            if packet_count % 50 == 0:
                elapsed = time.time() - start_time
                print(f"[Client] Sent {packet_count} packets ({elapsed:.1f}s elapsed)")

            # Wait for next packet
            time.sleep(packet_duration)

        print(f"\n[Client] Sent total {packet_count} RTP packets")

    except KeyboardInterrupt:
        print("\n[Client] Interrupted by user")
    finally:
        sock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send raw RTP packets")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=5004, help="RTP port")
    parser.add_argument("--duration", type=int, default=10, help="Duration in seconds")

    args = parser.parse_args()

    print("=" * 60)
    print("Raw RTP Client (No RTSP)")
    print("=" * 60)
    print()

    send_raw_rtp(
        server_host=args.host,
        server_port=args.port,
        duration=args.duration
    )