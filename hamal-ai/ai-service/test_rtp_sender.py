#!/usr/bin/env python3
"""
Quick script to send raw RTP packets to a server.

Usage:
    # Send to UDP (typical RTP)
    python test_rtp_sender.py --host localhost --port 5004 --protocol udp

    # Send to TCP relay server
    python test_rtp_sender.py --host ec2-server.com --port 5005 --protocol tcp

    # Send audio file
    python test_rtp_sender.py --host localhost --port 5004 --file audio.wav
"""

import socket
import struct
import time
import wave
import argparse
import sys


class RTPSender:
    """Simple RTP packet sender."""

    def __init__(self, host: str, port: int, protocol: str = 'udp'):
        """
        Args:
            host: Server hostname/IP
            port: Server port
            protocol: 'udp' or 'tcp'
        """
        self.host = host
        self.port = port
        self.protocol = protocol.lower()

        # RTP header fields
        self.sequence_number = 0
        self.timestamp = 0
        self.ssrc = 0x12345678  # Synchronization source identifier

        # Socket
        self.socket = None

    def connect(self):
        """Connect to server."""
        if self.protocol == 'tcp':
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            print(f"✓ Connected to TCP {self.host}:{self.port}")
        else:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            print(f"✓ Sending UDP to {self.host}:{self.port}")

    def build_rtp_packet(self, payload: bytes, payload_type: int = 0, marker: bool = False) -> bytes:
        """
        Build RTP packet with header + payload.

        RTP Header Format (12 bytes):
        0                   1                   2                   3
        0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
        +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        |V=2|P|X|  CC   |M|     PT      |       sequence number         |
        +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        |                           timestamp                           |
        +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        |           synchronization source (SSRC) identifier            |
        +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

        Args:
            payload: Audio/video data bytes
            payload_type: RTP payload type (0=PCMU, 8=PCMA, 11=L16, etc.)
            marker: Marker bit (indicates start of frame)

        Returns:
            Complete RTP packet (header + payload)
        """
        # Build first byte: V=2, P=0, X=0, CC=0
        byte0 = 0x80  # Version 2

        # Build second byte: M + PT
        byte1 = (1 if marker else 0) << 7 | (payload_type & 0x7F)

        # Pack RTP header (12 bytes)
        header = struct.pack(
            '!BBHII',
            byte0,                      # V, P, X, CC
            byte1,                      # M, PT
            self.sequence_number,       # Sequence number
            self.timestamp,             # Timestamp
            self.ssrc                   # SSRC
        )

        # Increment sequence number (wraps at 65536)
        self.sequence_number = (self.sequence_number + 1) % 65536

        return header + payload

    def send_packet(self, payload: bytes, payload_type: int = 0, marker: bool = False):
        """Send one RTP packet."""
        packet = self.build_rtp_packet(payload, payload_type, marker)

        if self.protocol == 'tcp':
            self.socket.sendall(packet)
        else:
            self.socket.sendto(packet, (self.host, self.port))

        print(f"→ Sent RTP packet: seq={self.sequence_number-1}, ts={self.timestamp}, size={len(packet)} bytes")

    def send_test_audio(self, duration: float = 5.0, sample_rate: int = 8000, payload_type: int = 0):
        """
        Send test audio (sine wave tone).

        Args:
            duration: Duration in seconds
            sample_rate: Audio sample rate (Hz)
            payload_type: RTP payload type (0=PCMU/G.711u, 8=PCMA/G.711a)
        """
        print(f"\n📻 Sending {duration}s test audio ({sample_rate} Hz, PT={payload_type})...")

        # Generate 440 Hz sine wave
        frequency = 440  # A4 note
        samples_per_packet = 160  # 20ms @ 8kHz
        timestamp_increment = samples_per_packet  # Timestamp units match sample rate

        start_time = time.time()
        packet_count = 0

        try:
            while time.time() - start_time < duration:
                # Generate audio samples (simple sine wave)
                t = packet_count * samples_per_packet / sample_rate
                samples = []
                for i in range(samples_per_packet):
                    t_sample = t + i / sample_rate
                    # Sine wave: -1 to 1
                    value = 0.5 * (2**15 - 1) * (1 + 0.8 * (t_sample % 1.0))  # Simple ramp
                    samples.append(int(value))

                # Convert to bytes (16-bit PCM)
                payload = struct.pack(f'!{len(samples)}h', *samples)

                # Send packet
                marker = (packet_count == 0)  # Mark first packet
                self.send_packet(payload, payload_type, marker)

                # Update timestamp
                self.timestamp += timestamp_increment
                packet_count += 1

                # Pace packets (20ms interval)
                time.sleep(0.02)

        except KeyboardInterrupt:
            print("\n⚠ Interrupted by user")

        print(f"✓ Sent {packet_count} RTP packets")

    def send_wav_file(self, wav_path: str, payload_type: int = 11):
        """
        Send audio from WAV file.

        Args:
            wav_path: Path to WAV file
            payload_type: RTP payload type (11=L16 linear PCM)
        """
        print(f"\n📁 Reading WAV file: {wav_path}...")

        try:
            with wave.open(wav_path, 'rb') as wav:
                channels = wav.getnchannels()
                sample_width = wav.getsampwidth()
                sample_rate = wav.getframerate()
                n_frames = wav.getnframes()

                print(f"   Channels: {channels}")
                print(f"   Sample width: {sample_width} bytes")
                print(f"   Sample rate: {sample_rate} Hz")
                print(f"   Frames: {n_frames}")
                print(f"   Duration: {n_frames/sample_rate:.2f}s")

                # Read all audio data
                audio_data = wav.readframes(n_frames)

                print(f"\n📻 Sending WAV as RTP packets...")

                # Send in chunks
                samples_per_packet = 160  # 20ms @ 8kHz (adjust based on sample rate)
                bytes_per_packet = samples_per_packet * sample_width * channels
                timestamp_increment = samples_per_packet

                packet_count = 0
                offset = 0

                while offset < len(audio_data):
                    chunk = audio_data[offset:offset + bytes_per_packet]
                    if len(chunk) == 0:
                        break

                    # Send packet
                    marker = (packet_count == 0)
                    self.send_packet(chunk, payload_type, marker)

                    # Update
                    self.timestamp += timestamp_increment
                    offset += bytes_per_packet
                    packet_count += 1

                    # Pace packets
                    time.sleep(0.02)

                print(f"✓ Sent {packet_count} RTP packets from WAV file")

        except FileNotFoundError:
            print(f"❌ Error: File not found: {wav_path}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error reading WAV: {e}")
            sys.exit(1)

    def close(self):
        """Close connection."""
        if self.socket:
            self.socket.close()
            print("✓ Connection closed")


def main():
    parser = argparse.ArgumentParser(description='Send raw RTP packets to server')
    parser.add_argument('--host', default='localhost', help='Server hostname/IP (default: localhost)')
    parser.add_argument('--port', type=int, default=5004, help='Server port (default: 5004)')
    parser.add_argument('--protocol', choices=['udp', 'tcp'], default='udp',
                        help='Protocol to use (default: udp)')
    parser.add_argument('--duration', type=float, default=5.0,
                        help='Test audio duration in seconds (default: 5)')
    parser.add_argument('--sample-rate', type=int, default=8000,
                        help='Sample rate in Hz (default: 8000)')
    parser.add_argument('--payload-type', type=int, default=0,
                        help='RTP payload type: 0=PCMU, 8=PCMA, 11=L16 (default: 0)')
    parser.add_argument('--file', help='Send audio from WAV file instead of test tone')

    args = parser.parse_args()

    print("=" * 60)
    print("RTP Packet Sender")
    print("=" * 60)

    # Create sender
    sender = RTPSender(args.host, args.port, args.protocol)

    try:
        # Connect
        sender.connect()

        # Send audio
        if args.file:
            sender.send_wav_file(args.file, args.payload_type)
        else:
            sender.send_test_audio(args.duration, args.sample_rate, args.payload_type)

    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    finally:
        sender.close()

    print("\n✓ Done")


if __name__ == '__main__':
    main()
