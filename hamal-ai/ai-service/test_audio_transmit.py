#!/usr/bin/env python3
"""
Test script for RTP audio transmission to EC2.

Commands via TCP on port 12345, audio via UDP on port 12347:
1. Connect to TCP port 12345
2. Send /ptt command to start transmission
3. Send RTP audio packets via UDP to port 12347
4. Send /pts command to stop transmission
5. Close TCP connection

Usage:
    python test_audio_transmit.py [--audio-file path/to/file.wav]

If no audio file is provided, generates a test tone.
"""

import os
import sys
import time
import argparse
import socket
import struct
import numpy as np

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

load_dotenv("dev.env")

# Configuration from environment
EC2_HOST = os.getenv("EC2_RTP_HOST", "34.232.67.91")
EC2_CMD_PORT = 12345   # TCP port for PTT commands
EC2_AUDIO_PORT = 12347  # UDP port for RTP audio

SAMPLE_RATE = 16000


class CommandConnection:
    """Manages persistent TCP connection for PTT commands."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.sock = None

    def connect(self) -> bool:
        """Establish TCP connection."""
        try:
            print(f"🔌 Connecting to {self.host}:{self.port} (TCP)...")
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(10.0)
            self.sock.connect((self.host, self.port))
            print(f"✅ Connected to {self.host}:{self.port}")
            return True
        except socket.timeout:
            print("❌ Connection timed out")
            return False
        except ConnectionRefusedError:
            print(f"❌ Connection refused to {self.host}:{self.port}")
            return False
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False

    def send_command(self, command: str) -> bool:
        """Send a command over the connection."""
        if not self.sock:
            print("❌ Not connected")
            return False

        try:
            print(f"📡 Sending command: {command}")
            self.sock.sendall(f"{command}\n".encode('utf-8'))
            print(f"✅ Command '{command}' sent")
            return True
        except Exception as e:
            print(f"❌ Send error: {e}")
            return False

    def close(self):
        """Close the connection."""
        if self.sock:
            try:
                self.sock.close()
                print("🔌 TCP connection closed")
            except Exception:
                pass
            self.sock = None


def generate_test_tone(duration_sec: float = 3.0, frequency: float = 440.0) -> bytes:
    """Generate a test tone (sine wave)."""
    print(f"🎵 Generating {duration_sec}s test tone at {frequency}Hz...")

    num_samples = int(SAMPLE_RATE * duration_sec)
    t = np.linspace(0, duration_sec, num_samples, dtype=np.float32)

    # Generate sine wave
    samples = np.sin(2 * np.pi * frequency * t)

    # Apply fade in/out to avoid clicks
    fade_samples = int(SAMPLE_RATE * 0.05)  # 50ms fade
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    samples[:fade_samples] *= fade_in
    samples[-fade_samples:] *= fade_out

    # Convert to int16
    samples = (samples * 32767).astype(np.int16)

    return samples.tobytes()


def load_audio_file(file_path: str) -> tuple[bytes, int]:
    """Load audio from file and return PCM data and sample rate."""
    import wave

    print(f"📁 Loading audio file: {file_path}")

    if file_path.lower().endswith(".wav"):
        with wave.open(file_path, "rb") as wf:
            sample_rate = wf.getframerate()
            num_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())

            print(f"   Sample rate: {sample_rate}Hz")
            print(f"   Channels: {num_channels}")
            print(f"   Sample width: {sample_width} bytes")
            print(
                f"   Duration: {len(frames) / (sample_rate * num_channels * sample_width):.2f}s"
            )

            # Convert stereo to mono if needed
            if num_channels == 2:
                samples = np.frombuffer(frames, dtype=np.int16)
                samples = samples.reshape(-1, 2).mean(axis=1).astype(np.int16)
                frames = samples.tobytes()
                print("   Converted stereo to mono")

            return frames, sample_rate
    else:
        raise ValueError(f"Unsupported audio format: {file_path}")


def resample_audio(audio_data: bytes, from_rate: int, to_rate: int) -> bytes:
    """Resample audio to target sample rate."""
    if from_rate == to_rate:
        return audio_data

    print(f"🔄 Resampling from {from_rate}Hz to {to_rate}Hz...")

    samples = np.frombuffer(audio_data, dtype=np.int16)
    ratio = to_rate / from_rate
    new_num_samples = int(len(samples) * ratio)

    old_indices = np.arange(len(samples))
    new_indices = np.linspace(0, len(samples) - 1, new_num_samples)
    resampled = np.interp(new_indices, old_indices, samples).astype(np.int16)

    return resampled.tobytes()


def create_rtp_packet(
    audio_data: bytes,
    sequence: int,
    timestamp: int,
    sample_rate: int = 16000,
    ssrc: int = 0x12345678,
) -> bytes:
    """Create an RTP packet with audio payload.

    RTP header (12 bytes):
    - Byte 0: V=2, P=0, X=0, CC=0 -> 0x80
    - Byte 1: M=0, PT=payload_type (4=8kHz, 5=16kHz)
    - Bytes 2-3: Sequence number
    - Bytes 4-7: Timestamp
    - Bytes 8-11: SSRC
    """
    # Payload type: 4 for 8000Hz, 5 for 16000Hz
    if sample_rate == 8000:
        payload_type = 4
    else:
        payload_type = 5  # 16kHz

    header = struct.pack(
        ">BBHII",
        0x80,  # V=2, P=0, X=0, CC=0
        payload_type,  # M=0, PT=4 or 5
        sequence & 0xFFFF,
        timestamp & 0xFFFFFFFF,
        ssrc,
    )

    return header + audio_data


def send_audio_rtp(
    host: str, port: int, audio_data: bytes, sample_rate: int = 16000
) -> bool:
    """Send audio as RTP packets over UDP."""

    # Split audio into chunks (20ms at given sample rate)
    # 20ms at 16kHz = 320 samples = 640 bytes
    # 20ms at 8kHz = 160 samples = 320 bytes
    samples_per_chunk = int(sample_rate * 0.02)  # 20ms
    chunk_size = samples_per_chunk * 2  # 16-bit samples = 2 bytes each
    num_chunks = (len(audio_data) + chunk_size - 1) // chunk_size

    print(
        f"📤 Sending {len(audio_data)} bytes as {num_chunks} RTP packets to {host}:{port} (UDP)"
    )
    print(
        f"   Sample rate: {sample_rate}Hz, Payload type: {4 if sample_rate == 8000 else 5}"
    )

    try:
        # Create UDP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        sequence = 0
        timestamp = 0
        packets_sent = 0
        bytes_sent = 0

        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i : i + chunk_size]

            # Create RTP packet with correct payload type for sample rate
            rtp_packet = create_rtp_packet(chunk, sequence, timestamp, sample_rate)

            # Send RTP packet over UDP
            sock.sendto(rtp_packet, (host, port))

            packets_sent += 1
            bytes_sent += len(rtp_packet)
            sequence += 1
            timestamp += len(chunk) // 2  # samples per chunk

            # Small delay to simulate real-time streaming (~20ms per packet)
            time.sleep(0.018)

            # Progress indicator
            if packets_sent % 50 == 0:
                print(f"   Sent {packets_sent}/{num_chunks} packets...")

        sock.close()
        print(f"✅ Successfully sent {packets_sent} packets ({bytes_sent} bytes)")
        return True

    except Exception as e:
        print(f"❌ Error sending audio: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test RTP audio transmission to EC2")
    parser.add_argument("--audio-file", "-f", help="Path to audio file (WAV)")
    parser.add_argument(
        "--duration", "-d", type=float, default=3.0, help="Test tone duration (seconds)"
    )
    parser.add_argument(
        "--frequency", type=float, default=440.0, help="Test tone frequency (Hz)"
    )
    parser.add_argument(
        "--no-ptt", action="store_true", help="Skip PTT signals (send audio only)"
    )
    parser.add_argument(
        "--ptt-delay", type=float, default=0.5, help="Delay after PTT START (seconds)"
    )
    parser.add_argument(
        "--host", default=EC2_HOST, help=f"EC2 host (default: {EC2_HOST})"
    )
    parser.add_argument(
        "--cmd-port", type=int, default=EC2_CMD_PORT, help=f"TCP port for commands (default: {EC2_CMD_PORT})"
    )
    parser.add_argument(
        "--audio-port", type=int, default=EC2_AUDIO_PORT, help=f"UDP port for audio (default: {EC2_AUDIO_PORT})"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("RTP Audio Transmission Test")
    print("=" * 60)
    print(f"EC2 Host:       {args.host}")
    print(f"Command Port:   {args.cmd_port} (TCP)")
    print(f"Audio Port:     {args.audio_port} (UDP)")
    print(f"Sample Rate:    {SAMPLE_RATE}Hz")
    print("Payload Type:   5 (16kHz)")
    print("=" * 60)

    # Load or generate audio
    if args.audio_file:
        audio_data, file_sample_rate = load_audio_file(args.audio_file)
        if file_sample_rate != SAMPLE_RATE:
            audio_data = resample_audio(audio_data, file_sample_rate, SAMPLE_RATE)
    else:
        audio_data = generate_test_tone(args.duration, args.frequency)

    print(
        f"📊 Audio ready: {len(audio_data)} bytes ({len(audio_data) / 2 / SAMPLE_RATE:.2f}s)"
    )
    print()

    # Create TCP connection for commands
    cmd_conn = None
    if not args.no_ptt:
        cmd_conn = CommandConnection(args.host, args.cmd_port)
        if not cmd_conn.connect():
            print("❌ Failed to connect to command server, aborting")
            return 1

    success = False
    try:
        # Step 1: Send PTT START (/ptt) via TCP
        if cmd_conn:
            if not cmd_conn.send_command("/ptt"):
                print("❌ Failed to send PTT START, aborting")
                return 1

            # Wait for radio to key up
            print(f"⏳ Waiting {args.ptt_delay}s for radio to key up...")
            time.sleep(args.ptt_delay)

        # Step 2: Send audio via UDP
        print()
        success = send_audio_rtp(args.host, args.audio_port, audio_data, SAMPLE_RATE)

        # Step 3: Send PTT STOP (/pts) via TCP
        print()
        if cmd_conn:
            time.sleep(0.2)  # Small delay before releasing PTT
            cmd_conn.send_command("/pts")

    finally:
        # Close TCP connection
        if cmd_conn:
            cmd_conn.close()

    print()
    print("=" * 60)
    if success:
        print("✅ TEST PASSED - Audio transmitted successfully")
    else:
        print("❌ TEST FAILED - Audio transmission failed")
    print("=" * 60)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
