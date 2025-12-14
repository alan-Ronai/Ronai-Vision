#!/usr/bin/env python3
"""Debug RTP receiver that logs ALL incoming UDP packets.

This will help diagnose why third-party RTP packets aren't being received.
Logs raw packet data, parsing attempts, and any errors.
"""

import sys
import os
import signal
import socket
import struct
import time
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DebugRTPReceiver:
    """RTP receiver with extensive debugging."""

    def __init__(self, port=5004):
        self.port = port
        self.socket = None
        self.running = False

    def start(self):
        """Start receiving and logging ALL UDP packets."""
        # Create UDP socket
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Try to set buffer size larger (in case of packet loss)
        try:
            self.socket.setsockopt(
                socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024
            )  # 1MB
            buffer_size = self.socket.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
            print(f"[DEBUG] Socket receive buffer: {buffer_size} bytes")
        except Exception as e:
            print(f"[DEBUG] Could not set buffer size: {e}")

        self.socket.bind(("0.0.0.0", self.port))
        self.socket.settimeout(1.0)

        print("=" * 70)
        print("DEBUG RTP RECEIVER - ALL PACKETS WILL BE LOGGED")
        print("=" * 70)
        print(f"[DEBUG] Listening on UDP 0.0.0.0:{self.port}")
        print(f"[DEBUG] Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[DEBUG] Waiting for packets...")
        print("=" * 70)
        print()

        self.running = True
        packet_count = 0
        last_ssrc = None
        last_sender = None

        try:
            while self.running:
                try:
                    data, addr = self.socket.recvfrom(4096)
                    packet_count += 1

                    # New sender detected
                    if addr != last_sender:
                        print("\n" + "=" * 70)
                        print(
                            f"[PACKET #{packet_count}] NEW SENDER: {addr[0]}:{addr[1]}"
                        )
                        print("=" * 70)
                        last_sender = addr

                    # Log raw packet info
                    print(
                        f"\n[PACKET #{packet_count}] From {addr[0]}:{addr[1]} | Size: {len(data)} bytes"
                    )
                    print(f"[RAW HEX] {data[: min(32, len(data))].hex()}")

                    # Try to parse as RTP
                    result = self._parse_and_log_rtp(data)

                    if result:
                        ssrc, pt, seq, ts = result
                        if ssrc != last_ssrc:
                            print(f"[INFO] New SSRC detected: 0x{ssrc:08x}")
                            last_ssrc = ssrc

                        # Log every 10th packet summary
                        if packet_count % 10 == 0:
                            print(
                                f"\n[SUMMARY] Received {packet_count} packets so far..."
                            )

                except socket.timeout:
                    # Timeout - check if we should stop
                    continue
                except Exception as e:
                    print(f"[ERROR] Exception while receiving: {e}")
                    import traceback

                    traceback.print_exc()

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user")
        finally:
            print(f"\n{'=' * 70}")
            print(f"[SUMMARY] Total packets received: {packet_count}")
            print(f"[INFO] Stopped at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 70)

            if self.socket:
                self.socket.close()

    def stop(self):
        """Stop receiving."""
        self.running = False

    def _parse_and_log_rtp(self, data):
        """Parse RTP packet and log details."""
        try:
            # Check minimum size
            if len(data) < 12:
                print(f"[WARN] Packet too small for RTP (< 12 bytes)")
                return None

            # Parse header bytes
            byte0, byte1 = data[0], data[1]

            # Extract fields
            version = (byte0 >> 6) & 0x03
            padding = (byte0 >> 5) & 0x01
            extension = (byte0 >> 4) & 0x01
            csrc_count = byte0 & 0x0F
            marker = (byte1 >> 7) & 0x01
            payload_type = byte1 & 0x7F

            # Parse rest of header
            header = struct.unpack("!BBHII", data[:12])
            sequence = header[2]
            timestamp = header[3]
            ssrc = header[4]

            # Calculate payload offset
            header_len = 12 + (csrc_count * 4)

            ext_profile = None
            ext_length_words = 0
            if extension:
                if len(data) >= header_len + 4:
                    ext_header = struct.unpack("!HH", data[header_len : header_len + 4])
                    ext_profile = ext_header[0]
                    ext_length_words = ext_header[1]
                    ext_len = ext_length_words * 4
                    header_len += 4 + ext_len
                    print(
                        f"[EXT] Profile=0x{ext_profile:04x} Length={ext_length_words} words ({ext_len} bytes)"
                    )

            payload_size = len(data) - header_len
            if padding and payload_size > 0:
                padding_len = data[-1]
                payload_size -= padding_len

            # Log parsed fields
            print(f"[RTP] Version={version} PT={payload_type} Marker={marker}")
            print(f"[RTP] Seq={sequence} Timestamp={timestamp} SSRC=0x{ssrc:08x}")
            print(
                f"[RTP] CSRC_count={csrc_count} Extension={extension} Padding={padding}"
            )
            print(f"[RTP] Header={header_len}B Payload={payload_size}B")

            # Validate version
            if version != 2:
                print(f"[ERROR] Invalid RTP version: {version} (expected 2)")
                return None

            # Decode codec from payload type
            codec_name = self._get_codec_name(payload_type)
            print(f"[CODEC] Payload Type {payload_type} = {codec_name}")

            return (ssrc, payload_type, sequence, timestamp)

        except Exception as e:
            print(f"[ERROR] Failed to parse RTP packet: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _get_codec_name(self, pt):
        """Get codec name from payload type."""
        # RFC 3551 standard payload types
        codecs = {
            0: "PCMU (G.711 μ-law)",
            3: "GSM",
            4: "G723",
            5: "DVI4 (8kHz)",
            6: "DVI4 (16kHz)",
            7: "LPC",
            8: "PCMA (G.711 A-law)",
            9: "G722",
            10: "L16 Stereo",
            11: "L16 Mono",
            12: "QCELP",
            13: "CN (Comfort Noise)",
            14: "MPA (MPEG Audio)",
            15: "G728",
            16: "DVI4 (11kHz)",
            17: "DVI4 (22kHz)",
            18: "G729",
            96: "Dynamic (Opus/etc)",
            97: "Dynamic",
            98: "Dynamic",
            99: "Dynamic",
        }

        if pt in codecs:
            return codecs[pt]
        elif 96 <= pt <= 127:
            return f"Dynamic (PT {pt})"
        else:
            return f"Unknown (PT {pt})"


def main():
    """Start debug receiver."""
    import argparse

    parser = argparse.ArgumentParser(description="Debug RTP receiver")
    parser.add_argument("--port", type=int, default=5004, help="UDP port to listen on")
    args = parser.parse_args()

    receiver = DebugRTPReceiver(port=args.port)

    # Handle Ctrl+C
    def signal_handler(sig, frame):
        print("\n[INFO] Shutting down...")
        receiver.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        receiver.start()
    except Exception as e:
        print(f"\n[ERROR] Fatal error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
