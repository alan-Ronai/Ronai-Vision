#!/usr/bin/env python3
"""Standalone audio server startup script."""

import sys
import os
import signal
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.audio import RTPAudioServer


def main():
    """Start the audio server."""
    print("=" * 60)
    print("RTP/RTSP Audio Server")
    print("=" * 60)

    # Create server
    server = RTPAudioServer(
        rtsp_host="0.0.0.0",
        rtsp_port=8554,
        rtp_base_port=5004,
        storage_path="audio_storage/recordings",
        session_timeout=60,
        jitter_buffer_ms=100
    )

    # Handle Ctrl+C
    def signal_handler(sig, frame):
        print("\n[Server] Shutting down...")
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Start server
    try:
        server.start()
        print("\n[Server] Running! Press Ctrl+C to stop")
        print("[Server] RTSP URL: rtsp://0.0.0.0:8554/audio")
        print("[Server] RTP Port: 5004")
        print("[Server] Storage: audio_storage/recordings/")
        print()

        # Keep running
        while True:
            time.sleep(1)

    except Exception as e:
        print(f"\n[Server] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
