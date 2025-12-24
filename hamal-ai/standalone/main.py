#!/usr/bin/env python3
"""Hebrew Transcription Service.

Run with: python main.py

Features:
- Enter file paths to transcribe local files
- Automatically connects to RTP/TCP server for live radio
- All results saved to output/transcriptions/
- Same model handles both file and live transcription
"""

import os
import threading
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List

# Load environment variables
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from transcriber import WhisperTranscriber, TranscriptionResult
from rtp_client import RTPTCPClient

# ============== CONFIGURATION ==============
# Edit these or use .env file

WHISPER_MODEL = os.getenv("MODEL", "ivrit-ai/whisper-large-v3-hebrew-ct2")
WHISPER_DEVICE = os.getenv("WDEVICE", "auto")
WHISPER_COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")

RTP_HOST = os.getenv("RTP_HOST", "34.232.67.91")  # Your EC2 relay IP
RTP_PORT = int(os.getenv("RTP_PORT", "5005"))

SAMPLE_RATE = 16000
BUFFER_DURATION = 4.0  # seconds of audio before transcribing
IDLE_TIMEOUT = 2.0  # seconds of silence before transcribing

OUTPUT_DIR = Path("output/transcriptions")

# ============== GLOBAL STATE ==============

transcriber: Optional[WhisperTranscriber] = None
rtp_client: Optional[RTPTCPClient] = None
transcribe_lock = threading.Lock()

# Audio buffer for live transcription
audio_buffer: List[bytes] = []
buffer_lock = threading.Lock()
last_audio_time = 0.0


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def save_transcription(result: TranscriptionResult, source: str) -> Path:
    """Save transcription to file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_source = source.replace("/", "_").replace(":", "_")[:50]
    outfile = OUTPUT_DIR / f"{safe_source}_{timestamp}.txt"

    with open(outfile, "w", encoding="utf-8") as f:
        f.write(f"Source: {source}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Duration: {result.duration_seconds:.2f} seconds\n")
        f.write("\n" + "=" * 70 + "\n")
        f.write("TRANSCRIBED TEXT:\n")
        f.write("=" * 70 + "\n\n")
        f.write(result.text if result.text else "(No speech detected)")
        f.write("\n\n")

        if result.segments:
            f.write("\n" + "=" * 70 + "\n")
            f.write("SEGMENTS:\n")
            f.write("=" * 70 + "\n\n")
            for i, seg in enumerate(result.segments, 1):
                f.write(
                    f"[{i}] {seg['start']:.1f}s - {seg['end']:.1f}s: {seg['text']}\n"
                )

    return outfile


def print_result(result: TranscriptionResult, source: str, outfile: Path):
    """Print transcription result."""
    print("\n" + "=" * 70)
    print(f"📝 TRANSCRIPTION [{source}] ({result.duration_seconds:.1f}s)")
    print("=" * 70)
    print(result.text if result.text else "(No speech detected)")
    print(f"\n💾 Saved to: {outfile}")
    print("=" * 70 + "\n")


# ============== LIVE RTP TRANSCRIPTION ==============


def on_audio_received(audio_data: bytes, sample_rate: int):
    """Callback when audio is received from RTP client."""
    global last_audio_time

    with buffer_lock:
        audio_buffer.append(audio_data)
        last_audio_time = time.time()

    # Check if buffer is full
    total_samples = sum(len(chunk) // 2 for chunk in audio_buffer)
    buffer_duration = total_samples / SAMPLE_RATE

    if buffer_duration >= BUFFER_DURATION:
        threading.Thread(
            target=process_live_buffer, args=("buffer full",), daemon=True
        ).start()


def process_live_buffer(reason: str = ""):
    """Process accumulated live audio buffer."""
    global audio_buffer

    with buffer_lock:
        if not audio_buffer:
            return
        audio_bytes = b"".join(audio_buffer)
        audio_buffer = []

    duration = len(audio_bytes) / (2 * SAMPLE_RATE)
    if duration < 0.5:
        return

    logging.info(f"[LIVE] Processing {duration:.1f}s audio ({reason})")

    with transcribe_lock:
        result = transcriber.transcribe_audio(audio_bytes)

    if result and result.text:
        outfile = save_transcription(result, "live_radio")
        print_result(result, "LIVE RADIO", outfile)
    elif result:
        logging.info(f"[LIVE] No speech detected in {duration:.1f}s audio")


def idle_checker_loop():
    """Background thread to check for idle buffer."""
    global last_audio_time

    while True:
        time.sleep(0.5)

        with buffer_lock:
            if not audio_buffer:
                continue
            total_samples = sum(len(chunk) // 2 for chunk in audio_buffer)

        buffer_duration = total_samples / SAMPLE_RATE
        current_time = time.time()

        if last_audio_time > 0:
            idle_duration = current_time - last_audio_time
            if idle_duration >= IDLE_TIMEOUT and buffer_duration >= 0.5:
                process_live_buffer(f"idle {idle_duration:.1f}s")


# ============== FILE TRANSCRIPTION ==============


def transcribe_file(filepath: str) -> Optional[TranscriptionResult]:
    """Transcribe a local file."""
    path = Path(filepath)

    if not path.exists():
        logging.error(f"File not found: {path}")
        return None

    logging.info(f"[FILE] Transcribing: {path}")

    with transcribe_lock:
        result = transcriber.transcribe_file(str(path))

    if result:
        outfile = save_transcription(result, f"file_{path.name}")
        print_result(result, f"FILE: {path.name}", outfile)

    return result


# ============== MAIN ==============


def main():
    global transcriber, rtp_client

    setup_logging()

    print("\n" + "=" * 70)
    print("🎙️  HEBREW TRANSCRIPTION SERVICE")
    print("=" * 70)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {OUTPUT_DIR.absolute()}")

    # Initialize transcriber
    print(f"\n🔄 Loading model: {WHISPER_MODEL}")
    print(f"   Device: {WHISPER_DEVICE}, Compute: {WHISPER_COMPUTE_TYPE}")

    transcriber = WhisperTranscriber(
        model_path=WHISPER_MODEL,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE,
        sample_rate=SAMPLE_RATE,
    )
    transcriber.load_model()
    print("✅ Model loaded!\n")

    # Start RTP client for live radio
    if RTP_HOST:
        print(f"📡 Connecting to RTP relay: {RTP_HOST}:{RTP_PORT}")
        rtp_client = RTPTCPClient(
            host=RTP_HOST,
            port=RTP_PORT,
            target_sample_rate=SAMPLE_RATE,
            audio_callback=on_audio_received,
        )
        rtp_client.start()

        # Start idle checker
        threading.Thread(target=idle_checker_loop, daemon=True).start()
        print("✅ Live radio transcription active!\n")
    else:
        print("⚠️  RTP_HOST not set - live radio disabled\n")

    # Interactive mode
    print("=" * 70)
    print("Enter audio file paths to transcribe.")
    print("Live radio transcriptions will appear automatically.")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 70 + "\n")

    while True:
        try:
            user_input = input("📂 File path > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit", "q"}:
            break

        if user_input.lower() == "stats":
            if rtp_client:
                stats = rtp_client.get_stats()
                print(
                    f"📊 RTP Stats: {stats['packets_received']} packets, {stats['bytes_received'] / 1024:.1f} KB"
                )
            continue

        transcribe_file(user_input)

    # Cleanup
    print("\n🛑 Shutting down...")
    if rtp_client:
        rtp_client.stop()

    print("👋 Goodbye!")


if __name__ == "__main__":
    main()
