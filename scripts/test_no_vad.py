"""Quick test to check if audio contains speech without VAD filtering.

This script tests transcription with VAD disabled to see if the audio
actually contains detectable speech, or if VAD is filtering it out.
"""

import sys
import os
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.audio.transcriber import HebrewTranscriber
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def main():
    audio_file = sys.argv[1] if len(sys.argv) > 1 else "assets/test_audio.wav"

    if not os.path.exists(audio_file):
        print(f"❌ File not found: {audio_file}")
        return 1

    print("=" * 70)
    print("Testing Transcription WITHOUT VAD (Voice Activity Detection)")
    print("=" * 70)
    print(f"\n📁 Audio: {audio_file}\n")

    transcriber = HebrewTranscriber(
        model_path="models/whisper-large-v3-hebrew-ct2",
        enable_preprocessing=True,
        target_audio_level=-12.0,
    )

    print("🎤 Transcribing with VAD DISABLED...")
    print("   This will process all audio, even silence\n")

    import time

    start = time.time()

    # Try with VAD disabled
    try:
        result = transcriber.transcribe_file(
            audio_file,
            vad_filter=False,  # DISABLE VAD
        )

        elapsed = time.time() - start

        print(f"\n✅ Completed in {elapsed:.1f}s")
        print(f"\n📝 Result: {result.get('text', '(empty)')}")
        print(f"🔢 Segments: {len(result.get('segments', []))}")

        if result.get("text"):
            print("\n✓ SUCCESS: Audio contains detectable speech!")
            print("   The issue was VAD filtering being too strict.")
        else:
            print("\n⚠️  No speech detected even without VAD")
            print("   The audio may be:")
            print("   - Too quiet (even after preprocessing)")
            print("   - Not actually containing speech")
            print("   - In a language/format the model doesn't recognize")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
