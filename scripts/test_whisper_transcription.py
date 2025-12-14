"""Test Whisper Hebrew transcription on audio file.

Usage:
    python scripts/test_whisper_transcription.py <audio_file>
    python scripts/test_whisper_transcription.py assets/test_audio.wav

Model Information:
    This script uses the Whisper Large v3 Hebrew model stored locally in
    models/whisper-large-v3-hebrew/ directory.

    The model should be the same as ivrit-ai/whisper-large-v3 from HuggingFace.
    Note: ivrit-ai/whisper-large-v3-ct2 is a CTranslate2-optimized version
    which is faster but requires the faster-whisper library instead of transformers.
"""

import sys
import os
from pathlib import Path
import warnings

# Suppress warnings before importing other modules
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch_dtype.*")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.audio.transcriber import HebrewTranscriber
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def main():
    # Get audio file path from command line or use default
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        # Default to assets directory
        audio_file = "assets/test_audio.wav"

    # Check for --sensitive-vad flag
    sensitive_vad = "--sensitive-vad" in sys.argv or "-s" in sys.argv

    # Check if file exists
    if not os.path.exists(audio_file):
        print(f"❌ Error: Audio file not found: {audio_file}")
        print(f"\nUsage: python {sys.argv[0]} <audio_file>")
        print(f"Example: python {sys.argv[0]} assets/test_audio.wav")
        return 1

    print("=" * 70)
    print("Hebrew Whisper Transcription Test")
    print("=" * 70)
    print(f"\n📁 Audio File: {audio_file}")

    # Get file info
    file_size = os.path.getsize(audio_file)
    print(f"📊 File Size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")

    # Initialize transcriber
    print("\n🔧 Initializing Hebrew Transcriber...")
    print("   (Model will load on first transcription)")
    print("   🎯 Optimized parameters for quality")
    print("   🔊 VAD enabled with default Whisper settings (works best)")

    transcriber = HebrewTranscriber(
        model_path="models/whisper-large-v3-hebrew-ct2",
        device="auto",
        compute_type="int8",  # int8 for fastest CPU inference
        enable_preprocessing=False,  # Disabled - interferes with VAD
        target_audio_level=-12.0,  # Only used if preprocessing enabled
    )

    # Transcribe audio file
    print("\n🎤 Transcribing audio...")
    if sensitive_vad:
        print("   🔊 SENSITIVE VAD MODE: Using lower threshold for quieter speech")
        print("   💡 Good for speech 5-10 dB quieter than average")
    else:
        print("   💡 TIP: Add --sensitive-vad or -s flag to catch quieter speech")
    print("   This may take 30-60 seconds for a 42-second audio file...")
    print("   (First run loads the model, subsequent runs are faster)")
    print("   ⏳ Please wait...")

    import time

    start_time = time.time()

    # Generate preprocessed audio filename
    audio_filename = Path(audio_file).stem
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    preprocessed_output = (
        f"output/transcriptions/{audio_filename}_preprocessed_{timestamp}.wav"
    )

    try:
        print(f"\n💾 Preprocessed audio will be saved to: {preprocessed_output}")

        result = transcriber.transcribe_file(
            audio_file,
            save_preprocessed=preprocessed_output,
            sensitive_vad=True,  # For now setting to true, sensitive_vad,
        )

        end_time = time.time()
        processing_time = end_time - start_time

        print(f"\n✅ Transcription completed in {processing_time:.1f} seconds!")
        print(f"✅ Preprocessed audio saved successfully!")

        print("\n" + "=" * 70)
        print("TRANSCRIPTION RESULTS")
        print("=" * 70)

        # Display the transcribed text
        text = result.get("text", "").strip()
        if text:
            print(f"\n📝 Hebrew Text:\n")
            print(f"   {text}")
            print()
        else:
            print("\n⚠️  No text transcribed (audio may be silent or unclear)")

        # Save transcription to file
        output_dir = Path("output/transcriptions")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate output filename with timestamp to avoid overwriting
        audio_filename = Path(audio_file).stem
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"{audio_filename}_transcript_{timestamp}.txt"

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"Transcription of: {audio_file}\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Duration: {result.get('duration', 0):.2f} seconds\n")
            f.write(f"Processing Time: {processing_time:.2f} seconds\n")
            f.write(f"Language: {result.get('language', 'he')}\n")
            f.write("\n" + "=" * 70 + "\n")
            f.write("TRANSCRIBED TEXT:\n")
            f.write("=" * 70 + "\n\n")
            f.write(text if text else "(No text transcribed)")
            f.write("\n\n")

            # Add segments if available
            if "segments" in result and result["segments"]:
                f.write("\n" + "=" * 70 + "\n")
                f.write("DETAILED SEGMENTS:\n")
                f.write("=" * 70 + "\n\n")
                for i, segment in enumerate(result["segments"], 1):
                    start = segment.get("start", 0)
                    end = segment.get("end", 0)
                    seg_text = segment.get("text", "").strip()
                    f.write(f"[{i}] {start:.1f}s - {end:.1f}s: {seg_text}\n")

        print(f"\n💾 Transcription saved to: {output_file}")

        # Display additional info if available
        if "language" in result:
            print(f"\n🌍 Detected Language: {result['language']}")

        if "duration" in result:
            print(f"⏱️  Audio Duration: {result['duration']:.2f} seconds")

        if "processing_time" in result:
            print(f"⚡ Processing Time: {result['processing_time']:.2f} seconds")
            if result["duration"] > 0:
                ratio = result["processing_time"] / result["duration"]
                print(f"   Speed: {ratio:.2f}x realtime")

        # Check for confidence scores if available
        if "segments" in result and result["segments"]:
            print(f"\n📊 Segments: {len(result['segments'])} detected")
            print("\nDetailed Segments:")
            for i, segment in enumerate(result["segments"], 1):
                start = segment.get("start", 0)
                end = segment.get("end", 0)
                seg_text = segment.get("text", "").strip()
                print(f"   [{i}] {start:.1f}s - {end:.1f}s: {seg_text}")

        print("\n" + "=" * 70)
        print("✅ Transcription Complete!")
        print("=" * 70)

        return 0

    except Exception as e:
        print(f"\n❌ Error during transcription: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Transcription interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
