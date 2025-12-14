"""Preprocess audio file with volume normalization.

This script demonstrates the audio preprocessing capabilities for improving
transcription quality. It loads an audio file, applies volume normalization,
and saves both the original and preprocessed versions for comparison.

Usage:
    python scripts/preprocess_audio.py <input_audio> [output_audio]
    python scripts/preprocess_audio.py assets/test_audio.wav output/preprocessed.wav
"""

import sys
import os
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.audio import AudioPreprocessor
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    if len(sys.argv) < 2:
        print("❌ Error: No input audio file specified")
        print(f"\nUsage: python {sys.argv[0]} <input_audio> [output_audio]")
        print(
            f"Example: python {sys.argv[0]} assets/test_audio.wav output/preprocessed.wav"
        )
        return 1

    input_file = sys.argv[1]

    # Generate output filename if not specified
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        input_path = Path(input_file)
        output_file = f"output/{input_path.stem}_preprocessed{input_path.suffix}"

    # Check if input exists
    if not os.path.exists(input_file):
        print(f"❌ Error: Input file not found: {input_file}")
        return 1

    print("=" * 70)
    print("Audio Preprocessing Tool")
    print("=" * 70)
    print(f"\n📁 Input:  {input_file}")
    print(f"💾 Output: {output_file}")

    try:
        import soundfile as sf

        # Load audio
        print("\n📥 Loading audio...")
        audio, sample_rate = sf.read(input_file, dtype="float32")
        duration = len(audio) / sample_rate

        print(f"   Sample rate: {sample_rate} Hz")
        print(f"   Duration: {duration:.2f} seconds")
        print(f"   Channels: {'Stereo' if audio.ndim > 1 else 'Mono'}")

        # Convert to mono if needed
        if audio.ndim > 1:
            print("   Converting stereo to mono...")
            audio = audio.mean(axis=1)

        # Initialize preprocessor
        print("\n🔧 Initializing preprocessor...")
        preprocessor = AudioPreprocessor(
            target_rms=-10.0,  # Target loudness (much louder for quiet audio)
            normalization_mode="rms",  # RMS normalization
            apply_noise_gate=False,  # Let VAD handle silence
            remove_dc_offset=True,  # Remove DC bias
            safety_headroom=0.98,  # Allow higher levels (98% of max)
        )

        # Get original statistics
        print("\n📊 Original Audio Statistics:")
        original_stats = preprocessor.get_audio_stats(audio)
        print(f"   RMS Level: {original_stats['rms_db']:.1f} dB")
        print(f"   Peak Level: {original_stats['peak_db']:.1f} dB")
        print(f"   Peak Value: {original_stats['peak_linear']:.3f}")
        if original_stats["is_clipping"]:
            print(
                f"   ⚠️  Clipping detected: {original_stats['clipping_percentage']:.2f}%"
            )
        else:
            print(f"   ✓ No clipping")

        # Process audio
        print("\n⚙️  Processing audio...")
        processed_audio = preprocessor.process(audio, sample_rate)

        # Get processed statistics
        print("\n📊 Preprocessed Audio Statistics:")
        processed_stats = preprocessor.get_audio_stats(processed_audio)
        print(f"   RMS Level: {processed_stats['rms_db']:.1f} dB")
        print(f"   Peak Level: {processed_stats['peak_db']:.1f} dB")
        print(f"   Peak Value: {processed_stats['peak_linear']:.3f}")
        if processed_stats["is_clipping"]:
            print(
                f"   ⚠️  Clipping detected: {processed_stats['clipping_percentage']:.2f}%"
            )
        else:
            print(f"   ✓ No clipping")

        # Calculate gain
        gain_db = processed_stats["rms_db"] - original_stats["rms_db"]
        print(f"\n📈 Gain Applied: {gain_db:+.1f} dB")

        if gain_db > 0:
            print(f"   Audio was amplified (louder)")
        elif gain_db < 0:
            print(f"   Audio was attenuated (quieter)")
        else:
            print(f"   No gain adjustment needed")

        # Save processed audio
        print(f"\n💾 Saving preprocessed audio to: {output_file}")
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), processed_audio, sample_rate)

        output_size = os.path.getsize(output_file)
        print(
            f"   File size: {output_size:,} bytes ({output_size / 1024 / 1024:.2f} MB)"
        )

        print("\n" + "=" * 70)
        print("✅ Preprocessing Complete!")
        print("=" * 70)
        print(f"\nYou can now compare the original and preprocessed audio:")
        print(f"  Original:     {input_file}")
        print(f"  Preprocessed: {output_file}")
        print(f"\nThe preprocessed audio should sound louder and clearer,")
        print(f"which will improve transcription accuracy for quiet speech.")

        return 0

    except ImportError:
        print("\n❌ Error: soundfile library not installed")
        print("   Install with: pip install soundfile")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
