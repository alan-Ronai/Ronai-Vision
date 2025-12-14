"""Analyze audio segments to understand why transcription is missing words.

This script loads an audio file and analyzes specific time segments to identify
issues like low volume, frequency problems, or other factors that might cause
the model to miss words.

Usage:
    python scripts/analyze_audio_segments.py <audio_file> <start1>-<end1> <start2>-<end2> ...
    python scripts/analyze_audio_segments.py assets/test_audio.wav 3-5 27-30
"""

import sys
import os
from pathlib import Path
import warnings
import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))


def analyze_segment(audio, sample_rate, start_sec, end_sec, label="Segment"):
    """Analyze a specific time segment of audio."""
    start_sample = int(start_sec * sample_rate)
    end_sample = int(end_sec * sample_rate)
    segment = audio[start_sample:end_sample]

    # Compute statistics
    rms = np.sqrt(np.mean(segment**2))
    rms_db = 20 * np.log10(rms) if rms > 1e-10 else -100

    peak = np.abs(segment).max()
    peak_db = 20 * np.log10(peak) if peak > 1e-10 else -100

    # Compute frequency content (simple)
    fft = np.fft.rfft(segment)
    freqs = np.fft.rfftfreq(len(segment), 1 / sample_rate)
    magnitude = np.abs(fft)

    # Find dominant frequency
    dominant_idx = np.argmax(magnitude)
    dominant_freq = freqs[dominant_idx]

    # Energy in speech range (300-3400 Hz typical for speech)
    speech_mask = (freqs >= 300) & (freqs <= 3400)
    speech_energy = np.sum(magnitude[speech_mask] ** 2)
    total_energy = np.sum(magnitude**2)
    speech_ratio = speech_energy / total_energy if total_energy > 0 else 0

    # Zero crossing rate (indicates voice vs noise)
    zero_crossings = np.sum(np.abs(np.diff(np.sign(segment)))) / 2
    zcr = zero_crossings / len(segment) * sample_rate

    return {
        "label": label,
        "duration": end_sec - start_sec,
        "rms_linear": rms,
        "rms_db": rms_db,
        "peak_linear": peak,
        "peak_db": peak_db,
        "dominant_freq": dominant_freq,
        "speech_energy_ratio": speech_ratio,
        "zero_crossing_rate": zcr,
        "samples": len(segment),
    }


def main():
    if len(sys.argv) < 3:
        print("❌ Error: Need audio file and time ranges")
        print(
            f"\nUsage: python {sys.argv[0]} <audio_file> <start1>-<end1> <start2>-<end2> ..."
        )
        print(f"Example: python {sys.argv[0]} assets/test_audio.wav 3-5 27-30")
        return 1

    audio_file = sys.argv[1]
    time_ranges = sys.argv[2:]

    if not os.path.exists(audio_file):
        print(f"❌ File not found: {audio_file}")
        return 1

    try:
        import soundfile as sf

        print("=" * 70)
        print("Audio Segment Analysis")
        print("=" * 70)
        print(f"\n📁 File: {audio_file}\n")

        # Load audio
        audio, sample_rate = sf.read(audio_file, dtype="float32")
        duration = len(audio) / sample_rate

        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        print(f"Sample Rate: {sample_rate} Hz")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Channels: Mono\n")

        # Parse time ranges
        segments_to_analyze = []
        for time_range in time_ranges:
            try:
                start, end = map(float, time_range.split("-"))
                segments_to_analyze.append((start, end))
            except:
                print(
                    f"⚠️  Invalid time range: {time_range} (expected format: start-end)"
                )
                continue

        if not segments_to_analyze:
            print("❌ No valid time ranges provided")
            return 1

        # Analyze full audio for baseline
        print("📊 BASELINE (Full Audio):")
        print("-" * 70)
        baseline = analyze_segment(audio, sample_rate, 0, duration, "Full Audio")
        print(f"  RMS Level: {baseline['rms_db']:.1f} dB")
        print(f"  Peak Level: {baseline['peak_db']:.1f} dB")
        print(f"  Speech Energy Ratio: {baseline['speech_energy_ratio']:.1%}")
        print(f"  Zero Crossing Rate: {baseline['zero_crossing_rate']:.0f} Hz")
        print(f"  Dominant Frequency: {baseline['dominant_freq']:.0f} Hz")
        print()

        # Analyze each segment
        print("🔍 PROBLEM SEGMENTS:")
        print("-" * 70)

        for i, (start, end) in enumerate(segments_to_analyze, 1):
            if end > duration:
                print(
                    f"\n⚠️  Segment {i} ({start}-{end}s) extends beyond audio duration"
                )
                continue

            result = analyze_segment(audio, sample_rate, start, end, f"Segment {i}")

            print(
                f"\n📍 Segment {i}: {start:.1f}s - {end:.1f}s ({result['duration']:.1f}s)"
            )
            print(
                f"  RMS Level: {result['rms_db']:.1f} dB (baseline: {baseline['rms_db']:.1f} dB)"
            )
            print(
                f"  Peak Level: {result['peak_db']:.1f} dB (baseline: {baseline['peak_db']:.1f} dB)"
            )
            print(
                f"  Speech Energy: {result['speech_energy_ratio']:.1%} (baseline: {baseline['speech_energy_ratio']:.1%})"
            )
            print(
                f"  Zero Crossing Rate: {result['zero_crossing_rate']:.0f} Hz (baseline: {baseline['zero_crossing_rate']:.0f} Hz)"
            )
            print(
                f"  Dominant Frequency: {result['dominant_freq']:.0f} Hz (baseline: {baseline['dominant_freq']:.0f} Hz)"
            )

            # Compute differences
            rms_diff = result["rms_db"] - baseline["rms_db"]
            peak_diff = result["peak_db"] - baseline["peak_db"]

            print(f"\n  📉 Differences from baseline:")
            print(f"     RMS: {rms_diff:+.1f} dB")
            print(f"     Peak: {peak_diff:+.1f} dB")

            # Diagnosis
            print(f"\n  🔬 Diagnosis:")
            issues = []

            if rms_diff < -10:
                issues.append(f"⚠️  VERY QUIET: {-rms_diff:.1f} dB quieter than average")
            elif rms_diff < -5:
                issues.append(f"⚠️  Quiet: {-rms_diff:.1f} dB quieter than average")

            if result["speech_energy_ratio"] < baseline["speech_energy_ratio"] * 0.5:
                issues.append(f"⚠️  Low speech frequencies (possible noise/silence)")

            if result["zero_crossing_rate"] < baseline["zero_crossing_rate"] * 0.3:
                issues.append(f"⚠️  Very low ZCR (likely silence or very low frequency)")
            elif result["zero_crossing_rate"] > baseline["zero_crossing_rate"] * 2.0:
                issues.append(f"⚠️  Very high ZCR (likely noise)")

            if result["peak_linear"] < 0.01:
                issues.append(f"⚠️  Extremely low amplitude (near silence)")

            if issues:
                for issue in issues:
                    print(f"     {issue}")
            else:
                print(f"     ✓ Audio properties look normal")
                print(
                    f"     ℹ️  Model might be struggling with Hebrew phonetics or accent"
                )

        # Summary and recommendations
        print("\n" + "=" * 70)
        print("💡 RECOMMENDATIONS:")
        print("=" * 70)

        all_results = [
            analyze_segment(audio, sample_rate, s, e, f"Seg{i}")
            for i, (s, e) in enumerate(segments_to_analyze, 1)
        ]

        quietest_rms = min(r["rms_db"] for r in all_results)
        avg_problem_rms = np.mean([r["rms_db"] for r in all_results])

        if quietest_rms < baseline["rms_db"] - 10:
            print("\n1. 🔊 VOLUME ISSUE DETECTED:")
            print(
                f"   - Problem segments are {baseline['rms_db'] - quietest_rms:.1f} dB quieter"
            )
            print(f"   - Whisper's VAD may be filtering these as silence")
            print(f"   - Solutions:")
            print(f"     a) Boost volume in audio editor before transcription")
            print(f"     b) Use preprocessing with VAD disabled:")
            print(f"        transcriber = HebrewTranscriber(enable_preprocessing=True)")
            print(
                f"        result = transcriber.transcribe_file('file.wav', vad_filter=False)"
            )
            print(f"     c) Use external VAD with lower threshold")

        elif all(
            r["speech_energy_ratio"] < baseline["speech_energy_ratio"] * 0.6
            for r in all_results
        ):
            print("\n2. 🎵 FREQUENCY ISSUE DETECTED:")
            print(f"   - Problem segments have low speech frequency content")
            print(f"   - May be background noise, breath sounds, or very quiet speech")
            print(f"   - Solution: Check if these segments actually contain speech")

        else:
            print("\n3. 🎤 AUDIO QUALITY IS ADEQUATE:")
            print(f"   - Problem segments have reasonable volume and frequency content")
            print(f"   - The issue is likely:")
            print(f"     • Hebrew pronunciation/accent recognition")
            print(f"     • Fast speech or unclear articulation")
            print(f"     • Model confidence thresholds")
            print(f"   - Solutions:")
            print(f"     a) Try with VAD disabled to force transcription")
            print(f"     b) Try different Whisper model size (large vs medium)")
            print(f"     c) Increase temperature for more aggressive decoding")
            print(f"     d) Lower beam size for faster, less conservative decoding")

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
