"""Test script for Hebrew audio processing system.

Tests transcription, command processing, and TTS independently.
"""

import numpy as np
from services.audio import HebrewTranscriber, CommandProcessor, HebrewTTS


def test_command_processor():
    """Test command processor with Hebrew text."""
    print("\n" + "=" * 60)
    print("TESTING COMMAND PROCESSOR")
    print("=" * 60)

    processor = CommandProcessor()

    # Test cases
    test_texts = [
        "עקוב אחרי אדם",  # Track person
        "זום פנימה",  # Zoom in
        "מה הסטטוס",  # Status report
        "פנה שמאלה",  # Pan left
        "עצור",  # Stop
        "שלום זה טקסט רגיל",  # No command
    ]

    for text in test_texts:
        print(f"\nText: '{text}'")
        match = processor.process(text)
        if match:
            print(f"  ✓ Command detected: {match.command_id}")
            print(f"    Keyword: {match.keyword}")
            print(f"    Parameters: {match.parameters}")
        else:
            print(f"  ✗ No command detected")

    # Show history
    print("\n" + "-" * 60)
    print("Command History:")
    history = processor.get_history(limit=10)
    for i, cmd in enumerate(history, 1):
        print(f"{i}. {cmd.command_id}: '{cmd.text}'")


def test_tts():
    """Test Hebrew TTS synthesis."""
    print("\n" + "=" * 60)
    print("TESTING HEBREW TTS")
    print("=" * 60)

    try:
        tts = HebrewTTS(sample_rate=16000)

        test_texts = [
            "שלום עולם",  # Hello world
            "המערכת פועלת כרגיל",  # System operating normally
            "מתחיל מעקב",  # Starting tracking
        ]

        for text in test_texts:
            print(f"\nSynthesizing: '{text}'")
            audio = tts.synthesize(text)

            if audio is not None:
                print(f"  ✓ Generated {len(audio)} samples at {tts.sample_rate}Hz")
                print(f"    Duration: {len(audio) / tts.sample_rate:.2f}s")
            else:
                print(f"  ✗ Synthesis failed")

    except Exception as e:
        print(f"\n✗ TTS initialization failed: {e}")
        print("  Make sure espeak is installed:")
        print("    macOS: brew install espeak")
        print("    Linux: sudo apt-get install espeak")


def test_transcriber():
    """Test Hebrew transcriber (requires model to be loaded)."""
    print("\n" + "=" * 60)
    print("TESTING HEBREW TRANSCRIBER")
    print("=" * 60)

    print("\nNote: This will load Whisper model (~2-3GB RAM)")
    print("Press Ctrl+C to skip if you want to test later.")

    try:
        import time

        time.sleep(2)  # Give user time to cancel

        transcriber = HebrewTranscriber(
            model_path="models/whisper-large-v3-hebrew",
            device="cpu",  # Use CPU for testing
        )

        # Create dummy audio (silence) - in real use, this would be actual speech
        print("\nCreating test audio (1 second of silence)...")
        sample_rate = 16000
        duration = 1.0
        audio = np.zeros(int(sample_rate * duration), dtype=np.float32)

        print("Transcribing...")
        result = transcriber.transcribe(audio, sample_rate)

        print(f"\n✓ Transcription result:")
        print(f"  Text: '{result['text']}'")
        print(f"  Language: {result['language']}")
        print(f"  Duration: {result['duration']:.2f}s")

        print(f"\n✓ Transcriber is ready: {transcriber.is_ready()}")

    except KeyboardInterrupt:
        print("\n\n⊘ Transcriber test skipped (Ctrl+C)")
    except Exception as e:
        print(f"\n✗ Transcriber test failed: {e}")
        print("  Make sure whisper-large-v3-hebrew model is in models/ directory")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("HEBREW AUDIO PROCESSING SYSTEM - COMPONENT TESTS")
    print("=" * 60)

    # Test 1: Command Processor (no dependencies)
    test_command_processor()

    # Test 2: TTS (requires espeak)
    test_tts()

    # Test 3: Transcriber (requires model, takes time)
    test_transcriber()

    print("\n" + "=" * 60)
    print("TESTS COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Start server: uvicorn api.server:app")
    print("2. Start audio server: curl -X POST http://localhost:8000/api/audio/start")
    print(
        "3. Start transcription: curl -X POST http://localhost:8000/api/audio/transcription/start"
    )
    print("4. Connect field device to rtsp://SERVER_IP:8554/audio")
    print()


if __name__ == "__main__":
    main()
