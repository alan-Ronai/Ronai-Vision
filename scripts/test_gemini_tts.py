#!/usr/bin/env python3
"""Test Gemini 2.5 Flash TTS with Sulafat voice."""

import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.audio.tts import HebrewTTS

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_tts():
    """Test Gemini 2.5 Flash TTS."""

    # Check for API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set. Testing will use fallback pyttsx3")
        logger.info("To test Gemini 2.5 Flash TTS:")
        logger.info("  1. Get API key from https://makersuite.google.com/app/apikey")
        logger.info("  2. Set: export GEMINI_API_KEY=your-api-key")
        logger.info("  3. Run: python scripts/test_gemini_tts.py")

    # Initialize TTS
    logger.info("Initializing Hebrew TTS with Sulafat voice...")
    try:
        tts = HebrewTTS(api_key=api_key or "test-key")
        logger.info("✅ TTS initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize TTS: {e}")
        return

    # Test Hebrew text synthesis
    test_texts = [
        "שלום עולם",  # Hello world
        "זהו בדיקה של סינתזה קולית",  # This is a test of speech synthesis
        "הקול של סולאפט נשמע טוב",  # The Sulafat voice sounds good
    ]

    logger.info("Testing TTS synthesis...")
    for text in test_texts:
        logger.info(f"Synthesizing: {text}")
        try:
            audio = tts.synthesize(text)
            if audio is not None:
                logger.info(f"✅ Synthesized successfully: {len(audio)} samples")
            else:
                logger.warning(
                    f"⚠️  Synthesis returned None (fallback may have been used)"
                )
        except Exception as e:
            logger.error(f"❌ Synthesis failed: {e}")

    # Test file output
    output_file = "output/test_tts_output.wav"
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    logger.info(f"Testing file output to: {output_file}")
    try:
        success = tts.synthesize_to_file("בדיקה של שמירה לקובץ", output_file)
        if success:
            logger.info(f"✅ Audio saved to {output_file}")
            if os.path.exists(output_file):
                size = os.path.getsize(output_file)
                logger.info(f"   File size: {size} bytes")
        else:
            logger.warning(f"⚠️  synthesize_to_file returned False")
    except Exception as e:
        logger.error(f"❌ File output failed: {e}")

    # Test RTP payload
    logger.info("Testing RTP payload generation...")
    try:
        for codec in ["pcm", "g711_ulaw", "g711_alaw"]:
            result = tts.text_to_rtp_payload("בדיקה", codec=codec)
            if result:
                payload, sr = result
                logger.info(f"✅ {codec}: {len(payload)} bytes at {sr}Hz")
            else:
                logger.warning(
                    f"⚠️  {codec}: payload generation failed (fallback may have been used)"
                )
    except Exception as e:
        logger.error(f"❌ RTP payload test failed: {e}")

    logger.info("TTS testing complete!")


if __name__ == "__main__":
    test_tts()
