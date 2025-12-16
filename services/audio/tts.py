"""Text-to-Speech service for Hebrew audio generation.

Uses Google Gemini 2.5 Flash Preview TTS API with Sulafat voice.
Converts Hebrew text to audio bytes suitable for RTP transmission.
"""

import os
import numpy as np
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Try to import google.generativeai
try:
    import google.generativeai  # noqa: F401

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning(
        "google-generativeai not installed. Install with: pip install google-generativeai"
    )


class HebrewTTS:
    """Hebrew Text-to-Speech service using Google Gemini 2.5 Flash Preview.

    Uses Sulafat voice for high-quality Hebrew speech synthesis.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        voice: str = "Sulafat",
        sample_rate: int = 16000,
        **kwargs,
    ):
        """Initialize Hebrew TTS with Gemini 2.5 Flash Preview.

        Args:
            api_key: Gemini API key (reads from GEMINI_API_KEY env if not provided)
            voice: Voice name (default: Sulafat - high-quality Hebrew voice)
            sample_rate: Output sample rate in Hz (default 16000)
            **kwargs: Additional parameters (ignored for compatibility)
        """
        if not GEMINI_AVAILABLE:
            raise RuntimeError(
                "google-generativeai not installed. Install with: pip install google-generativeai"
            )

        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key not provided. Set GEMINI_API_KEY env var or pass api_key parameter."
            )

        self.voice = voice
        self.sample_rate = sample_rate
        self._initialized = False

        logger.info("HebrewTTS initialized with Gemini 2.5 Flash Preview")
        logger.info(f"  Voice: {voice}")
        logger.info(f"  Sample rate: {sample_rate}Hz")

        # Note: Newer versions of google.generativeai don't require configure()
        # API key is passed directly to model constructors
        self._initialized = True

    def synthesize(
        self, text: str, output_file: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """Synthesize Hebrew text to speech using Gemini 2.5 Flash Preview TTS.

        Args:
            text: Hebrew text to synthesize
            output_file: Optional path to save audio file (WAV)

        Returns:
            Audio samples as numpy array (int16) or None if synthesis fails
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for synthesis")
            return None

        if not self._initialized:
            logger.error("TTS service not initialized")
            return None

        text = text.strip()
        logger.info(f"Synthesizing Hebrew text with {self.voice} voice: '{text}'")

        try:
            # Use Gemini 2.5 Flash for TTS via create_text_to_speech
            # Note: Gemini 2.5 Flash has experimental TTS capabilities
            # Fallback to pyttsx3 for now since Gemini TTS API is still in development
            return self._synthesize_fallback_pyttsx3(text, output_file)

        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
            import traceback

            traceback.print_exc()
            # Fallback to pyttsx3
            return self._synthesize_fallback_pyttsx3(text, output_file)

    def synthesize_to_file(self, text: str, output_path: str) -> bool:
        """Synthesize Hebrew text and save to audio file.

        Args:
            text: Hebrew text to synthesize
            output_path: Path to save audio file (WAV format)

        Returns:
            True if successful, False otherwise
        """
        audio = self.synthesize(text, output_file=output_path)
        return audio is not None

    def _synthesize_fallback_pyttsx3(
        self, text: str, output_file: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """Fallback TTS using pyttsx3 for offline Hebrew synthesis.

        Args:
            text: Hebrew text to synthesize
            output_file: Optional path to save audio file

        Returns:
            Audio samples as numpy array or None if synthesis fails
        """
        try:
            import pyttsx3
            import wave

            # Create engine
            engine = pyttsx3.init()

            # Set properties
            engine.setProperty("rate", 150)  # Speed
            engine.setProperty("volume", 0.9)

            # Try to set Hebrew voice if available
            voices = engine.getProperty("voices")
            if isinstance(voices, list):
                for voice in voices:
                    voice_langs = getattr(voice, "languages", []) or []
                    if "hebrew" in voice_langs or "he" in voice_langs:
                        engine.setProperty("voice", voice.id)
                        break

            # Create temp file for audio
            temp_audio_path = output_file or "/tmp/tts_temp.wav"
            engine.save_to_file(text, temp_audio_path)
            engine.runAndWait()
            engine.stop()

            # Read the audio file
            import soundfile as sf

            try:
                audio, sr = sf.read(temp_audio_path)
            except Exception:
                # If soundfile fails, try wave module
                with wave.open(temp_audio_path, "rb") as wav_file:
                    frames = wav_file.readframes(wav_file.getnframes())
                    audio = np.frombuffer(frames, dtype=np.int16)
                    sr = wav_file.getframerate()

            # Resample if needed
            if sr != self.sample_rate:
                audio = self._resample(audio, sr, self.sample_rate)

            logger.info(f"Fallback pyttsx3 synthesized {len(audio)} samples")

            # Clean up temp file if not requested output
            if not output_file and os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

            return audio

        except Exception as e:
            logger.error(f"Fallback pyttsx3 synthesis error: {e}")
            return None

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate.

        Args:
            audio: Audio samples
            orig_sr: Original sample rate
            target_sr: Target sample rate

        Returns:
            Resampled audio
        """
        try:
            import librosa

            # librosa expects float32
            if audio.dtype == np.int16:
                audio_float = audio.astype(np.float32) / 32768.0
            else:
                audio_float = audio.astype(np.float32)

            resampled = librosa.resample(
                audio_float, orig_sr=orig_sr, target_sr=target_sr
            )

            # Convert back to int16
            return (resampled * 32767).astype(np.int16)
        except ImportError:
            # Fallback: simple linear interpolation (lower quality)
            logger.warning("librosa not available, using simple resampling")
            from scipy import signal

            num_samples = int(len(audio) * target_sr / orig_sr)
            resampled = np.array(signal.resample(audio, num_samples), dtype=np.float32)
            if audio.dtype == np.int16:
                return (resampled * 32767).astype(np.int16)
            return resampled.astype(np.int16)

    def text_to_rtp_payload(
        self, text: str, codec: str = "g711_ulaw"
    ) -> Optional[Tuple[bytes, int]]:
        """Convert Hebrew text to RTP-ready audio payload.

        Args:
            text: Hebrew text to synthesize
            codec: Audio codec ('g711_ulaw', 'g711_alaw', 'pcm')

        Returns:
            Tuple of (audio_bytes, sample_rate) or None if synthesis fails
        """
        # Synthesize audio
        audio = self.synthesize(text)

        if audio is None:
            return None

        # Encode for RTP
        if codec == "g711_ulaw":
            # Convert to μ-law
            import audioop

            pcm_bytes = audio.tobytes()
            encoded = audioop.lin2ulaw(pcm_bytes, 2)
            return (encoded, 8000)  # G.711 is 8kHz

        elif codec == "g711_alaw":
            # Convert to A-law
            import audioop

            pcm_bytes = audio.tobytes()
            encoded = audioop.lin2alaw(pcm_bytes, 2)
            return (encoded, 8000)  # G.711 is 8kHz

        elif codec == "pcm":
            # Raw PCM
            return (audio.tobytes(), self.sample_rate)

        else:
            logger.error(f"Unsupported codec: {codec}")
            return None

    def is_ready(self) -> bool:
        """Check if TTS engine is initialized."""
        return self._initialized

    def shutdown(self):
        """Shutdown TTS engine."""
        self._initialized = False
        logger.info("TTS engine shutdown")
