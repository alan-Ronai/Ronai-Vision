"""Text-to-Speech service for Hebrew audio generation.

Offline Hebrew TTS using pyttsx3 or fallback to espeak.
Converts Hebrew text to audio bytes suitable for RTP transmission.
"""

import os
import numpy as np
import logging
from typing import Optional, Tuple
from pathlib import Path
import tempfile

logger = logging.getLogger(__name__)


class HebrewTTS:
    """Hebrew Text-to-Speech service.

    Generates Hebrew speech audio from text input.
    Uses offline engines (pyttsx3 with espeak backend).
    """

    def __init__(self, sample_rate: int = 16000, engine: str = "pyttsx3"):
        """Initialize Hebrew TTS.

        Args:
            sample_rate: Output sample rate in Hz (default 16000)
            engine: TTS engine to use ('pyttsx3' or 'espeak')
        """
        self.sample_rate = sample_rate
        self.engine_name = engine
        self.engine = None
        self._initialized = False

        logger.info(f"HebrewTTS initialized with {engine} engine")
        logger.info(f"  Sample rate: {sample_rate}Hz")

    def _lazy_load(self):
        """Lazy load TTS engine on first use."""
        if self._initialized:
            return

        logger.info(f"Loading {self.engine_name} TTS engine...")

        try:
            if self.engine_name == "pyttsx3":
                import pyttsx3

                self.engine = pyttsx3.init()

                # Configure for Hebrew
                voices = self.engine.getProperty("voices")

                # Try to find Hebrew voice
                hebrew_voice = None
                for voice in voices:
                    if "hebrew" in voice.name.lower() or "he" in voice.languages:
                        hebrew_voice = voice
                        break

                if hebrew_voice:
                    self.engine.setProperty("voice", hebrew_voice.id)
                    logger.info(f"Using Hebrew voice: {hebrew_voice.name}")
                else:
                    logger.warning("No Hebrew voice found, using default")

                # Set rate and volume
                self.engine.setProperty("rate", 150)  # Speech rate
                self.engine.setProperty("volume", 1.0)  # Volume (0.0 to 1.0)

            else:
                raise ValueError(f"Unsupported engine: {self.engine_name}")

            self._initialized = True
            logger.info("TTS engine loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load TTS engine: {e}")
            logger.error(
                "Install espeak for Hebrew TTS: sudo apt-get install espeak (Linux) or brew install espeak (macOS)"
            )
            raise

    def synthesize(
        self, text: str, output_file: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """Synthesize Hebrew text to speech audio.

        Args:
            text: Hebrew text to synthesize
            output_file: Optional path to save audio file (WAV)

        Returns:
            Audio samples as numpy array (int16) or None if synthesis fails
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for synthesis")
            return None

        # Lazy load engine
        self._lazy_load()

        text = text.strip()
        logger.info(f"Synthesizing Hebrew text: '{text}'")

        try:
            # Create temporary file if no output specified
            temp_file = None
            if output_file is None:
                temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                output_file = temp_file.name
                temp_file.close()

            # Synthesize to file
            self.engine.save_to_file(text, output_file)
            self.engine.runAndWait()

            # Load audio from file
            import soundfile as sf

            audio, sr = sf.read(output_file)

            # Convert to int16 if needed
            if audio.dtype == np.float32 or audio.dtype == np.float64:
                audio = (audio * 32767).astype(np.int16)

            # Resample if needed
            if sr != self.sample_rate:
                audio = self._resample(audio, sr, self.sample_rate)

            logger.info(f"Synthesized {len(audio)} samples at {self.sample_rate}Hz")

            # Clean up temporary file
            if temp_file:
                try:
                    os.unlink(output_file)
                except:
                    pass

            return audio

        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
            return None

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
            resampled = signal.resample(audio, num_samples)
            if audio.dtype == np.int16:
                return resampled.astype(np.int16)
            return resampled

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
        if self._initialized and self.engine:
            try:
                self.engine.stop()
            except:
                pass
            self.engine = None
            self._initialized = False
            logger.info("TTS engine shutdown")
