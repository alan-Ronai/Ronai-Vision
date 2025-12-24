"""Hebrew Whisper Transcriber using faster-whisper (CTranslate2).

Minimal implementation for file and live audio transcription.
"""

import logging
import time
from pathlib import Path
from typing import Optional, List, Dict, Callable
from dataclasses import dataclass
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

# Try to import faster-whisper
try:
    from faster_whisper import WhisperModel

    WHISPER_AVAILABLE = True
except ImportError:
    WhisperModel = None
    WHISPER_AVAILABLE = False
    logger.warning("faster-whisper not installed. Run: pip install faster-whisper")


@dataclass
class TranscriptionResult:
    """Result of transcription."""

    text: str
    timestamp: datetime
    duration_seconds: float
    segments: Optional[List[Dict]] = None


class WhisperTranscriber:
    """Hebrew speech-to-text transcriber using faster-whisper."""

    def __init__(
        self,
        model_path: str = "ivrit-ai/whisper-large-v3-turbo-ct2",
        device: str = "auto",
        compute_type: str = "int8",
        sample_rate: int = 16000,
    ):
        """Initialize transcriber.

        Args:
            model_path: Path to local model or HuggingFace model ID
            device: 'cpu', 'cuda', or 'auto'
            compute_type: 'int8', 'float16', 'float32'
            sample_rate: Audio sample rate (default 16000)
        """
        self.model_path = model_path
        self.device = self._resolve_device(device)
        self.compute_type = compute_type
        self.sample_rate = sample_rate
        self.model = None
        self._initialized = False

        logger.info(f"  Transcriber initialized")
        # logger.info(f"  Model: {model_path}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Compute type: {compute_type}")

    def _resolve_device(self, device: str) -> str:
        """Resolve 'auto' device to actual device."""
        if device == "auto":
            try:
                import torch

                if torch.cuda.is_available():
                    return "cuda"
            except ImportError:
                pass
            return "cpu"
        return device

    def load_model(self):
        """Load the Whisper model."""
        if self._initialized:
            return

        if not WHISPER_AVAILABLE:
            raise RuntimeError("faster-whisper not installed")

        # logger.info("Loading faster-whisper model...")
        start_time = time.time()

        self.model = WhisperModel(
            self.model_path,
            device=self.device,
            compute_type=self.compute_type,
            local_files_only=True,  # Allow downloading from HuggingFace
        )

        load_time = time.time() - start_time
        self._initialized = True
        logger.info(f"Model loaded in {load_time:.1f}s on {self.device}")

    def transcribe_file(self, filepath: str) -> Optional[TranscriptionResult]:
        """Transcribe an audio file.

        Args:
            filepath: Path to audio file (WAV, MP3, etc.)

        Returns:
            TranscriptionResult or None if failed
        """
        self.load_model()

        if not self._initialized or self.model is None:
            logger.error("Model not loaded")
            return None

        try:
            logger.info(f"Transcribing file: {filepath}")
            start_time = time.time()

            # VAD parameters for Hebrew military radio
            vad_params = {
                "threshold": 0.02,  # Speech probability threshold (very sensitive)
                "min_speech_duration_ms": 7,  # Minimum 7ms to count as speech
                "min_silence_duration_ms": 130,  # 130ms silence to split segments
                "speech_pad_ms": 400,  # Add padding around detected speech
                "max_speech_duration_s": 20.0,  # Max continuous speech before forcing split
            }

            segments_iter, info = self.model.transcribe(
                filepath,
                language="he",
                beam_size=1,
                best_of=5,
                temperature=[0.0, 0.2, 0.4, 0.6, 0.8],
                vad_filter=True,
                vad_parameters=vad_params,
                condition_on_previous_text=False,
                initial_prompt="תמלול אודיו בעברית של שיח צבאי בין מספר אנשים.",
            )

            # Collect segments
            segments = []
            full_text = []
            for segment in segments_iter:
                segments.append(
                    {
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text.strip(),
                    }
                )
                full_text.append(segment.text.strip())

            text = " ".join(full_text)
            duration = info.duration if hasattr(info, "duration") else 0.0
            elapsed = time.time() - start_time

            logger.info(f"Transcription complete in {elapsed:.1f}s: '{text[:100]}...'")

            return TranscriptionResult(
                text=text,
                timestamp=datetime.now(),
                duration_seconds=duration,
                segments=segments,
            )

        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            return None

    def transcribe_audio(self, audio_bytes: bytes) -> Optional[TranscriptionResult]:
        """Transcribe raw PCM audio bytes.

        Args:
            audio_bytes: Raw 16-bit PCM audio at self.sample_rate

        Returns:
            TranscriptionResult or None if failed
        """
        self.load_model()

        if not self._initialized or self.model is None:
            logger.error("Model not loaded")
            return None

        try:
            duration = len(audio_bytes) / (2 * self.sample_rate)
            logger.info(
                f"Transcribing audio buffer: {len(audio_bytes)} bytes, {duration:.2f}s"
            )
            start_time = time.time()

            # Convert bytes to float32 numpy array
            audio = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_float = audio.astype(np.float32) / 32768.0

            # VAD parameters
            vad_params = {
                "threshold": 0.02,  # Speech probability threshold (very sensitive)
                "min_speech_duration_ms": 7,  # Minimum 7ms to count as speech
                "min_silence_duration_ms": 130,  # 130ms silence to split segments
                "speech_pad_ms": 400,  # Add padding around detected speech
                "max_speech_duration_s": 20.0,  # Max continuous speech before forcing split
            }

            segments_iter, _info = self.model.transcribe(
                audio_float,
                language="he",
                beam_size=1,
                best_of=5,
                temperature=[0.0, 0.2, 0.4, 0.6, 0.8],
                vad_filter=True,
                vad_parameters=vad_params,
                condition_on_previous_text=False,
                initial_prompt="תמלול אודיו בעברית של שיח צבאי בין מספר אנשים.",
            )

            # Collect segments
            segments = []
            full_text = []
            for segment in segments_iter:
                segments.append(
                    {
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text.strip(),
                    }
                )
                full_text.append(segment.text.strip())

            text = " ".join(full_text)
            elapsed = time.time() - start_time

            if text:
                logger.info(
                    f"Transcription complete in {elapsed:.1f}s: '{text[:100]}...'"
                )
            else:
                logger.info(f"No speech detected in {duration:.1f}s audio")

            return TranscriptionResult(
                text=text,
                timestamp=datetime.now(),
                duration_seconds=duration,
                segments=segments,
            )

        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            return None

    def is_ready(self) -> bool:
        """Check if transcriber is ready."""
        return self._initialized and self.model is not None
