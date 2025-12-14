"""Hebrew audio transcription using faster-whisper with CTranslate2.

Offline transcription service for Hebrew speech recognition.
Uses locally-stored Whisper Large v3 Hebrew model in CTranslate2 format.
This is 4-5x faster than the transformers version.
"""

import os
import numpy as np
from typing import Optional, Dict, Any, List
from pathlib import Path
import logging
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

# Import audio preprocessor
try:
    from .audio_preprocessor import AudioPreprocessor

    PREPROCESSOR_AVAILABLE = True
except ImportError:
    PREPROCESSOR_AVAILABLE = False
    logger.warning("AudioPreprocessor not available - volume normalization disabled")


class HebrewTranscriber:
    """Hebrew speech-to-text transcriber using faster-whisper (CTranslate2).

    This is an offline transcription service that uses a locally downloaded
    Whisper model fine-tuned for Hebrew language in CTranslate2 format.

    4-5x faster than transformers-based Whisper.
    """

    def __init__(
        self,
        model_path: str = "models/whisper-large-v3-hebrew-ct2",
        device: str = "auto",
        compute_type: str = "int8",
        cpu_threads: int = 0,
        num_workers: int = 1,
        enable_preprocessing: bool = False,
        target_audio_level: float = -12.0,
    ):
        """Initialize Hebrew transcriber.

        Args:
            model_path: Path to whisper CT2 model directory (local)
            device: Device to run on ('cpu', 'cuda', or 'auto')
            compute_type: Computation type ('int8', 'int8_float16', 'int16', 'float16', 'float32')
                - int8: Fastest, lowest memory, good quality (recommended for CPU)
                - float16: Faster, needs more memory (good for GPU)
                - float32: Slowest, best quality
            cpu_threads: Number of CPU threads (0 = auto-detect)
            num_workers: Number of parallel workers for batching
            enable_preprocessing: Enable audio volume normalization (NOTE: Can interfere with VAD, use only for extremely quiet audio)
            target_audio_level: Target RMS level in dB (-10 to -12 for very loud, -16 for loud, -20 for normal)
        """
        self.model_path = Path(model_path)
        self.device = self._resolve_device(device)
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self.num_workers = num_workers
        self.enable_preprocessing = enable_preprocessing and PREPROCESSOR_AVAILABLE

        self.model = None
        self._initialized = False

        # Initialize audio preprocessor if enabled
        self.preprocessor = None
        if self.enable_preprocessing:
            self.preprocessor = AudioPreprocessor(
                target_rms=target_audio_level,
                normalization_mode="rms",
                apply_noise_gate=False,  # VAD in transcriber handles this
                remove_dc_offset=True,
            )
            logger.info(
                f"Audio preprocessing enabled (target: {target_audio_level} dB RMS)"
            )
        else:
            logger.info("Audio preprocessing disabled")

        logger.info(f"HebrewTranscriber initialized (model will load on first use)")
        logger.info(f"  Model path: {self.model_path}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Compute type: {self.compute_type}")
        logger.info(
            f"  CPU threads: {self.cpu_threads if self.cpu_threads > 0 else 'auto'}"
        )

    def _resolve_device(self, device: str) -> str:
        """Resolve device string to actual device."""
        if device == "auto":
            try:
                import torch

                if torch.cuda.is_available():
                    return "cuda"
            except ImportError:
                pass
            return "cpu"
        return device

    def _lazy_load(self):
        """Lazy load model on first transcription."""
        if self._initialized:
            return

        logger.info("Loading faster-whisper model (CTranslate2)...")

        try:
            from faster_whisper import WhisperModel

            # Load model from local directory
            self.model = WhisperModel(
                str(self.model_path),
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads,
                num_workers=self.num_workers,
                download_root=None,  # Prevent any downloads
                local_files_only=True,  # Only use local files
            )

            self._initialized = True
            logger.info(f"faster-whisper model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load faster-whisper model: {e}")
            logger.error(f"Make sure the model is downloaded to: {self.model_path}")
            raise

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: str = "he",
        beam_size: int = 1,
        vad_filter: bool = True,
        skip_preprocessing: bool = False,
        sensitive_vad: bool = False,
    ) -> Dict[str, Any]:
        """Transcribe audio chunk to Hebrew text.

        Args:
            audio: Audio samples as numpy array (float32)
            sample_rate: Sample rate in Hz
            language: Language code (default "he" for Hebrew)
            beam_size: Beam size for decoding (1 = greedy, 5 = better quality but slower)
            vad_filter: Enable voice activity detection to skip silence
            skip_preprocessing: Skip preprocessing even if enabled (use when audio is already preprocessed)
            sensitive_vad: Use more sensitive VAD for quieter speech (good for 5-10 dB quieter segments)

        Returns:
            Dictionary with:
                - text: Transcribed text in Hebrew
                - language: Detected/specified language
                - duration: Audio duration in seconds
                - segments: List of segments with timestamps
        """
        # Lazy load model
        self._lazy_load()

        # Validate input
        if audio is None or len(audio) == 0:
            logger.warning("Empty audio provided for transcription")
            return {
                "text": "",
                "language": language,
                "duration": 0.0,
                "segments": [],
            }

        # Convert to float32 if needed
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Ensure 1D array
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # Convert stereo to mono

        # Apply audio preprocessing (volume normalization) if enabled and not skipped
        if self.preprocessor is not None and not skip_preprocessing:
            audio = self.preprocessor.process(audio, sample_rate)

        duration = len(audio) / sample_rate

        try:
            # Transcribe with faster-whisper
            # Quality parameters without aggressive VAD customization

            # Prepare VAD parameters if sensitive mode is enabled
            vad_params = None
            if sensitive_vad and vad_filter:
                # Very sensitive for quieter speech (5-10 dB below average)
                vad_params = {
                    "threshold": 0.04,  # Speech probability threshold (0.1 = very sensitive)
                    "min_speech_duration_ms": 10,  # Current: Minimum 50ms to count as speech
                    "min_silence_duration_ms": 60,  # Current: 300ms silence to split segments
                    "speech_pad_ms": 300,  # NEW: Add padding around detected speech
                    "max_speech_duration_s": 30.0,  # NEW: Max continuous speech before forcing split
                }

            segments_iter, info = self.model.transcribe(
                audio,
                language=language,
                beam_size=beam_size,
                best_of=5,  # Better quality
                temperature=[
                    0.0,
                    0.2,
                    0.4,
                    0.6,
                    0.8,
                ],  # Progressive fallback for difficult audio
                vad_filter=vad_filter,
                vad_parameters=vad_params,  # Use custom params only if sensitive_vad=True
                condition_on_previous_text=False,  # Better context
                initial_prompt="תמלול אודיו בעברית של שיח צבאי בין מספר אנשים.",  # Hebrew context
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

            transcription = " ".join(full_text)

            logger.info(
                f"Transcribed {duration:.2f}s audio: '{transcription[:100]}...'"
            )

            return {
                "text": transcription,
                "language": info.language if hasattr(info, "language") else language,
                "duration": duration,
                "segments": segments,
            }

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return {
                "text": "",
                "language": language,
                "duration": duration,
                "segments": [],
                "error": str(e),
            }

    def transcribe_file(
        self,
        audio_path: str,
        language: str = "he",
        beam_size: int = 1,
        vad_filter: bool = True,
        save_preprocessed: Optional[str] = None,
        sensitive_vad: bool = False,
    ) -> Dict[str, Any]:
        """Transcribe audio from file.

        Args:
            audio_path: Path to audio file (WAV, MP3, etc.)
            language: Language code (default "he")
            beam_size: Beam size for decoding (1 = fastest, 5 = better quality)
            vad_filter: Enable voice activity detection
            save_preprocessed: Path to save preprocessed audio (WAV format)
            sensitive_vad: Use more sensitive VAD for quieter speech

        Returns:
            Transcription result dictionary with segments
        """
        # Lazy load model
        self._lazy_load()

        try:
            logger.info(f"Transcribing file: {audio_path}")

            # Transcribe directly from file (faster-whisper handles file loading)
            # Quality parameters with default VAD (works best)

            # Prepare VAD parameters if sensitive mode is enabled
            vad_params = None
            if sensitive_vad and vad_filter:
                vad_params = {
                    "threshold": 0.02,  # Speech probability threshold (0.1 = very sensitive)
                    "min_speech_duration_ms": 7,  # Current: Minimum 50ms to count as speech
                    "min_silence_duration_ms": 130,  # Current: 300ms silence to split segments
                    "speech_pad_ms": 400,  # NEW: Add padding around detected speech
                    "max_speech_duration_s": 20.0,  # NEW: Max continuous speech before forcing split
                }

            segments_iter, info = self.model.transcribe(
                audio_path,
                language=language,
                beam_size=beam_size,
                best_of=5,
                temperature=[0.0, 0.2, 0.4, 0.6, 0.8],
                vad_filter=vad_filter,
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

            transcription = " ".join(full_text)
            duration = info.duration if hasattr(info, "duration") else 0.0

            logger.info(
                f"Transcription complete: {len(segments)} segments, {duration:.2f}s total"
            )

            return {
                "text": transcription,
                "language": info.language if hasattr(info, "language") else language,
                "duration": duration,
                "segments": segments,
            }

        except Exception as e:
            logger.error(f"Failed to transcribe file {audio_path}: {e}")
            return {
                "text": "",
                "language": language,
                "duration": 0.0,
                "segments": [],
                "error": str(e),
            }

    def is_ready(self) -> bool:
        """Check if transcriber is initialized and ready."""
        return self._initialized

    def unload(self):
        """Unload model from memory."""
        if self._initialized:
            del self.model
            self.model = None
            self._initialized = False
            logger.info("Model unloaded from memory")
