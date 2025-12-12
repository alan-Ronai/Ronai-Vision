"""Hebrew audio transcription using Whisper Large v3 Hebrew model.

Offline transcription service for Hebrew speech recognition.
Uses locally-stored whisper-large-v3-hebrew model.
"""

import os
import numpy as np
import torch
from typing import Optional, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class HebrewTranscriber:
    """Hebrew speech-to-text transcriber using Whisper Large v3 Hebrew.

    This is an offline transcription service that uses a locally downloaded
    Whisper model fine-tuned for Hebrew language.
    """

    def __init__(
        self,
        model_path: str = "models/whisper-large-v3-hebrew",
        device: str = "auto",
        compute_type: str = "float32",
    ):
        """Initialize Hebrew transcriber.

        Args:
            model_path: Path to whisper-large-v3-hebrew model directory
            device: Device to run on ('cpu', 'cuda', 'mps', or 'auto')
            compute_type: Computation precision ('float32', 'float16', 'int8')
        """
        self.model_path = Path(model_path)
        self.device = self._resolve_device(device)
        self.compute_type = compute_type

        self.model = None
        self.processor = None
        self._initialized = False

        logger.info(f"HebrewTranscriber initialized (model will load on first use)")
        logger.info(f"  Model path: {self.model_path}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Compute type: {self.compute_type}")

    def _resolve_device(self, device: str) -> str:
        """Resolve device string to actual device."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            # MPS is very slow for Whisper generation, use CPU instead
            # elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            #     return "mps"
            else:
                return "cpu"
        return device

    def _lazy_load(self):
        """Lazy load model and processor on first transcription."""
        if self._initialized:
            return

        logger.info("Loading Whisper model and processor...")

        try:
            from transformers import WhisperProcessor, WhisperForConditionalGeneration

            # Load processor (tokenizer, feature extractor)
            self.processor = WhisperProcessor.from_pretrained(
                str(self.model_path), local_files_only=True
            )

            # Load model
            self.model = WhisperForConditionalGeneration.from_pretrained(
                str(self.model_path),
                local_files_only=True,
                torch_dtype=torch.float32
                if self.compute_type == "float32"
                else torch.float16,
            )

            # Move to device
            self.model.to(self.device)
            self.model.eval()

            self._initialized = True
            logger.info(f"Whisper model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise

    def transcribe(
        self, audio: np.ndarray, sample_rate: int = 16000, language: str = "he"
    ) -> Dict[str, Any]:
        """Transcribe audio chunk to Hebrew text.

        Args:
            audio: Audio samples as numpy array (float32 or int16)
            sample_rate: Sample rate in Hz (default 16000, Whisper native rate)
            language: Language code (default "he" for Hebrew)

        Returns:
            Dictionary with:
                - text: Transcribed text in Hebrew
                - language: Detected/specified language
                - duration: Audio duration in seconds
                - sample_rate: Input sample rate
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
                "sample_rate": sample_rate,
            }

        # Convert to float32 if needed
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Ensure 1D array
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # Convert stereo to mono

        # Resample if needed (Whisper expects 16kHz)
        if sample_rate != 16000:
            audio = self._resample(audio, sample_rate, 16000)
            sample_rate = 16000

        duration = len(audio) / sample_rate

        try:
            # Process audio through feature extractor
            input_features = self.processor(
                audio, sampling_rate=sample_rate, return_tensors="pt"
            ).input_features

            # Move to device
            input_features = input_features.to(self.device)

            # Generate transcription
            with torch.no_grad():
                # Force Hebrew language and transcription task
                forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                    language=language, task="transcribe"
                )

                # Use faster generation for MPS/CPU
                logger.info("Generating transcription (this may take a moment)...")
                predicted_ids = self.model.generate(
                    input_features,
                    forced_decoder_ids=forced_decoder_ids,
                    max_length=448,  # Whisper max length
                    num_beams=1,  # Greedy decoding (faster than beam search)
                    do_sample=False,
                )

            # Decode to text
            transcription = self.processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]

            logger.info(f"Transcribed {duration:.2f}s audio: '{transcription}'")

            return {
                "text": transcription.strip(),
                "language": language,
                "duration": duration,
                "sample_rate": sample_rate,
            }

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return {
                "text": "",
                "language": language,
                "duration": duration,
                "sample_rate": sample_rate,
                "error": str(e),
            }

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

            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            # Fallback: simple linear interpolation (lower quality)
            logger.warning("librosa not available, using simple resampling")
            from scipy import signal

            num_samples = int(len(audio) * target_sr / orig_sr)
            return signal.resample(audio, num_samples)

    def transcribe_file(self, audio_path: str, language: str = "he") -> Dict[str, Any]:
        """Transcribe audio from file.

        Args:
            audio_path: Path to audio file (WAV, MP3, etc.)
            language: Language code (default "he")

        Returns:
            Transcription result dictionary with segments
        """
        try:
            import soundfile as sf

            # Load audio file
            audio, sample_rate = sf.read(audio_path)

            logger.info(
                f"Loaded audio file: {audio_path} ({len(audio)} samples, {sample_rate}Hz)"
            )

            # Convert to mono if stereo
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            # Whisper can only handle 30 seconds at a time, so chunk it
            chunk_duration = 30  # seconds
            chunk_samples = chunk_duration * sample_rate
            
            full_text = []
            segments = []
            
            for i in range(0, len(audio), chunk_samples):
                chunk = audio[i:i + chunk_samples]
                chunk_start_time = i / sample_rate
                
                logger.info(f"Processing chunk {i//chunk_samples + 1} (starting at {chunk_start_time:.1f}s)")
                
                result = self.transcribe(chunk, sample_rate, language)
                chunk_text = result.get("text", "").strip()
                
                if chunk_text:
                    full_text.append(chunk_text)
                    segments.append({
                        "start": chunk_start_time,
                        "end": min(chunk_start_time + chunk_duration, len(audio) / sample_rate),
                        "text": chunk_text
                    })

            total_duration = len(audio) / sample_rate
            combined_text = " ".join(full_text)
            
            logger.info(f"Transcription complete: {len(segments)} segments, {total_duration:.2f}s total")

            return {
                "text": combined_text,
                "language": language,
                "duration": total_duration,
                "sample_rate": sample_rate,
                "segments": segments
            }

        except Exception as e:
            logger.error(f"Failed to transcribe file {audio_path}: {e}")
            return {"text": "", "language": language, "duration": 0.0, "error": str(e)}

    def is_ready(self) -> bool:
        """Check if transcriber is initialized and ready."""
        return self._initialized

    def unload(self):
        """Unload model from memory."""
        if self._initialized:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            self._initialized = False

            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Whisper model unloaded from memory")
