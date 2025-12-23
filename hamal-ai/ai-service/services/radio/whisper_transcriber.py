"""Whisper-based Hebrew audio transcription using faster-whisper with CTranslate2.

Offline transcription service for Hebrew speech recognition.
Uses locally-stored Whisper Large v3 Hebrew model in CTranslate2 format.
This is 4-5x faster than the transformers version.
"""

import io
import wave
import asyncio
import logging
import time
import warnings
from pathlib import Path
from typing import Optional, Callable, List, Dict, Union
from dataclasses import dataclass
from concurrent.futures import Future
from datetime import datetime
import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

# Try to import faster-whisper
try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WhisperModel = None  # type: ignore
    WHISPER_AVAILABLE = False
    logger.warning("faster-whisper not installed. Whisper transcription will be disabled.")


@dataclass
class TranscriptionResult:
    """Result of transcription."""
    text: str
    timestamp: datetime
    duration_seconds: float
    confidence: Optional[float] = None
    is_command: bool = False
    command_type: Optional[str] = None
    segments: Optional[List[Dict]] = None


# Hebrew voice commands to detect (same as Gemini transcriber)
VOICE_COMMANDS = {
    "רחפן": "drone_dispatch",
    "הקפיצו": "drone_dispatch",
    "חוזי": "drone_dispatch",
    "צפרדע": "code_broadcast",
    "התקשרו": "phone_call",
    "חייגו": "phone_call",
    "כריזה": "pa_announcement",
    "מגורים": "pa_announcement",
    "חדל": "end_incident",
    "נוטרל": "end_incident",
    "סיום": "end_incident",
}


class WhisperTranscriber:
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
        sample_rate: int = 16000,
        chunk_duration: float = 5.0,
        silence_threshold: float = 500.0,
        silence_duration: float = 1.5,
        min_duration: float = 1.5,
        idle_timeout: float = 2.0,
        save_audio: bool = False,
        on_transcription: Optional[Callable[[TranscriptionResult], None]] = None,
    ):
        """Initialize Whisper transcriber.

        Args:
            model_path: Path to whisper CT2 model directory (local)
            device: Device to run on ('cpu', 'cuda', or 'auto')
            compute_type: Computation type ('int8', 'int8_float16', 'int16', 'float16', 'float32')
            cpu_threads: Number of CPU threads (0 = auto-detect)
            num_workers: Number of parallel workers for batching
            sample_rate: Audio sample rate in Hz
            chunk_duration: Seconds of audio to accumulate before transcribing
            silence_threshold: RMS threshold for silence detection
            silence_duration: Seconds of silence to trigger processing
            min_duration: Minimum audio duration before processing
            idle_timeout: Seconds of no audio before processing buffer
            save_audio: Whether to save audio files for debugging
            on_transcription: Callback when transcription is ready
        """
        self.model_path = Path(model_path)
        self.device = self._resolve_device(device)
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self.num_workers = num_workers
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.min_duration = min_duration
        self.idle_timeout = idle_timeout
        self.save_audio = save_audio
        self.on_transcription = on_transcription

        self.model = None
        self._initialized = False

        # Audio output directory for debugging
        if self.save_audio:
            self.audio_output_dir = Path("audio_output_whisper")
            self.audio_output_dir.mkdir(exist_ok=True)
            logger.info(f"Whisper audio files will be saved to: {self.audio_output_dir.absolute()}")
        else:
            self.audio_output_dir = None

        # Audio buffer
        self._audio_buffer: List[bytes] = []
        self._buffer_samples = 0
        self._samples_per_chunk = int(sample_rate * chunk_duration)

        # Silence detection state
        self._silence_samples = 0
        self._samples_per_silence = int(sample_rate * silence_duration)
        self._min_samples = int(sample_rate * min_duration)
        self._has_speech = False

        # Idle timeout state
        self._last_audio_time = 0.0
        self._idle_check_task: Union[asyncio.Task, Future, None] = None

        # Processing state
        self._processing = False
        self._last_transcription_time = 0.0

        # Event loop for cross-thread scheduling
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

        # Statistics
        self._stats = {
            "chunks_processed": 0,
            "total_duration": 0.0,
            "transcriptions": 0,
            "errors": 0,
        }

        logger.info("WhisperTranscriber initialized")
        logger.info(f"  Model path: {self.model_path}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Compute type: {self.compute_type}")

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

        if not WHISPER_AVAILABLE or WhisperModel is None:
            logger.error("faster-whisper not available, cannot load model")
            return

        logger.info("Loading faster-whisper model (CTranslate2)...")

        try:
            # Check if model path exists
            if not self.model_path.exists():
                logger.error(f"Model path does not exist: {self.model_path}")
                logger.error("Please download the model first")
                return

            self.model = WhisperModel(  # type: ignore[misc]
                str(self.model_path),
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads,
                num_workers=self.num_workers,
                download_root=None,
                local_files_only=True,
            )

            self._initialized = True
            logger.info(f"faster-whisper model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load faster-whisper model: {e}")
            logger.error(f"Make sure the model is downloaded to: {self.model_path}")

    def is_configured(self) -> bool:
        """Check if transcriber is configured and model exists."""
        return WHISPER_AVAILABLE and self.model_path.exists()

    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        """Set the event loop for cross-thread task scheduling."""
        self._event_loop = loop
        logger.debug("Event loop set for Whisper transcriber")
        self._start_idle_checker()

    def _start_idle_checker(self):
        """Start background task to check for idle buffer."""
        if self._event_loop and self._event_loop.is_running():
            self._idle_check_task = asyncio.run_coroutine_threadsafe(
                self._idle_check_loop(), self._event_loop
            )
            logger.debug("Whisper idle checker task started")

    async def _idle_check_loop(self):
        """Background loop to check if buffer has been idle too long."""
        while True:
            try:
                await asyncio.sleep(0.5)

                if not self._audio_buffer or self._processing:
                    continue

                current_time = time.time()
                if self._last_audio_time > 0:
                    idle_duration = current_time - self._last_audio_time

                    if (idle_duration >= self.idle_timeout and
                        self._buffer_samples >= self._min_samples):
                        buffer_duration = self._buffer_samples / self.sample_rate
                        logger.info(
                            f"[Whisper] Triggering transcription (idle timeout): "
                            f"{buffer_duration:.1f}s audio"
                        )
                        await self._process_buffer()

            except asyncio.CancelledError:
                logger.debug("Whisper idle checker task cancelled")
                break
            except Exception as e:
                logger.error(f"Whisper idle checker error: {e}")

    def _is_silent(self, audio_data: bytes) -> bool:
        """Check if audio chunk is silent based on RMS threshold."""
        samples = np.frombuffer(audio_data, dtype=np.int16)
        rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2))
        return rms < self.silence_threshold

    def add_audio(self, audio_data: bytes):
        """Add audio data to buffer.

        Triggers transcription when buffer is full or silence is detected.
        """
        self._last_audio_time = time.time()
        self._audio_buffer.append(audio_data)
        num_samples = len(audio_data) // 2
        self._buffer_samples += num_samples

        is_silent = self._is_silent(audio_data)

        if is_silent:
            self._silence_samples += num_samples
        else:
            self._silence_samples = 0
            self._has_speech = True

        # Decide whether to trigger transcription
        should_process = False
        reason = ""

        if self._buffer_samples >= self._samples_per_chunk:
            should_process = True
            reason = "buffer full"
        elif (self._has_speech and
              self._buffer_samples >= self._min_samples and
              self._silence_samples >= self._samples_per_silence):
            should_process = True
            reason = "silence detected"

        if should_process:
            buffer_duration = self._buffer_samples / self.sample_rate
            logger.info(f"[Whisper] Triggering transcription ({reason}): {buffer_duration:.1f}s audio")

            if self._event_loop and self._event_loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self._process_buffer(), self._event_loop
                )
            else:
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self._process_buffer())
                except RuntimeError:
                    logger.warning("No event loop available for Whisper transcription")

    async def _process_buffer(self):
        """Process accumulated audio buffer."""
        if self._processing or not self._audio_buffer:
            return

        self._processing = True

        try:
            audio_bytes = b"".join(self._audio_buffer)
            duration = len(audio_bytes) / (2 * self.sample_rate)
            logger.info(f"[Whisper] Processing audio buffer: {len(audio_bytes)} bytes, {duration:.2f}s")

            # Clear buffer
            self._audio_buffer = []
            self._buffer_samples = 0
            self._silence_samples = 0
            self._has_speech = False

            # Transcribe
            result = await self.transcribe_audio(audio_bytes, duration)

            if result and result.text:
                self._stats["transcriptions"] += 1
                logger.info(f"[Whisper] Transcription: '{result.text}'")

                if self.on_transcription:
                    self.on_transcription(result)

            self._stats["chunks_processed"] += 1
            self._stats["total_duration"] += duration

        except Exception as e:
            logger.error(f"[Whisper] Buffer processing error: {e}")
            self._stats["errors"] += 1
        finally:
            self._processing = False

    async def transcribe_audio(
        self, audio_bytes: bytes, duration: float
    ) -> Optional[TranscriptionResult]:
        """Transcribe audio bytes using Whisper.

        Args:
            audio_bytes: Raw 16-bit PCM audio
            duration: Duration in seconds

        Returns:
            TranscriptionResult or None if failed
        """
        # Lazy load model
        self._lazy_load()

        if not self._initialized:
            logger.warning("Whisper model not initialized")
            return None

        try:
            # Convert bytes to numpy array
            audio = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_float = audio.astype(np.float32) / 32768.0

            # Save audio for debugging if enabled (in background to avoid blocking)
            saved_path = None
            if self.audio_output_dir:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                saved_path = self.audio_output_dir / f"whisper_{timestamp}.wav"

                # Save in background thread to avoid blocking
                def save_audio_file(path, audio_data, sample_rate):
                    wav_buffer = io.BytesIO()
                    with wave.open(wav_buffer, "wb") as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(sample_rate)
                        wav_file.writeframes(audio_data)
                    with open(path, "wb") as f:
                        f.write(wav_buffer.getvalue())

                asyncio.get_event_loop().run_in_executor(
                    None, save_audio_file, saved_path, audio_bytes, self.sample_rate
                )
                logger.debug(f"[Whisper] Saving audio to: {saved_path}")

            # Transcribe with faster-whisper
            # VAD parameters tuned for Hebrew military radio
            vad_params = {
                "threshold": 0.02,  # Speech probability threshold (very sensitive)
                "min_speech_duration_ms": 7,  # Minimum 7ms to count as speech
                "min_silence_duration_ms": 130,  # 130ms silence to split segments
                "speech_pad_ms": 400,  # Add padding around detected speech
                "max_speech_duration_s": 20.0,  # Max continuous speech before forcing split
            }

            if self.model is None:
                logger.error("[Whisper] Model not loaded")
                return None

            # IMPORTANT: model.transcribe returns a generator - we must consume it in the thread
            # to avoid blocking the event loop during iteration
            def do_transcription():
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
                # Consume the generator inside the thread
                segments = []
                full_text = []
                for segment in segments_iter:
                    segments.append({
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text.strip(),
                    })
                    full_text.append(segment.text.strip())
                return segments, full_text

            segments, full_text = await asyncio.to_thread(do_transcription)

            text = " ".join(full_text)

            # Check for commands
            is_command = False
            command_type = None
            for keyword, cmd_type in VOICE_COMMANDS.items():
                if keyword in text:
                    is_command = True
                    command_type = cmd_type
                    logger.info(f"[Whisper] Voice command detected: {keyword} -> {cmd_type}")
                    break

            return TranscriptionResult(
                text=text,
                timestamp=datetime.now(),
                duration_seconds=duration,
                is_command=is_command,
                command_type=command_type,
                segments=segments,
            )

        except Exception as e:
            logger.error(f"[Whisper] Transcription error: {e}")
            self._stats["errors"] += 1
            return None

    async def transcribe_file(self, filepath: str) -> Optional[TranscriptionResult]:
        """Transcribe audio file (WAV format).

        Args:
            filepath: Path to WAV file

        Returns:
            TranscriptionResult or None
        """
        # Lazy load model
        self._lazy_load()

        if not self._initialized:
            logger.warning("Whisper model not initialized, cannot transcribe file")
            return None

        try:
            logger.info(f"[Whisper] Transcribing file: {filepath}")

            # VAD parameters tuned for Hebrew military radio
            vad_params = {
                "threshold": 0.02,  # Speech probability threshold (very sensitive)
                "min_speech_duration_ms": 7,  # Minimum 7ms to count as speech
                "min_silence_duration_ms": 130,  # 130ms silence to split segments
                "speech_pad_ms": 400,  # Add padding around detected speech
                "max_speech_duration_s": 20.0,  # Max continuous speech before forcing split
            }

            if self.model is None:
                logger.error("[Whisper] Model not loaded")
                return None

            # Transcribe directly from file
            # IMPORTANT: model.transcribe returns a generator - we must consume it in the thread
            # to avoid blocking the event loop during iteration
            def do_transcription():
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
                # Consume the generator inside the thread
                segments = []
                full_text = []
                for segment in segments_iter:
                    segments.append({
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text.strip(),
                    })
                    full_text.append(segment.text.strip())
                duration = info.duration if hasattr(info, "duration") else 0.0
                return segments, full_text, duration

            segments, full_text, duration = await asyncio.to_thread(do_transcription)

            text = " ".join(full_text)

            logger.info(f"[Whisper] File transcription: '{text}' ({duration:.2f}s)")

            # Check for commands
            is_command = False
            command_type = None
            for keyword, cmd_type in VOICE_COMMANDS.items():
                if keyword in text:
                    is_command = True
                    command_type = cmd_type
                    logger.info(f"[Whisper] Voice command detected in file: {keyword} -> {cmd_type}")
                    break

            return TranscriptionResult(
                text=text,
                timestamp=datetime.now(),
                duration_seconds=duration,
                is_command=is_command,
                command_type=command_type,
                segments=segments,
            )

        except Exception as e:
            logger.error(f"[Whisper] File transcription error: {e}", exc_info=True)
            self._stats["errors"] += 1
            return None

    def get_stats(self) -> dict:
        """Get transcriber statistics."""
        return {
            **self._stats,
            "configured": self.is_configured(),
            "initialized": self._initialized,
            "device": self.device,
            "compute_type": self.compute_type,
            "model_path": str(self.model_path),
            "buffer_samples": self._buffer_samples,
            "buffer_duration": self._buffer_samples / self.sample_rate if self._buffer_samples else 0,
        }

    def unload(self):
        """Unload model from memory."""
        if self._initialized:
            del self.model
            self.model = None
            self._initialized = False
            logger.info("[Whisper] Model unloaded from memory")
