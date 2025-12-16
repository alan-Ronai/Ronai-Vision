"""Gemini-based Hebrew audio transcription.

Uses Google's Gemini API to transcribe Hebrew audio in near real-time.
Processes audio chunks and returns Hebrew text.
"""

import os
import io
import wave
import asyncio
import logging
import tempfile
import time
from typing import Optional, Callable, List
from dataclasses import dataclass
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

# Try to import google.generativeai
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    logger.warning("google-generativeai not installed. Transcription will be disabled.")


@dataclass
class TranscriptionResult:
    """Result of transcription."""
    text: str
    timestamp: datetime
    duration_seconds: float
    confidence: Optional[float] = None
    is_command: bool = False
    command_type: Optional[str] = None


# Hebrew voice commands to detect
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


class GeminiTranscriber:
    """Transcribes Hebrew audio using Gemini API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.0-flash-exp",
        sample_rate: int = 16000,
        chunk_duration: float = 5.0,  # Seconds of audio per transcription
        on_transcription: Optional[Callable[[TranscriptionResult], None]] = None
    ):
        """Initialize Gemini transcriber.

        Args:
            api_key: Gemini API key (or use GEMINI_API_KEY env var)
            model: Gemini model to use
            sample_rate: Audio sample rate in Hz
            chunk_duration: Seconds of audio to accumulate before transcribing
            on_transcription: Callback when transcription is ready
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.on_transcription = on_transcription

        # Configure Gemini
        self.model = None
        if GENAI_AVAILABLE and self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            logger.info(f"Gemini transcriber initialized with model: {model}")
        else:
            if not GENAI_AVAILABLE:
                logger.warning("google-generativeai not available - transcription disabled")
            else:
                logger.warning("No Gemini API key - transcription disabled")

        # Audio buffer
        self._audio_buffer: List[bytes] = []
        self._buffer_samples = 0
        self._samples_per_chunk = int(sample_rate * chunk_duration)

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
            "errors": 0
        }

    def is_configured(self) -> bool:
        """Check if transcriber is configured."""
        return self.model is not None

    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        """Set the event loop for cross-thread task scheduling."""
        self._event_loop = loop
        logger.debug("Event loop set for transcriber")

    def add_audio(self, audio_data: bytes):
        """Add audio data to buffer.

        When buffer reaches chunk_duration, triggers transcription.

        Args:
            audio_data: Raw 16-bit PCM audio bytes
        """
        self._audio_buffer.append(audio_data)
        self._buffer_samples += len(audio_data) // 2  # 2 bytes per sample

        # Check if we have enough audio
        if self._buffer_samples >= self._samples_per_chunk:
            # Trigger async transcription - use stored event loop for cross-thread scheduling
            if self._event_loop and self._event_loop.is_running():
                asyncio.run_coroutine_threadsafe(self._process_buffer(), self._event_loop)
            else:
                # Try to get current loop
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self._process_buffer())
                except RuntimeError:
                    logger.warning("No event loop available for transcription")

    async def _process_buffer(self):
        """Process accumulated audio buffer."""
        if self._processing or not self._audio_buffer:
            return

        self._processing = True

        try:
            # Combine buffer
            audio_bytes = b''.join(self._audio_buffer)
            duration = len(audio_bytes) / (2 * self.sample_rate)

            # Clear buffer
            self._audio_buffer = []
            self._buffer_samples = 0

            # Transcribe
            result = await self.transcribe_audio(audio_bytes, duration)

            if result and result.text:
                self._stats["transcriptions"] += 1

                # Call callback
                if self.on_transcription:
                    self.on_transcription(result)

            self._stats["chunks_processed"] += 1
            self._stats["total_duration"] += duration

        except Exception as e:
            logger.error(f"Buffer processing error: {e}")
            self._stats["errors"] += 1
        finally:
            self._processing = False

    async def transcribe_audio(
        self,
        audio_bytes: bytes,
        duration: float
    ) -> Optional[TranscriptionResult]:
        """Transcribe audio bytes using Gemini.

        Args:
            audio_bytes: Raw 16-bit PCM audio
            duration: Duration in seconds

        Returns:
            TranscriptionResult or None if failed
        """
        if not self.model:
            return None

        try:
            # Create WAV file in memory
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_bytes)

            wav_buffer.seek(0)
            wav_data = wav_buffer.read()

            # Create temp file for Gemini (it needs a file path)
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp.write(wav_data)
                tmp_path = tmp.name

            try:
                # Upload to Gemini
                audio_file = genai.upload_file(tmp_path, mime_type="audio/wav")

                # Transcribe with Hebrew prompt
                prompt = """תמלל את האודיו הזה לעברית.

כללים:
- תמלל בדיוק מה שנאמר בעברית
- אם יש רעשי רקע או דיבור לא ברור, התעלם
- אם אין דיבור ברור, החזר מחרוזת ריקה
- אל תוסיף הסברים או הערות
- החזר רק את הטקסט המתומלל"""

                response = await asyncio.to_thread(
                    self.model.generate_content,
                    [prompt, audio_file]
                )

                # Clean up uploaded file
                try:
                    genai.delete_file(audio_file.name)
                except:
                    pass

                text = response.text.strip() if response.text else ""

                # Check for commands
                is_command = False
                command_type = None

                for keyword, cmd_type in VOICE_COMMANDS.items():
                    if keyword in text:
                        is_command = True
                        command_type = cmd_type
                        logger.info(f"Voice command detected: {keyword} -> {cmd_type}")
                        break

                return TranscriptionResult(
                    text=text,
                    timestamp=datetime.now(),
                    duration_seconds=duration,
                    is_command=is_command,
                    command_type=command_type
                )

            finally:
                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                except:
                    pass

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            self._stats["errors"] += 1
            return None

    async def transcribe_file(self, filepath: str) -> Optional[TranscriptionResult]:
        """Transcribe audio file.

        Args:
            filepath: Path to WAV/PCM file

        Returns:
            TranscriptionResult or None
        """
        if not self.model:
            return None

        try:
            # Read file
            with open(filepath, 'rb') as f:
                audio_bytes = f.read()

            # Calculate duration
            duration = len(audio_bytes) / (2 * self.sample_rate)

            return await self.transcribe_audio(audio_bytes, duration)

        except Exception as e:
            logger.error(f"File transcription error: {e}")
            return None

    def get_stats(self) -> dict:
        """Get transcriber statistics."""
        return {
            **self._stats,
            "configured": self.is_configured(),
            "buffer_samples": self._buffer_samples,
            "buffer_duration": self._buffer_samples / self.sample_rate if self._buffer_samples else 0
        }


class StreamingGeminiTranscriber(GeminiTranscriber):
    """Streaming transcriber with shorter chunks for lower latency."""

    def __init__(
        self,
        chunk_duration: float = 3.0,  # Shorter chunks for faster response
        overlap_duration: float = 0.5,  # Overlap to avoid cutting words
        **kwargs
    ):
        super().__init__(chunk_duration=chunk_duration, **kwargs)
        self.overlap_duration = overlap_duration
        self._overlap_samples = int(self.sample_rate * overlap_duration)
        self._previous_overlap: bytes = b''

    async def _process_buffer(self):
        """Process buffer with overlap for continuity."""
        if self._processing or not self._audio_buffer:
            return

        self._processing = True

        try:
            # Combine buffer with previous overlap
            audio_bytes = self._previous_overlap + b''.join(self._audio_buffer)
            duration = len(audio_bytes) / (2 * self.sample_rate)

            # Save overlap for next chunk
            overlap_bytes = self._overlap_samples * 2
            if len(audio_bytes) > overlap_bytes:
                self._previous_overlap = audio_bytes[-overlap_bytes:]

            # Clear buffer
            self._audio_buffer = []
            self._buffer_samples = 0

            # Transcribe
            result = await self.transcribe_audio(audio_bytes, duration)

            if result and result.text:
                self._stats["transcriptions"] += 1
                if self.on_transcription:
                    self.on_transcription(result)

            self._stats["chunks_processed"] += 1
            self._stats["total_duration"] += duration

        except Exception as e:
            logger.error(f"Streaming buffer error: {e}")
            self._stats["errors"] += 1
        finally:
            self._processing = False
