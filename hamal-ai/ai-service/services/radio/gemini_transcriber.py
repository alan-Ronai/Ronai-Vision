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
import shutil
from pathlib import Path
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
        silence_threshold: float = 500.0,  # RMS threshold for silence
        silence_duration: float = 1.5,  # Seconds of silence to trigger processing
        min_duration: float = 1.5,  # Minimum audio duration before processing
        idle_timeout: float = 2.0,  # Seconds of no audio before processing
        save_audio: bool = True,  # Save audio files for debugging
        on_transcription: Optional[Callable[[TranscriptionResult], None]] = None,
    ):
        """Initialize Gemini transcriber.

        Args:
            api_key: Gemini API key (or use GEMINI_API_KEY env var)
            model: Gemini model to use
            sample_rate: Audio sample rate in Hz
            chunk_duration: Seconds of audio to accumulate before transcribing
            silence_threshold: RMS threshold below which audio is considered silent
            silence_duration: Seconds of silence after speech to trigger early processing
            min_duration: Minimum seconds of audio required before processing
            idle_timeout: Seconds of no new audio before processing buffer
            save_audio: Whether to save audio files to audio_output directory
            on_transcription: Callback when transcription is ready
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.min_duration = min_duration
        self.idle_timeout = idle_timeout
        self.save_audio = save_audio
        self.on_transcription = on_transcription

        # Configure Gemini
        self.model = None
        if GENAI_AVAILABLE and self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            logger.info(f"Gemini transcriber initialized with model: {model}")
        else:
            if not GENAI_AVAILABLE:
                logger.warning(
                    "google-generativeai not available - transcription disabled"
                )
            else:
                logger.warning("No Gemini API key - transcription disabled")

        # Audio output directory for debugging
        if self.save_audio:
            self.audio_output_dir = Path("audio_output")
            self.audio_output_dir.mkdir(exist_ok=True)
            logger.info(f"Audio files will be saved to: {self.audio_output_dir.absolute()}")
        else:
            self.audio_output_dir = None
            logger.info("Audio file saving disabled")

        # Audio buffer
        self._audio_buffer: List[bytes] = []
        self._buffer_samples = 0
        self._samples_per_chunk = int(sample_rate * chunk_duration)

        # Silence detection state
        self._silence_samples = 0  # Consecutive silent samples
        self._samples_per_silence = int(sample_rate * silence_duration)
        self._min_samples = int(sample_rate * min_duration)
        self._has_speech = False  # Whether we've detected speech in current buffer

        # Idle timeout state (for when RTP stream stops)
        self._last_audio_time = 0.0  # Last time add_audio() was called
        self._idle_check_task: Optional[asyncio.Task] = None

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

    def is_configured(self) -> bool:
        """Check if transcriber is configured."""
        return self.model is not None

    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        """Set the event loop for cross-thread task scheduling."""
        self._event_loop = loop
        logger.debug("Event loop set for transcriber")

        # Start idle timeout checker
        self._start_idle_checker()

    def _start_idle_checker(self):
        """Start background task to check for idle buffer."""
        if self._event_loop and self._event_loop.is_running():
            self._idle_check_task = asyncio.run_coroutine_threadsafe(
                self._idle_check_loop(), self._event_loop
            )
            logger.debug("Idle checker task started")

    async def _idle_check_loop(self):
        """Background loop to check if buffer has been idle too long."""
        while True:
            try:
                await asyncio.sleep(0.5)  # Check every 500ms

                # Skip if no buffer or already processing
                if not self._audio_buffer or self._processing:
                    continue

                # Check if we have audio and it's been idle
                current_time = time.time()
                if self._last_audio_time > 0:
                    idle_duration = current_time - self._last_audio_time

                    # If idle too long and we have minimum audio
                    if (
                        idle_duration >= self.idle_timeout
                        and self._buffer_samples >= self._min_samples
                    ):
                        buffer_duration = self._buffer_samples / self.sample_rate
                        logger.info(
                            f"🔔 Triggering transcription (idle timeout): "
                            f"{buffer_duration:.1f}s audio, "
                            f"{idle_duration:.1f}s idle"
                        )
                        await self._process_buffer()

            except asyncio.CancelledError:
                logger.debug("Idle checker task cancelled")
                break
            except Exception as e:
                logger.error(f"Idle checker error: {e}", exc_info=True)

    def _is_silent(self, audio_data: bytes) -> bool:
        """Check if audio chunk is silent based on RMS threshold.

        Args:
            audio_data: Raw 16-bit PCM audio bytes

        Returns:
            True if audio is below silence threshold
        """
        # Convert bytes to numpy array
        samples = np.frombuffer(audio_data, dtype=np.int16)

        # Calculate RMS (Root Mean Square) - measure of audio level
        rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2))

        return rms < self.silence_threshold

    def add_audio(self, audio_data: bytes):
        """Add audio data to buffer.

        Triggers transcription when:
        1. Buffer reaches chunk_duration (max wait), OR
        2. Silence detected after speech (early processing), OR
        3. No new audio for idle_timeout (handles PTT release)

        Args:
            audio_data: Raw 16-bit PCM audio bytes
        """
        # Update last audio time for idle detection
        self._last_audio_time = time.time()

        self._audio_buffer.append(audio_data)
        num_samples = len(audio_data) // 2  # 2 bytes per sample
        self._buffer_samples += num_samples

        # Check if this chunk is silent
        is_silent = self._is_silent(audio_data)

        if is_silent:
            self._silence_samples += num_samples
        else:
            # Non-silent audio - reset silence counter and mark that we have speech
            self._silence_samples = 0
            self._has_speech = True

        # Log buffer progress every 10 chunks
        if len(self._audio_buffer) % 10 == 1:
            buffer_duration = self._buffer_samples / self.sample_rate
            target_duration = self.chunk_duration
            progress = (buffer_duration / target_duration) * 100
            silence_duration = self._silence_samples / self.sample_rate
            logger.debug(
                f"📊 Buffer: {len(self._audio_buffer)} chunks, "
                f"{buffer_duration:.2f}s / {target_duration:.1f}s ({progress:.0f}%), "
                f"silence: {silence_duration:.1f}s, speech: {self._has_speech}"
            )

        # Decide whether to trigger transcription
        should_process = False
        reason = ""

        # Condition 1: Buffer is full (reached max duration)
        if self._buffer_samples >= self._samples_per_chunk:
            should_process = True
            reason = "buffer full"

        # Condition 2: Silence after speech (early processing)
        elif (
            self._has_speech  # We have speech in the buffer
            and self._buffer_samples >= self._min_samples  # Met minimum duration
            and self._silence_samples >= self._samples_per_silence  # Enough silence detected
        ):
            should_process = True
            reason = "silence detected"

        if should_process:
            buffer_duration = self._buffer_samples / self.sample_rate
            logger.info(
                f"🔔 Triggering transcription ({reason}): "
                f"{buffer_duration:.1f}s audio, "
                f"{self._silence_samples / self.sample_rate:.1f}s silence"
            )

            # Trigger async transcription - use stored event loop for cross-thread scheduling
            if self._event_loop and self._event_loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self._process_buffer(), self._event_loop
                )
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
            audio_bytes = b"".join(self._audio_buffer)
            duration = len(audio_bytes) / (2 * self.sample_rate)
            logger.info(f"🎙️  Processing audio buffer: {len(audio_bytes)} bytes, {duration:.2f}s")

            # Clear buffer and reset silence detection state
            self._audio_buffer = []
            self._buffer_samples = 0
            self._silence_samples = 0
            self._has_speech = False

            # Transcribe
            result = await self.transcribe_audio(audio_bytes, duration)

            if result and result.text:
                self._stats["transcriptions"] += 1
                logger.info(f"✅ Transcription successful: '{result.text}' ({len(result.text)} chars)")

                # Call callback
                if self.on_transcription:
                    self.on_transcription(result)
            else:
                logger.warning(f"⚠️  Transcription returned empty or None")

            self._stats["chunks_processed"] += 1
            self._stats["total_duration"] += duration

        except Exception as e:
            logger.error(f"Buffer processing error: {e}", exc_info=True)
            self._stats["errors"] += 1
        finally:
            self._processing = False

    async def transcribe_audio(
        self, audio_bytes: bytes, duration: float
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
            logger.debug(f"Creating WAV file from {len(audio_bytes)} bytes")
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_bytes)

            wav_buffer.seek(0)
            wav_data = wav_buffer.read()
            logger.debug(f"WAV file created: {len(wav_data)} bytes")

            # Create temp file for Gemini (it needs a file path)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(wav_data)
                tmp_path = tmp.name

            try:
                # Save to audio_output directory for debugging (if enabled)
                saved_path = None
                if self.audio_output_dir:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    saved_path = self.audio_output_dir / f"audio_{timestamp}.wav"
                    shutil.copy2(tmp_path, saved_path)
                    logger.info(f"💾 Saved audio to: {saved_path}")

                # Upload to Gemini
                logger.debug(f"Uploading audio to Gemini: {tmp_path}")
                audio_file = genai.upload_file(tmp_path, mime_type="audio/wav")
                logger.debug(f"Upload successful, file: {audio_file.name}")

                # Transcribe with Hebrew prompt
                prompt = """תמלל את האודיו הזה לעברית.

כללים:
- תמלל בדיוק מה שנאמר בעברית
- אם יש רעשי רקע או דיבור לא ברור, התעלם
- אם אין דיבור ברור, החזר מחרוזת ריקה
- אל תוסיף הסברים או הערות
- החזר רק את הטקסט המתומלל"""

                logger.debug(f"Sending to Gemini model: {self.model_name}")
                response = await asyncio.to_thread(
                    self.model.generate_content, [prompt, audio_file]
                )
                logger.debug(f"Gemini response received")

                # Clean up uploaded file from Gemini
                try:
                    genai.delete_file(audio_file.name)
                except Exception as del_err:
                    logger.debug(f"Could not delete uploaded file: {del_err}")

                # Check if response has candidates
                if not response.candidates or len(response.candidates) == 0:
                    warning_msg = "⚠️  Gemini returned empty candidates - likely no speech detected or audio too noisy"
                    if saved_path:
                        warning_msg += f"\n   Audio saved to: {saved_path}\n   You can play this file to debug what Gemini received"
                    logger.warning(warning_msg)
                    return TranscriptionResult(
                        text="",
                        timestamp=datetime.now(),
                        duration_seconds=duration,
                        is_command=False,
                        command_type=None,
                    )

                # Check if candidate has content
                if not response.candidates[0].content or not response.candidates[0].content.parts:
                    warning_msg = "⚠️  Gemini candidate has no content parts - likely no speech detected"
                    if saved_path:
                        warning_msg += f"\n   Audio saved to: {saved_path}\n   You can play this file to debug what Gemini received"
                    logger.warning(warning_msg)
                    return TranscriptionResult(
                        text="",
                        timestamp=datetime.now(),
                        duration_seconds=duration,
                        is_command=False,
                        command_type=None,
                    )

                text = response.text.strip() if response.text else ""
                logger.debug(f"Extracted text: '{text}'")

                # Log where the audio was saved (if enabled)
                if saved_path:
                    if text:
                        logger.info(f"📝 Transcribed: '{text}' (audio: {saved_path.name})")
                    else:
                        logger.info(f"📝 Empty transcription (audio: {saved_path.name})")
                else:
                    if text:
                        logger.info(f"📝 Transcribed: '{text}'")
                    else:
                        logger.info(f"📝 Empty transcription")

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
                    command_type=command_type,
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
            with open(filepath, "rb") as f:
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
            "buffer_duration": self._buffer_samples / self.sample_rate
            if self._buffer_samples
            else 0,
        }


class StreamingGeminiTranscriber(GeminiTranscriber):
    """Streaming transcriber with silence detection for lower latency."""

    def __init__(
        self,
        chunk_duration: float = 3.0,  # Max duration before forcing transcription
        overlap_duration: float = 0.5,  # Overlap to avoid cutting words
        **kwargs,
    ):
        """Initialize streaming transcriber with silence detection.

        The StreamingGeminiTranscriber uses the same silence detection as the base
        class but with a shorter max chunk_duration for faster response.

        Args:
            chunk_duration: Maximum seconds before forcing transcription
            overlap_duration: Seconds of overlap between chunks (to avoid cutting words)
            **kwargs: Passed to GeminiTranscriber (includes silence_threshold, etc.)
        """
        super().__init__(chunk_duration=chunk_duration, **kwargs)
        self.overlap_duration = overlap_duration
        self._overlap_samples = int(self.sample_rate * overlap_duration)
        self._previous_overlap: bytes = b""

    async def _process_buffer(self):
        """Process buffer with overlap for continuity."""
        if self._processing or not self._audio_buffer:
            return

        self._processing = True

        try:
            # Combine buffer with previous overlap
            audio_bytes = self._previous_overlap + b"".join(self._audio_buffer)
            duration = len(audio_bytes) / (2 * self.sample_rate)

            # Save overlap for next chunk
            overlap_bytes = self._overlap_samples * 2
            if len(audio_bytes) > overlap_bytes:
                self._previous_overlap = audio_bytes[-overlap_bytes:]

            # Clear buffer and reset silence detection state
            self._audio_buffer = []
            self._buffer_samples = 0
            self._silence_samples = 0
            self._has_speech = False

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
