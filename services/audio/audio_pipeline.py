"""Audio processing pipeline orchestrator.

Coordinates audio transcription, command processing, and TTS response.
Integrates with RTP server for bidirectional audio communication.
"""

import numpy as np
import threading
import queue
import time
import logging
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from datetime import datetime

from .transcriber import HebrewTranscriber
from .command_processor import CommandProcessor, CommandMatch
from .tts import HebrewTTS
from .rtp_server import RTPAudioServer

logger = logging.getLogger(__name__)


@dataclass
class AudioChunk:
    """Represents an audio chunk from RTP stream."""

    session_id: str
    audio: np.ndarray
    sample_rate: int
    timestamp: datetime
    sequence_number: int


@dataclass
class TranscriptionResult:
    """Result from transcription process."""

    session_id: str
    text: str
    language: str
    duration: float
    timestamp: datetime
    command_match: Optional[CommandMatch] = None


class AudioPipeline:
    """Audio processing pipeline for Hebrew voice commands.

    Orchestrates:
    1. Audio reception from RTP
    2. Hebrew transcription
    3. Command processing
    4. TTS response generation
    5. Audio transmission back via RTP
    """

    def __init__(
        self,
        rtp_server: RTPAudioServer,
        transcriber: Optional[HebrewTranscriber] = None,
        command_processor: Optional[CommandProcessor] = None,
        tts: Optional[HebrewTTS] = None,
        enable_auto_response: bool = True,
    ):
        """Initialize audio pipeline.

        Args:
            rtp_server: RTP server instance
            transcriber: Hebrew transcriber (optional, will create if None)
            command_processor: Command processor (optional, will create if None)
            tts: Hebrew TTS (optional, will create if None)
            enable_auto_response: Auto-send TTS responses for commands
        """
        self.rtp_server = rtp_server
        self.transcriber = transcriber or HebrewTranscriber()
        self.command_processor = command_processor or CommandProcessor()
        self.tts = tts or HebrewTTS()
        self.enable_auto_response = enable_auto_response

        # Processing queue
        self._audio_queue: queue.Queue = queue.Queue(maxsize=100)
        self._result_queue: queue.Queue = queue.Queue(maxsize=100)

        # Worker threads
        self._transcription_thread: Optional[threading.Thread] = None
        self._processing_thread: Optional[threading.Thread] = None

        # State
        self._running = False
        self._transcription_callbacks: list[Callable] = []
        self._command_callbacks: list[Callable] = []

        # Statistics
        self._stats = {
            "chunks_processed": 0,
            "transcriptions": 0,
            "commands_detected": 0,
            "responses_sent": 0,
        }

        logger.info("AudioPipeline initialized")

    def start(self):
        """Start audio processing pipeline."""
        if self._running:
            logger.warning("Pipeline already running")
            return

        logger.info("Starting audio processing pipeline...")
        self._running = True

        # Start worker threads
        self._transcription_thread = threading.Thread(
            target=self._transcription_worker,
            daemon=True,
            name="AudioPipeline-Transcription",
        )
        self._transcription_thread.start()

        self._processing_thread = threading.Thread(
            target=self._processing_worker, daemon=True, name="AudioPipeline-Processing"
        )
        self._processing_thread.start()

        logger.info("Audio processing pipeline started")

    def stop(self):
        """Stop audio processing pipeline."""
        if not self._running:
            return

        logger.info("Stopping audio processing pipeline...")
        self._running = False

        # Wait for threads
        if self._transcription_thread:
            self._transcription_thread.join(timeout=2.0)
        if self._processing_thread:
            self._processing_thread.join(timeout=2.0)

        logger.info("Audio processing pipeline stopped")

    def process_audio_chunk(
        self,
        session_id: str,
        audio: np.ndarray,
        sample_rate: int = 8000,
        sequence_number: int = 0,
    ):
        """Queue audio chunk for processing.

        Args:
            session_id: RTP session identifier
            audio: Audio samples (int16 or float32)
            sample_rate: Sample rate in Hz
            sequence_number: RTP sequence number
        """
        chunk = AudioChunk(
            session_id=session_id,
            audio=audio,
            sample_rate=sample_rate,
            timestamp=datetime.now(),
            sequence_number=sequence_number,
        )

        try:
            self._audio_queue.put(chunk, timeout=0.1)
            self._stats["chunks_processed"] += 1
        except queue.Full:
            logger.warning(
                f"Audio queue full, dropping chunk from session {session_id}"
            )

    def send_text_response(self, session_id: str, text: str) -> bool:
        """Generate speech from text and send to session.

        Args:
            session_id: RTP session identifier
            text: Hebrew text to synthesize and send

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Generating TTS response for session {session_id}: '{text}'")

        # Synthesize speech
        audio = self.tts.synthesize(text)
        if audio is None:
            logger.error("TTS synthesis failed")
            return False

        # Send via RTP
        success = self.rtp_server.send_audio_to_session(
            session_id, audio, sample_rate=self.tts.sample_rate
        )

        if success:
            self._stats["responses_sent"] += 1

        return success

    def send_audio_response(
        self, session_id: str, audio: np.ndarray, sample_rate: int = 16000
    ) -> bool:
        """Send pre-recorded audio to session.

        Args:
            session_id: RTP session identifier
            audio: Audio samples (int16 or float32)
            sample_rate: Sample rate in Hz

        Returns:
            True if successful, False otherwise
        """
        logger.info(
            f"Sending audio response to session {session_id} ({len(audio)} samples)"
        )

        success = self.rtp_server.send_audio_to_session(session_id, audio, sample_rate)

        if success:
            self._stats["responses_sent"] += 1

        return success

    def register_transcription_callback(self, callback: Callable):
        """Register callback for transcription results.

        Args:
            callback: Function(TranscriptionResult) -> None
        """
        self._transcription_callbacks.append(callback)
        logger.debug(f"Registered transcription callback: {callback.__name__}")

    def register_command_callback(self, callback: Callable):
        """Register callback for detected commands.

        Args:
            callback: Function(CommandMatch) -> None
        """
        self._command_callbacks.append(callback)
        logger.debug(f"Registered command callback: {callback.__name__}")

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics.

        Returns:
            Statistics dictionary
        """
        return {
            **self._stats,
            "running": self._running,
            "queue_size": self._audio_queue.qsize(),
            "transcriber_ready": self.transcriber.is_ready(),
            "tts_ready": self.tts.is_ready(),
        }

    # ========================================================================
    # WORKER THREADS
    # ========================================================================

    def _transcription_worker(self):
        """Worker thread for audio transcription."""
        logger.info("Transcription worker started")

        while self._running:
            try:
                # Get audio chunk from queue
                chunk = self._audio_queue.get(timeout=0.5)

                # Transcribe
                result = self.transcriber.transcribe(
                    chunk.audio, chunk.sample_rate, language="he"
                )

                if result["text"]:
                    logger.info(
                        f"Transcribed from session {chunk.session_id}: '{result['text']}'"
                    )
                    self._stats["transcriptions"] += 1

                    # Create result
                    transcription_result = TranscriptionResult(
                        session_id=chunk.session_id,
                        text=result["text"],
                        language=result["language"],
                        duration=result["duration"],
                        timestamp=chunk.timestamp,
                    )

                    # Queue for processing
                    self._result_queue.put(transcription_result)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Transcription worker error: {e}", exc_info=True)

        logger.info("Transcription worker stopped")

    def _processing_worker(self):
        """Worker thread for command processing and response."""
        logger.info("Processing worker started")

        while self._running:
            try:
                # Get transcription result
                result = self._result_queue.get(timeout=0.5)

                # Notify transcription callbacks
                for callback in self._transcription_callbacks:
                    try:
                        callback(result)
                    except Exception as e:
                        logger.error(f"Transcription callback error: {e}")

                # Process for commands
                command_match = self.command_processor.process(
                    result.text, context={"session_id": result.session_id}
                )

                if command_match:
                    logger.info(f"Command detected: {command_match.command_id}")
                    self._stats["commands_detected"] += 1
                    result.command_match = command_match

                    # Notify command callbacks
                    for callback in self._command_callbacks:
                        try:
                            callback(command_match)
                        except Exception as e:
                            logger.error(f"Command callback error: {e}")

                    # Auto-send response if enabled
                    if self.enable_auto_response:
                        response_text = self._generate_response(command_match)
                        if response_text:
                            self.send_text_response(result.session_id, response_text)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Processing worker error: {e}", exc_info=True)

        logger.info("Processing worker stopped")

    def _generate_response(self, command_match: CommandMatch) -> Optional[str]:
        """Generate Hebrew response text for command.

        Args:
            command_match: Detected command

        Returns:
            Hebrew response text or None
        """
        # Default responses in Hebrew
        responses = {
            "track_person": "מתחיל מעקב אחרי אדם",
            "track_car": "מתחיל מעקב אחרי רכב",
            "zoom_in": "מקרב תמונה",
            "zoom_out": "מרחיק תמונה",
            "pan_left": "זז שמאלה",
            "pan_right": "זז ימינה",
            "status_report": "המערכת פועלת כרגיל",
            "list_tracks": "אין מעקבים פעילים כרגע",
            "stop": "עוצר",
            "start": "מתחיל",
        }

        return responses.get(command_match.command_id)
