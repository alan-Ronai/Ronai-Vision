"""Radio Service - Receives RTP audio via EC2 relay and transcribes to Hebrew.

Integrates RTPTCPClient with both GeminiTranscriber and WhisperTranscriber,
sending transcriptions to the backend for display in UI.

Also processes transcription events through the Rule Engine for
transcription_keyword conditions.

Architecture:
    Radio → RTP/UDP → EC2 Relay → TCP → This Service → Gemini  → Backend → UI (Tab 2)
                                                     → Whisper → Backend → UI (Tab 1)
                                                      ↓
                                                Rule Engine → Actions
"""

import os
import asyncio
import logging
from typing import Optional, Callable
from datetime import datetime

import httpx

from .rtp_tcp_client import RTPTCPClient
from .gemini_transcriber import StreamingGeminiTranscriber, TranscriptionResult
from .whisper_transcriber import WhisperTranscriber
from ..rules import get_rule_engine, RuleContext

logger = logging.getLogger(__name__)


class RadioService:
    """Main radio service orchestrating RTP reception via EC2 relay and transcription."""

    def __init__(
        self,
        ec2_host: Optional[str] = None,
        ec2_port: Optional[int] = None,
        sample_rate: int = 16000,
        backend_url: str = "http://localhost:3000",
        chunk_duration: float = 3.0,
        silence_threshold: float = 500.0,
        silence_duration: float = 1.5,
        min_duration: float = 1.5,
        idle_timeout: float = 2.0,
        save_audio: bool = True,
        use_vad: bool = False,
        whisper_model_path: str = "models/whisper-large-v3-hebrew-ct2",
        on_transcription: Optional[Callable[[TranscriptionResult], None]] = None
    ):
        """Initialize radio service.

        Args:
            ec2_host: EC2 relay server hostname (default from EC2_RTP_HOST env)
            ec2_port: EC2 relay TCP port (default from EC2_RTP_PORT env)
            sample_rate: Audio sample rate
            backend_url: Backend API URL
            chunk_duration: Maximum audio chunk duration before forcing transcription
            silence_threshold: RMS threshold for silence detection
            silence_duration: Seconds of silence to trigger early processing
            min_duration: Minimum audio duration before processing
            idle_timeout: Seconds of no audio before processing buffer (handles PTT release)
            save_audio: Whether to save audio files for debugging
            use_vad: Whether to use Voice Activity Detection for speaker segmentation
            whisper_model_path: Path to Whisper CT2 model directory
            on_transcription: Optional callback for transcriptions
        """
        # Get EC2 relay config from environment or parameters
        self.ec2_host = ec2_host or os.getenv("EC2_RTP_HOST", "localhost")
        self.ec2_port = ec2_port or int(os.getenv("EC2_RTP_PORT", "5005"))
        self.sample_rate = sample_rate
        self.backend_url = backend_url
        self.on_transcription = on_transcription

        # HTTP client for backend
        self._http_client: Optional[httpx.AsyncClient] = None

        # Audio statistics
        self._audio_stats = {
            "chunks_received": 0,
            "bytes_received": 0,
            "last_chunk_time": None,
        }

        # Initialize Gemini transcriber (Tab 2 - מתמלל 2)
        self.gemini_transcriber = StreamingGeminiTranscriber(
            sample_rate=sample_rate,
            chunk_duration=chunk_duration,
            silence_threshold=silence_threshold,
            silence_duration=silence_duration,
            min_duration=min_duration,
            idle_timeout=idle_timeout,
            save_audio=save_audio,
            use_vad=use_vad,
            on_transcription=self._handle_gemini_transcription
        )

        # Initialize Whisper transcriber (Tab 1 - מתמלל 1)
        self.whisper_transcriber = WhisperTranscriber(
            model_path=whisper_model_path,
            device="auto",
            compute_type="int8",
            sample_rate=sample_rate,
            chunk_duration=chunk_duration,
            silence_threshold=silence_threshold,
            silence_duration=silence_duration,
            min_duration=min_duration,
            idle_timeout=idle_timeout,
            save_audio=save_audio,
            on_transcription=self._handle_whisper_transcription
        )

        # Keep reference to main transcriber for backwards compatibility
        self.transcriber = self.gemini_transcriber

        # Initialize TCP client for EC2 relay
        self.tcp_client = RTPTCPClient(
            host=self.ec2_host,
            port=self.ec2_port,
            target_sample_rate=sample_rate,
            audio_callback=self._on_audio_received
        )

        self._running = False

        # Diagnostic logging
        logger.info("=" * 60)
        logger.info("RadioService Configuration (Dual Transcriber):")
        logger.info(f"  EC2 Relay: {self.ec2_host}:{self.ec2_port}")
        logger.info(f"  Sample Rate: {sample_rate} Hz")
        logger.info(f"  Chunk Duration: {chunk_duration}s (max)")
        logger.info(f"  Silence Detection: threshold={silence_threshold}, duration={silence_duration}s")
        logger.info(f"  Idle Timeout: {idle_timeout}s")
        logger.info(f"  Minimum Duration: {min_duration}s")
        logger.info(f"  Save Audio Files: {save_audio}")
        logger.info(f"  VAD Enabled: {use_vad}")
        logger.info(f"  Backend URL: {backend_url}")
        logger.info(f"  Gemini Transcriber: {'configured' if self.gemini_transcriber.is_configured() else 'NOT configured'}")
        logger.info(f"  Whisper Transcriber: {'configured' if self.whisper_transcriber.is_configured() else 'NOT configured'}")
        logger.info(f"  Whisper Model Path: {whisper_model_path}")
        logger.info("=" * 60)

    async def start(self):
        """Start radio service."""
        if self._running:
            return

        self._running = True
        self._http_client = httpx.AsyncClient(timeout=10.0)

        # Set event loop for both transcribers (for cross-thread async calls)
        try:
            loop = asyncio.get_running_loop()
            self.gemini_transcriber.set_event_loop(loop)
            self.whisper_transcriber.set_event_loop(loop)
        except RuntimeError:
            logger.warning("Could not get running loop for transcribers")

        # Start TCP client
        self.tcp_client.start()

        logger.info("Radio service started (dual transcriber mode)")

    async def stop(self):
        """Stop radio service."""
        if not self._running:
            return

        self._running = False

        # Stop TCP client
        self.tcp_client.stop()

        # Close HTTP client
        if self._http_client:
            await self._http_client.aclose()

        logger.info("Radio service stopped")

    def _on_audio_received(self, audio_data: bytes, sample_rate: int):
        """Callback when audio is received from TCP client.

        Feeds audio to BOTH transcribers.

        Args:
            audio_data: Raw PCM audio bytes
            sample_rate: Sample rate
        """
        # Update stats
        self._audio_stats["chunks_received"] += 1
        self._audio_stats["bytes_received"] += len(audio_data)
        self._audio_stats["last_chunk_time"] = datetime.now()

        # Log every 10th chunk to confirm audio flow
        if self._audio_stats["chunks_received"] % 10 == 1:
            duration_ms = (len(audio_data) / 2 / sample_rate) * 1000
            logger.info(
                f"🎤 Audio received: {len(audio_data)} bytes, "
                f"{duration_ms:.1f}ms @ {sample_rate}Hz "
                f"(chunk #{self._audio_stats['chunks_received']})"
            )

        # Feed to BOTH transcribers
        try:
            self.gemini_transcriber.add_audio(audio_data)
        except Exception as e:
            logger.error(f"Failed to add audio to Gemini transcriber: {e}", exc_info=True)

        try:
            self.whisper_transcriber.add_audio(audio_data)
        except Exception as e:
            logger.error(f"Failed to add audio to Whisper transcriber: {e}", exc_info=True)

    def _handle_gemini_transcription(self, result: TranscriptionResult):
        """Handle Gemini transcription result.

        Args:
            result: Transcription result
        """
        if not result.text:
            return

        logger.info(f"[Gemini] Transcription: {result.text}")

        # Send to backend with gemini source
        asyncio.create_task(self._send_to_backend(result, transcriber="gemini"))

        # Process through rule engine for transcription_keyword rules
        asyncio.create_task(self._process_transcription_rules(result))

        # Call external callback
        if self.on_transcription:
            self.on_transcription(result)

    def _handle_whisper_transcription(self, result: TranscriptionResult):
        """Handle Whisper transcription result.

        Args:
            result: Transcription result
        """
        if not result.text:
            return

        logger.info(f"[Whisper] Transcription: {result.text}")

        # Send to backend with whisper source
        asyncio.create_task(self._send_to_backend(result, transcriber="whisper"))

        # Process through rule engine for transcription_keyword rules
        asyncio.create_task(self._process_transcription_rules(result))

        # Call external callback
        if self.on_transcription:
            self.on_transcription(result)

    async def _process_transcription_rules(self, result: TranscriptionResult):
        """Process transcription through rule engine.

        This allows rules with transcription_keyword conditions to be triggered
        independently of the detection loop.

        Args:
            result: Transcription result
        """
        try:
            rule_engine = get_rule_engine()
            if not rule_engine:
                logger.debug("Rule engine not available for transcription processing")
                return

            # Create context for transcription event
            context = RuleContext(
                event_type="transcription",
                transcription=result.text,
            )
            # Set timestamp if available
            if result.timestamp:
                context.timestamp = result.timestamp.timestamp()

            # Process through rule engine
            results = await rule_engine.process_event(context)

            if results:
                logger.info(f"[RadioService] Transcription triggered {len(results)} rule(s)")
                for r in results:
                    logger.debug(f"  - Rule '{r.get('rule_name')}': {r.get('actions_executed')} actions")

        except Exception as e:
            logger.error(f"Error processing transcription rules: {e}")

    async def _send_to_backend(self, result: TranscriptionResult, transcriber: str = "gemini"):
        """Send transcription to backend API.

        Args:
            result: Transcription result
            transcriber: Which transcriber produced this ("gemini" or "whisper")
        """
        if not self._http_client:
            return

        try:
            payload = {
                "text": result.text,
                "timestamp": result.timestamp.isoformat(),
                "source": "radio",
                "confidence": result.confidence,
                "transcriber": transcriber
            }

            # Send to backend - different endpoints for different transcribers
            endpoint = f"/api/radio/transcription/{transcriber}"
            response = await self._http_client.post(
                f"{self.backend_url}{endpoint}",
                json=payload
            )

            if response.status_code in (200, 201):
                logger.debug(f"[{transcriber}] Transcription sent to backend")
            else:
                logger.warning(f"[{transcriber}] Backend response: {response.status_code}")

        except Exception as e:
            logger.error(f"[{transcriber}] Failed to send transcription: {e}")

    def get_stats(self) -> dict:
        """Get service statistics."""
        return {
            "running": self._running,
            "ec2_host": self.ec2_host,
            "ec2_port": self.ec2_port,
            "audio": self._audio_stats,
            "tcp_client": self.tcp_client.get_stats(),
            "gemini_transcriber": self.gemini_transcriber.get_stats(),
            "whisper_transcriber": self.whisper_transcriber.get_stats(),
            # Keep for backwards compatibility
            "transcriber": self.gemini_transcriber.get_stats()
        }


# Global singleton
_service: Optional[RadioService] = None


def get_radio_service() -> Optional[RadioService]:
    """Get the radio service instance."""
    return _service


async def init_radio_service(
    ec2_host: Optional[str] = None,
    ec2_port: Optional[int] = None,
    sample_rate: int = 16000,
    backend_url: str = "http://localhost:3000",
    chunk_duration: float = 3.0,
    silence_threshold: float = 500.0,
    silence_duration: float = 1.5,
    min_duration: float = 1.5,
    idle_timeout: float = 2.0,
    save_audio: bool = True,
    use_vad: bool = False,
    whisper_model_path: str = "models/whisper-large-v3-hebrew-ct2"
) -> RadioService:
    """Initialize and start the global radio service.

    Args:
        ec2_host: EC2 relay server hostname (default from EC2_RTP_HOST env)
        ec2_port: EC2 relay TCP port (default from EC2_RTP_PORT env)
        sample_rate: Audio sample rate
        backend_url: Backend API URL
        chunk_duration: Maximum audio chunk duration before forcing transcription
        silence_threshold: RMS threshold for silence detection
        silence_duration: Seconds of silence to trigger early processing
        min_duration: Minimum audio duration before processing
        idle_timeout: Seconds of no audio before processing buffer
        save_audio: Whether to save audio files for debugging
        use_vad: Whether to use Voice Activity Detection for speaker segmentation
        whisper_model_path: Path to Whisper CT2 model directory

    Returns:
        RadioService instance
    """
    global _service

    if _service is not None:
        logger.warning("Radio service already initialized")
        return _service

    _service = RadioService(
        ec2_host=ec2_host,
        ec2_port=ec2_port,
        sample_rate=sample_rate,
        backend_url=backend_url,
        chunk_duration=chunk_duration,
        silence_threshold=silence_threshold,
        silence_duration=silence_duration,
        min_duration=min_duration,
        idle_timeout=idle_timeout,
        save_audio=save_audio,
        use_vad=use_vad,
        whisper_model_path=whisper_model_path
    )

    await _service.start()
    return _service


async def stop_radio_service():
    """Stop the global radio service."""
    global _service

    if _service:
        await _service.stop()
        _service = None
