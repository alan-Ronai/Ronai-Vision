"""Radio Service - Receives RTP audio via EC2 relay and transcribes to Hebrew.

Integrates RTPTCPClient with GeminiTranscriber and sends
transcriptions to the backend for display in UI.

Architecture:
    Radio → RTP/UDP → EC2 Relay → TCP → This Service → Gemini → Backend → UI
"""

import os
import asyncio
import logging
from typing import Optional, Callable
from datetime import datetime

import httpx

from .rtp_tcp_client import RTPTCPClient
from .gemini_transcriber import StreamingGeminiTranscriber, TranscriptionResult

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
        on_transcription: Optional[Callable[[TranscriptionResult], None]] = None
    ):
        """Initialize radio service.

        Args:
            ec2_host: EC2 relay server hostname (default from EC2_RTP_HOST env)
            ec2_port: EC2 relay TCP port (default from EC2_RTP_PORT env)
            sample_rate: Audio sample rate
            backend_url: Backend API URL
            chunk_duration: Audio chunk duration for transcription
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

        # Initialize transcriber
        self.transcriber = StreamingGeminiTranscriber(
            sample_rate=sample_rate,
            chunk_duration=chunk_duration,
            on_transcription=self._handle_transcription
        )

        # Initialize TCP client for EC2 relay
        self.tcp_client = RTPTCPClient(
            host=self.ec2_host,
            port=self.ec2_port,
            target_sample_rate=sample_rate,
            audio_callback=self._on_audio_received
        )

        self._running = False
        logger.info(f"RadioService initialized - EC2 relay: {self.ec2_host}:{self.ec2_port}")

    async def start(self):
        """Start radio service."""
        if self._running:
            return

        self._running = True
        self._http_client = httpx.AsyncClient(timeout=10.0)

        # Set event loop for transcriber (for cross-thread async calls)
        try:
            loop = asyncio.get_running_loop()
            self.transcriber.set_event_loop(loop)
        except RuntimeError:
            logger.warning("Could not get running loop for transcriber")

        # Start TCP client
        self.tcp_client.start()

        logger.info("Radio service started")

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

        Args:
            audio_data: Raw PCM audio bytes
            sample_rate: Sample rate
        """
        # Feed to transcriber
        self.transcriber.add_audio(audio_data)

    def _handle_transcription(self, result: TranscriptionResult):
        """Handle transcription result.

        Args:
            result: Transcription result
        """
        if not result.text:
            return

        logger.info(f"Transcription: {result.text}")

        # Send to backend
        asyncio.create_task(self._send_to_backend(result))

        # Call external callback
        if self.on_transcription:
            self.on_transcription(result)

    async def _send_to_backend(self, result: TranscriptionResult):
        """Send transcription to backend API.

        Args:
            result: Transcription result
        """
        if not self._http_client:
            return

        try:
            payload = {
                "text": result.text,
                "timestamp": result.timestamp.isoformat(),
                "source": "radio",
                "confidence": result.confidence
            }

            # Send to backend
            response = await self._http_client.post(
                f"{self.backend_url}/api/radio/transcription",
                json=payload
            )

            if response.status_code in (200, 201):
                logger.debug(f"Transcription sent to backend")
            else:
                logger.warning(f"Backend response: {response.status_code}")

        except Exception as e:
            logger.error(f"Failed to send transcription: {e}")

    def get_stats(self) -> dict:
        """Get service statistics."""
        return {
            "running": self._running,
            "ec2_host": self.ec2_host,
            "ec2_port": self.ec2_port,
            "tcp_client": self.tcp_client.get_stats(),
            "transcriber": self.transcriber.get_stats()
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
    chunk_duration: float = 3.0
) -> RadioService:
    """Initialize and start the global radio service.

    Args:
        ec2_host: EC2 relay server hostname (default from EC2_RTP_HOST env)
        ec2_port: EC2 relay TCP port (default from EC2_RTP_PORT env)
        sample_rate: Audio sample rate
        backend_url: Backend API URL
        chunk_duration: Audio chunk duration for transcription

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
        chunk_duration=chunk_duration
    )

    await _service.start()
    return _service


async def stop_radio_service():
    """Stop the global radio service."""
    global _service

    if _service:
        await _service.stop()
        _service = None
