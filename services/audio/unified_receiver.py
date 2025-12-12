"""Unified audio receiver manager.

Coordinates multiple audio input protocols:
- RTSP (via RTPAudioServer)
- Raw RTP/UDP (via RawRTPReceiver)
- SIP + RTP (via SIPServer)
- FFmpeg (via FFmpegAudioReceiver)
- GStreamer (via GStreamerAudioReceiver)
- HTTP streaming

Manages multiple concurrent audio streams and routes them to processing pipeline.
"""

import logging
from typing import Optional, Dict, List, Callable
from enum import Enum

from .rtp_server import RTPAudioServer
from .raw_rtp_receiver import RawRTPReceiver
from .sip_server import SIPServer
from .stream_integrations import FFmpegAudioReceiver, GStreamerAudioReceiver
from .audio_pipeline import AudioPipeline

logger = logging.getLogger(__name__)


class AudioProtocol(str, Enum):
    """Supported audio protocols."""

    RTSP = "rtsp"
    RAW_RTP = "raw_rtp"
    SIP = "sip"
    FFMPEG = "ffmpeg"
    GSTREAMER = "gstreamer"
    HTTP = "http"


class UnifiedAudioReceiver:
    """Unified manager for all audio input protocols.

    Manages multiple audio receivers and routes audio to processing pipeline.
    Provides single interface for starting/stopping different protocol receivers.
    """

    def __init__(
        self,
        audio_pipeline: Optional[AudioPipeline] = None,
        storage_path: str = "audio_storage/recordings",
    ):
        """Initialize unified audio receiver.

        Args:
            audio_pipeline: Audio processing pipeline (optional)
            storage_path: Path to store audio recordings
        """
        self.audio_pipeline = audio_pipeline
        self.storage_path = storage_path

        # Protocol servers
        self.rtsp_server: Optional[RTPAudioServer] = None
        self.sip_server: Optional[SIPServer] = None

        # Active receivers
        self._raw_rtp_receivers: Dict[str, RawRTPReceiver] = {}
        self._ffmpeg_receivers: Dict[str, FFmpegAudioReceiver] = {}
        self._gstreamer_receivers: Dict[str, GStreamerAudioReceiver] = {}

        # State
        self._enabled_protocols: set[AudioProtocol] = set()

        logger.info("UnifiedAudioReceiver initialized")

    # ========================================================================
    # PROTOCOL ENABLE/DISABLE
    # ========================================================================

    def enable_rtsp(
        self,
        rtsp_host: str = "0.0.0.0",
        rtsp_port: int = 8554,
        rtp_base_port: int = 5004,
    ):
        """Enable RTSP audio server.

        Args:
            rtsp_host: RTSP server bind address
            rtsp_port: RTSP server port
            rtp_base_port: Base port for RTP streams
        """
        if AudioProtocol.RTSP in self._enabled_protocols:
            logger.warning("RTSP already enabled")
            return

        logger.info("Enabling RTSP audio server...")

        self.rtsp_server = RTPAudioServer(
            rtsp_host=rtsp_host,
            rtsp_port=rtsp_port,
            rtp_base_port=rtp_base_port,
            storage_path=self.storage_path,
        )
        self.rtsp_server.start()

        self._enabled_protocols.add(AudioProtocol.RTSP)
        logger.info("RTSP audio server enabled")

    def disable_rtsp(self):
        """Disable RTSP audio server."""
        if AudioProtocol.RTSP not in self._enabled_protocols:
            return

        logger.info("Disabling RTSP audio server...")

        if self.rtsp_server:
            self.rtsp_server.stop()
            self.rtsp_server = None

        self._enabled_protocols.discard(AudioProtocol.RTSP)
        logger.info("RTSP audio server disabled")

    def enable_sip(
        self,
        sip_host: str = "0.0.0.0",
        sip_port: int = 5060,
        rtp_base_port: int = 10000,
    ):
        """Enable SIP + RTP server.

        Args:
            sip_host: SIP server bind address
            sip_port: SIP server port
            rtp_base_port: Base port for RTP streams
        """
        if AudioProtocol.SIP in self._enabled_protocols:
            logger.warning("SIP already enabled")
            return

        logger.info("Enabling SIP audio server...")

        self.sip_server = SIPServer(
            sip_host=sip_host,
            sip_port=sip_port,
            rtp_base_port=rtp_base_port,
            storage_path=self.storage_path,
        )
        self.sip_server.start()

        self._enabled_protocols.add(AudioProtocol.SIP)
        logger.info("SIP audio server enabled")

    def disable_sip(self):
        """Disable SIP server."""
        if AudioProtocol.SIP not in self._enabled_protocols:
            return

        logger.info("Disabling SIP audio server...")

        if self.sip_server:
            self.sip_server.stop()
            self.sip_server = None

        self._enabled_protocols.discard(AudioProtocol.SIP)
        logger.info("SIP audio server disabled")

    def add_raw_rtp_receiver(
        self,
        receiver_id: str,
        listen_port: int,
        codec: str = "g711_ulaw",
        audio_callback: Optional[Callable] = None,
    ) -> str:
        """Add raw RTP/UDP receiver.

        Args:
            receiver_id: Unique receiver identifier
            listen_port: UDP port to listen on
            codec: Audio codec (default "g711_ulaw")
            audio_callback: Optional audio callback

        Returns:
            Receiver ID
        """
        if receiver_id in self._raw_rtp_receivers:
            logger.warning(f"Raw RTP receiver '{receiver_id}' already exists")
            return receiver_id

        logger.info(f"Adding raw RTP receiver '{receiver_id}' on port {listen_port}")

        # Create callback that routes to pipeline
        def pipeline_callback(audio, sample_rate):
            if self.audio_pipeline:
                self.audio_pipeline.process_audio_chunk(
                    session_id=receiver_id, audio=audio, sample_rate=sample_rate
                )
            if audio_callback:
                audio_callback(audio, sample_rate)

        receiver = RawRTPReceiver(
            listen_host="0.0.0.0",
            listen_port=listen_port,
            storage_path=self.storage_path,
            default_codec=codec,
            audio_callback=pipeline_callback,
        )
        receiver.start()

        self._raw_rtp_receivers[receiver_id] = receiver
        self._enabled_protocols.add(AudioProtocol.RAW_RTP)

        logger.info(f"Raw RTP receiver '{receiver_id}' started")
        return receiver_id

    def remove_raw_rtp_receiver(self, receiver_id: str):
        """Remove raw RTP receiver.

        Args:
            receiver_id: Receiver identifier
        """
        receiver = self._raw_rtp_receivers.get(receiver_id)
        if not receiver:
            logger.warning(f"Raw RTP receiver '{receiver_id}' not found")
            return

        logger.info(f"Removing raw RTP receiver '{receiver_id}'")

        receiver.stop()
        del self._raw_rtp_receivers[receiver_id]

        if len(self._raw_rtp_receivers) == 0:
            self._enabled_protocols.discard(AudioProtocol.RAW_RTP)

        logger.info(f"Raw RTP receiver '{receiver_id}' removed")

    def add_ffmpeg_receiver(
        self,
        receiver_id: str,
        input_url: str,
        sample_rate: int = 16000,
        audio_callback: Optional[Callable] = None,
    ) -> str:
        """Add FFmpeg audio receiver.

        Args:
            receiver_id: Unique receiver identifier
            input_url: Input URL or file path
            sample_rate: Output sample rate
            audio_callback: Optional audio callback

        Returns:
            Receiver ID
        """
        if receiver_id in self._ffmpeg_receivers:
            logger.warning(f"FFmpeg receiver '{receiver_id}' already exists")
            return receiver_id

        logger.info(f"Adding FFmpeg receiver '{receiver_id}' for {input_url}")

        # Create callback that routes to pipeline
        def pipeline_callback(audio, sample_rate):
            if self.audio_pipeline:
                self.audio_pipeline.process_audio_chunk(
                    session_id=receiver_id, audio=audio, sample_rate=sample_rate
                )
            if audio_callback:
                audio_callback(audio, sample_rate)

        receiver = FFmpegAudioReceiver(
            input_url=input_url,
            sample_rate=sample_rate,
            audio_callback=pipeline_callback,
        )
        receiver.start()

        self._ffmpeg_receivers[receiver_id] = receiver
        self._enabled_protocols.add(AudioProtocol.FFMPEG)

        logger.info(f"FFmpeg receiver '{receiver_id}' started")
        return receiver_id

    def remove_ffmpeg_receiver(self, receiver_id: str):
        """Remove FFmpeg receiver.

        Args:
            receiver_id: Receiver identifier
        """
        receiver = self._ffmpeg_receivers.get(receiver_id)
        if not receiver:
            logger.warning(f"FFmpeg receiver '{receiver_id}' not found")
            return

        logger.info(f"Removing FFmpeg receiver '{receiver_id}'")

        receiver.stop()
        del self._ffmpeg_receivers[receiver_id]

        if len(self._ffmpeg_receivers) == 0:
            self._enabled_protocols.discard(AudioProtocol.FFMPEG)

        logger.info(f"FFmpeg receiver '{receiver_id}' removed")

    def add_gstreamer_receiver(
        self,
        receiver_id: str,
        pipeline_string: Optional[str] = None,
        input_url: Optional[str] = None,
        sample_rate: int = 16000,
        audio_callback: Optional[Callable] = None,
    ) -> str:
        """Add GStreamer audio receiver.

        Args:
            receiver_id: Unique receiver identifier
            pipeline_string: GStreamer pipeline string (optional)
            input_url: Input URL (used if pipeline_string not provided)
            sample_rate: Output sample rate
            audio_callback: Optional audio callback

        Returns:
            Receiver ID
        """
        if receiver_id in self._gstreamer_receivers:
            logger.warning(f"GStreamer receiver '{receiver_id}' already exists")
            return receiver_id

        logger.info(f"Adding GStreamer receiver '{receiver_id}'")

        # Create callback that routes to pipeline
        def pipeline_callback(audio, sample_rate):
            if self.audio_pipeline:
                self.audio_pipeline.process_audio_chunk(
                    session_id=receiver_id, audio=audio, sample_rate=sample_rate
                )
            if audio_callback:
                audio_callback(audio, sample_rate)

        receiver = GStreamerAudioReceiver(
            pipeline_string=pipeline_string,
            input_url=input_url,
            sample_rate=sample_rate,
            audio_callback=pipeline_callback,
        )
        receiver.start()

        self._gstreamer_receivers[receiver_id] = receiver
        self._enabled_protocols.add(AudioProtocol.GSTREAMER)

        logger.info(f"GStreamer receiver '{receiver_id}' started")
        return receiver_id

    def remove_gstreamer_receiver(self, receiver_id: str):
        """Remove GStreamer receiver.

        Args:
            receiver_id: Receiver identifier
        """
        receiver = self._gstreamer_receivers.get(receiver_id)
        if not receiver:
            logger.warning(f"GStreamer receiver '{receiver_id}' not found")
            return

        logger.info(f"Removing GStreamer receiver '{receiver_id}'")

        receiver.stop()
        del self._gstreamer_receivers[receiver_id]

        if len(self._gstreamer_receivers) == 0:
            self._enabled_protocols.discard(AudioProtocol.GSTREAMER)

        logger.info(f"GStreamer receiver '{receiver_id}' removed")

    # ========================================================================
    # STATUS & MANAGEMENT
    # ========================================================================

    def get_status(self) -> dict:
        """Get unified receiver status.

        Returns:
            Status dictionary with all active receivers
        """
        return {
            "enabled_protocols": list(self._enabled_protocols),
            "rtsp": self.rtsp_server.get_status() if self.rtsp_server else None,
            "sip": self.sip_server.get_stats() if self.sip_server else None,
            "raw_rtp_receivers": list(self._raw_rtp_receivers.keys()),
            "ffmpeg_receivers": list(self._ffmpeg_receivers.keys()),
            "gstreamer_receivers": list(self._gstreamer_receivers.keys()),
            "total_receivers": (
                len(self._raw_rtp_receivers)
                + len(self._ffmpeg_receivers)
                + len(self._gstreamer_receivers)
            ),
        }

    def stop_all(self):
        """Stop all audio receivers."""
        logger.info("Stopping all audio receivers...")

        # Stop protocol servers
        self.disable_rtsp()
        self.disable_sip()

        # Stop all receivers
        for receiver_id in list(self._raw_rtp_receivers.keys()):
            self.remove_raw_rtp_receiver(receiver_id)

        for receiver_id in list(self._ffmpeg_receivers.keys()):
            self.remove_ffmpeg_receiver(receiver_id)

        for receiver_id in list(self._gstreamer_receivers.keys()):
            self.remove_gstreamer_receiver(receiver_id)

        logger.info("All audio receivers stopped")
