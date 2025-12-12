"""RTP/RTSP Audio Server for military-grade audio ingestion.

This module provides a standalone RTP/RTSP server for receiving audio streams
from military equipment. It supports multiple codecs (G.711, Opus, AMR, MELPe)
and provides reliable ingestion with jitter buffering and packet loss handling.

Extended with bidirectional audio, Hebrew transcription, and command processing.
"""

from .rtp_server import RTPAudioServer
from .session_manager import RTSPSessionManager
from .transcriber import HebrewTranscriber
from .command_processor import CommandProcessor, CommandMatch
from .tts import HebrewTTS
from .audio_pipeline import AudioPipeline
from .rtp_sender import RTPAudioSender, RTPSenderConfig

__all__ = [
    "RTPAudioServer",
    "RTSPSessionManager",
    "HebrewTranscriber",
    "CommandProcessor",
    "CommandMatch",
    "HebrewTTS",
    "AudioPipeline",
    "RTPAudioSender",
    "RTPSenderConfig",
]
