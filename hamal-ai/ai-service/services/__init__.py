"""AI Service modules

Organized into subpackages:
- gemini: Gemini AI analysis and verification
- reid: Re-identification and tracking
- streaming: Camera feed streaming (RTSP, FFMPEG)
- detection: Object detection and tracking
- radio: Radio communication services
"""

# Re-export commonly used classes for backwards compatibility
from .gemini import GeminiAnalyzer, GeminiVerifier
from .reid import ReIDTracker, OSNetReID, UniversalReID, TransReIDVehicle
from .streaming import (
    RTSPConfig,
    FFmpegRTSPReader,
    RTSPReaderManager,
    get_rtsp_manager,
    FFmpegConfig,
    get_ffmpeg_manager,
)
from .tts_service import TTSService

__all__ = [
    # Gemini
    'GeminiAnalyzer',
    'GeminiVerifier',
    # ReID
    'ReIDTracker',
    'OSNetReID',
    'UniversalReID',
    'TransReIDVehicle',
    # Streaming
    'RTSPConfig',
    'FFmpegRTSPReader',
    'RTSPReaderManager',
    'get_rtsp_manager',
    'FFmpegConfig',
    'get_ffmpeg_manager',
    # TTS
    'TTSService',
]
