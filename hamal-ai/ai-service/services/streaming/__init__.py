"""Streaming services for camera feeds (RTSP, FFMPEG)."""

from .rtsp_reader import RTSPConfig, FFmpegRTSPReader, RTSPReaderManager, get_rtsp_manager
from .ffmpeg_rtsp import FFmpegConfig, FFmpegStream, FFmpegStreamManager, get_ffmpeg_manager

__all__ = [
    # RTSP Reader
    "RTSPConfig",
    "FFmpegRTSPReader",
    "RTSPReaderManager",
    "get_rtsp_manager",
    # FFmpeg
    "FFmpegConfig",
    "FFmpegStream",
    "FFmpegStreamManager",
    "get_ffmpeg_manager",
]
