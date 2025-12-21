"""Streaming services for camera feeds (RTSP, FFMPEG, GStreamer)."""

import os
import logging

from .rtsp_reader import RTSPConfig, FFmpegRTSPReader, RTSPReaderManager
from .rtsp_reader import get_rtsp_manager as _get_ffmpeg_rtsp_manager
from .ffmpeg_rtsp import FFmpegConfig, FFmpegStream, FFmpegStreamManager, get_ffmpeg_manager
from .gstreamer_rtsp import (
    GStreamerConfig,
    GStreamerRTSPReader,
    GStreamerStreamManager,
    get_gstreamer_manager,
    is_gstreamer_available,
)

logger = logging.getLogger(__name__)

# Cached backend choice
_rtsp_backend = None
_unified_manager = None


def get_rtsp_backend() -> str:
    """Get the configured RTSP backend (ffmpeg or gstreamer)."""
    global _rtsp_backend
    if _rtsp_backend is None:
        backend = os.environ.get("RTSP_BACKEND", "ffmpeg").lower()
        if backend == "gstreamer":
            if is_gstreamer_available():
                _rtsp_backend = "gstreamer"
                logger.info("✅ Using GStreamer RTSP backend")
            else:
                logger.warning("⚠️ GStreamer requested but not available, falling back to FFmpeg")
                _rtsp_backend = "ffmpeg"
        else:
            _rtsp_backend = "ffmpeg"
            logger.info("Using FFmpeg RTSP backend")
    return _rtsp_backend


def get_rtsp_manager():
    """Get the RTSP manager based on RTSP_BACKEND environment variable.

    Returns GStreamerStreamManager if RTSP_BACKEND=gstreamer and GStreamer is available,
    otherwise returns FFmpeg-based RTSPReaderManager.
    """
    global _unified_manager
    if _unified_manager is None:
        backend = get_rtsp_backend()
        if backend == "gstreamer":
            _unified_manager = get_gstreamer_manager()
        else:
            _unified_manager = _get_ffmpeg_rtsp_manager()
    return _unified_manager


__all__ = [
    # RTSP Reader (unified - uses backend based on RTSP_BACKEND env var)
    "RTSPConfig",
    "FFmpegRTSPReader",
    "RTSPReaderManager",
    "get_rtsp_manager",
    "get_rtsp_backend",
    # FFmpeg
    "FFmpegConfig",
    "FFmpegStream",
    "FFmpegStreamManager",
    "get_ffmpeg_manager",
    # GStreamer (recommended for RTSP)
    "GStreamerConfig",
    "GStreamerRTSPReader",
    "GStreamerStreamManager",
    "get_gstreamer_manager",
    "is_gstreamer_available",
]
