"""Video Recording Service - Manages video recording with pre-buffer support.

This module provides:
- FrameBuffer: Circular buffer for storing recent frames per camera
- RecordingManager: Manages active recordings with FFmpeg encoding
- Integration with rule engine's start_recording action
"""

from .frame_buffer import FrameBuffer, get_frame_buffer
from .recording_manager import RecordingManager, get_recording_manager, init_recording_manager

__all__ = [
    "FrameBuffer",
    "get_frame_buffer",
    "RecordingManager",
    "get_recording_manager",
    "init_recording_manager",
]
