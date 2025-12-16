"""Detection services package."""

from .stable_tracker import (
    StableTracker,
    TrackedObject,
    get_stable_tracker,
    reset_stable_tracker
)

from .frame_buffer import (
    FrameBuffer,
    FrameBufferManager,
    get_frame_buffer_manager
)

__all__ = [
    "StableTracker",
    "TrackedObject",
    "get_stable_tracker",
    "reset_stable_tracker",
    "FrameBuffer",
    "FrameBufferManager",
    "get_frame_buffer_manager"
]
