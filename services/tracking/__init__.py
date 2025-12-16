"""Tracking services for persistent object identification."""

from .reid_tracker import (
    ReIDTracker,
    TrackedObject,
    get_reid_tracker,
    reset_reid_tracker,
    DEEPSORT_AVAILABLE
)

__all__ = [
    'ReIDTracker',
    'TrackedObject',
    'get_reid_tracker',
    'reset_reid_tracker',
    'DEEPSORT_AVAILABLE'
]
