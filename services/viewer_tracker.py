"""Cross-process viewer tracking using file system.

When running server and worker in separate processes, we can't use
in-memory broadcaster for viewer counts. This module uses the file system
to track active viewers.
"""

import os
import json
import threading
from typing import Dict

VIEWER_STATE_DIR = "/tmp/ronai_viewers"  # Shared state directory
VIEWER_STATE_FILE = os.path.join(VIEWER_STATE_DIR, "viewers.json")


def _ensure_dir():
    """Ensure viewer state directory exists."""
    os.makedirs(VIEWER_STATE_DIR, exist_ok=True)


def _load_viewers() -> Dict[str, int]:
    """Load viewer counts from file.

    Returns:
        Dict mapping camera_id to viewer_count
    """
    _ensure_dir()

    if not os.path.exists(VIEWER_STATE_FILE):
        return {}

    try:
        with open(VIEWER_STATE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_viewers(viewers: Dict[str, int]):
    """Save viewer counts to file.

    Args:
        viewers: Dict mapping camera_id to viewer_count
    """
    _ensure_dir()

    try:
        with open(VIEWER_STATE_FILE, "w") as f:
            json.dump(viewers, f)
    except Exception:
        pass


def increment_viewer(cam_id: str) -> int:
    """Increment viewer count for a camera (cross-process safe).

    Args:
        cam_id: Camera identifier

    Returns:
        New viewer count
    """
    viewers = _load_viewers()
    viewers[cam_id] = viewers.get(cam_id, 0) + 1
    _save_viewers(viewers)
    return viewers[cam_id]


def decrement_viewer(cam_id: str) -> int:
    """Decrement viewer count for a camera (cross-process safe).

    Args:
        cam_id: Camera identifier

    Returns:
        New viewer count (minimum 0)
    """
    viewers = _load_viewers()
    viewers[cam_id] = max(0, viewers.get(cam_id, 0) - 1)
    _save_viewers(viewers)
    return viewers[cam_id]


def get_viewer_count(cam_id: str) -> int:
    """Get viewer count for a camera.

    Args:
        cam_id: Camera identifier

    Returns:
        Number of active viewers
    """
    viewers = _load_viewers()
    return viewers.get(cam_id, 0)


def get_active_cameras() -> list:
    """Get list of cameras with active viewers.

    Returns:
        List of camera IDs with viewer_count > 0
    """
    viewers = _load_viewers()
    return [cam_id for cam_id, count in viewers.items() if count > 0]


def get_all_viewers() -> Dict[str, int]:
    """Get all viewer counts.

    Returns:
        Dict mapping camera_id to viewer_count
    """
    return _load_viewers()


def reset_viewers():
    """Reset all viewer counts (useful for cleanup)."""
    _save_viewers({})


def cleanup():
    """Clean up viewer state files."""
    try:
        if os.path.exists(VIEWER_STATE_FILE):
            os.remove(VIEWER_STATE_FILE)
    except Exception:
        pass
