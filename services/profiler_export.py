"""Export profiler statistics to file for cross-process access.

When running server and worker in separate processes, the profiler stats
are not shared. This module exports stats to a JSON file that both processes
can read.
"""

import os
import json
from typing import Dict, Optional

PROFILER_STATE_DIR = "/tmp/ronai_profiler"
PROFILER_STATE_FILE = os.path.join(PROFILER_STATE_DIR, "stats.json")


def _ensure_dir():
    """Ensure profiler state directory exists."""
    os.makedirs(PROFILER_STATE_DIR, exist_ok=True)


def export_stats(stats: Dict, camera_stats: Dict):
    """Export profiler stats to file.

    Args:
        stats: Global stats dictionary
        camera_stats: Per-camera stats dictionary
    """
    _ensure_dir()

    try:
        data = {
            "global": stats,
            "cameras": camera_stats,
        }
        with open(PROFILER_STATE_FILE, "w") as f:
            json.dump(data, f)
    except Exception as e:
        # Don't fail if export fails
        pass


def load_stats() -> Dict:
    """Load profiler stats from file.

    Returns:
        Dict with 'global' and 'cameras' keys
    """
    _ensure_dir()

    if not os.path.exists(PROFILER_STATE_FILE):
        return {"global": {}, "cameras": {}}

    try:
        with open(PROFILER_STATE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {"global": {}, "cameras": {}}


def cleanup():
    """Clean up profiler state files."""
    try:
        if os.path.exists(PROFILER_STATE_FILE):
            os.remove(PROFILER_STATE_FILE)
    except Exception:
        pass
