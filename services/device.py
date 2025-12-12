"""Device selection helper.

Provides a small helper to pick the best device based on environment and
runtime availability. Supports: 'cpu', 'cuda' and 'mps' (Apple Silicon Metal).

Environment variable supported:
  DEVICE=auto|cpu|cuda|mps|cuda:0

Behavior:
  - If DEVICE is set to 'cpu', 'cuda', 'mps' or a specific CUDA id, that value
    is returned (after a basic validation).
  - If DEVICE is 'auto' (default) the helper will return 'cuda' if CUDA is
    available, then 'mps' if MPS is available, otherwise 'cpu'.
"""

from __future__ import annotations

import os
from typing import Optional

try:
    import torch
except Exception:  # pragma: no cover - torch may be missing in some test envs
    torch = None


def choose_device(env_var: str = "DEVICE") -> str:
    """Return the device string to use for models.

    Args:
        env_var: environment variable name to read (default: DEVICE)

    Returns:
        One of: 'cpu', 'cuda', 'mps' or a CUDA device like 'cuda:0'
    """
    raw = os.getenv(env_var, "auto").strip().lower()

    # Accept explicit values directly
    if raw in ("cpu", "mps"):
        return raw
    if raw.startswith("cuda"):
        # cuda, cuda:0, cuda:1 etc.
        return raw

    # Auto-detect using torch if available
    if raw == "auto":
        if torch is not None:
            try:
                if torch.cuda.is_available():
                    return "cuda"
            except Exception:
                pass
            # MPS (Apple Silicon) support
            try:
                if (
                    getattr(torch.backends, "mps", None) is not None
                    and torch.backends.mps.is_available()
                ):
                    return "mps"
            except Exception:
                pass
        return "cpu"

    # Unknown value — fall back to cpu
    return "cpu"


__all__ = ["choose_device"]
