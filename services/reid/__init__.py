"""ReID module: export a CPU/GPU-aware ReID implementation.

If `torchreid` and an OSNet checkpoint are available in `models/`, this
module will expose `OSNetReID` as the preferred implementation. Otherwise
it falls back to `HistogramReID` (fast, CPU-only).
"""

import os

from services.reid.base_reid import BaseReID
from services.reid.histogram_reid import HistogramReID

OSNetReID = None
try:
    from services.reid.osnet_reid import OSNetReID

    OSNetReID = OSNetReID
except Exception:
    OSNetReID = None


def get_reid(model_name: str = None, device: str = None) -> BaseReID:
    """Factory: return an instantiated ReID extractor.

    Preference order:
      1. OSNetReID if available and checkpoint present
      2. HistogramReID (fallback)
    """
    # Respect explicit device or env
    device = device or os.environ.get("DEVICE", "cpu")

    if OSNetReID is not None:
        try:
            # model_name may be None; OSNetReID will resolve default
            return OSNetReID(
                model_name if model_name else "osnet_x0_5_imagenet.pth", device=device
            )
        except Exception:
            # Fall back
            return HistogramReID()

    return HistogramReID()


__all__ = ["BaseReID", "HistogramReID", "get_reid", "OSNetReID"]
