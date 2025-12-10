"""OSNet-based ReID wrapper using torchreid and local checkpoints.

This module loads an OSNet checkpoint from `models/` and exposes
`OSNetReID.extract_features(frame, boxes)` which returns L2-normalized
embeddings as a NumPy array.

It expects a CPU or CUDA PyTorch runtime; behavior is governed by the
`DEVICE` environment variable (default: 'cpu').
"""

import os
import numpy as np
from typing import Optional

import torch

try:
    import torchreid
    from torchreid.utils import FeatureExtractor

    TORCHREID_AVAILABLE = True
except Exception:
    TORCHREID_AVAILABLE = False

from services.reid.base_reid import BaseReID, ReIDResult


class OSNetReID(BaseReID):
    """OSNet ReID feature extractor.

    Loads a local checkpoint named `osnet_x1_0_imagenet.pth` (or other
    provided filename) from the repository `models/` directory.
    """

    def __init__(
        self, model_name: str = "osnet_x0_5_imagenet.pth", device: Optional[str] = None
    ):
        if not TORCHREID_AVAILABLE:
            raise ImportError(
                "`torchreid` is required for OSNetReID. Install: `pip install torchreid`."
            )

        # Resolve device
        env_device = os.environ.get("DEVICE", "cpu")
        if device is None:
            device = env_device
        self.device = (
            device
            if device == "cpu"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        models_dir = os.path.join(repo_root, "models")
        checkpoint_path = os.path.join(models_dir, model_name)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"OSNet checkpoint not found at {checkpoint_path}. Place a .pth file in models/."
            )

        # Build a torchreid FeatureExtractor using an OSNet backbone
        # torchreid FeatureExtractor expects a model name registered in torchreid or a model file.
        # We'll use torchreid's model building utilities.
        try:
            self.extractor = FeatureExtractor(
                model_name="osnet_x0_5",
                device=self.device,
                model_path=checkpoint_path,
            )
        except Exception as e:
            # Fall back to loading via torchreid.models if needed
            raise RuntimeError(
                f"Failed to initialize torchreid FeatureExtractor: {e}"
            ) from e

    def extract_features(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Extract L2-normalized embeddings for each box.

        Args:
            frame: (H, W, 3) BGR uint8 image as produced by OpenCV
            boxes: (N, 4) array of [x1, y1, x2, y2] in pixel coords

        Returns:
            ReIDResult containing `features` as (N, D) numpy array (float32)
        """
        if boxes is None or len(boxes) == 0:
            return np.zeros((0, 512), dtype=np.float32)

        crops = []
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                # invalid, append a black patch
                crop = np.zeros((128, 64, 3), dtype=np.uint8)
            else:
                crop = frame[y1:y2, x1:x2]
            crops.append(crop)

        # FeatureExtractor from torchreid accepts a list of PIL/numpy images (RGB expected)
        # Convert BGR->RGB for each crop
        crops_rgb = [c[:, :, ::-1] if c.ndim == 3 else c for c in crops]

        try:
            feats = self.extractor(crops_rgb)  # returns numpy array (N, D)
        except Exception as e:
            raise RuntimeError(f"OSNet extractor failed: {e}") from e

        # L2-normalize and return numpy array
        norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-6
        feats = feats.astype(np.float32) / norms.astype(np.float32)

        return feats
