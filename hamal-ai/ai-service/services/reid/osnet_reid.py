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
import logging

logger = logging.getLogger(__name__)

try:
    from torchreid.reid.utils.feature_extractor import FeatureExtractor
    TORCHREID_AVAILABLE = True
except ImportError:
    TORCHREID_AVAILABLE = False
    logger.warning("torchreid not available - ReID feature extraction disabled")


class OSNetReID:
    """OSNet ReID feature extractor.

    Loads a local checkpoint named `osnet_x0_5_imagenet.pth` (or other
    provided filename) from the repository `models/` directory.
    """

    def __init__(
        self, model_name: str = "osnet_x0_5_imagenet.pth", device: Optional[str] = None
    ):
        if not TORCHREID_AVAILABLE:
            raise ImportError("torchreid is required for OSNet ReID")

        # Resolve device
        env_device = os.environ.get("DEVICE", "cpu")
        if device is None:
            device = env_device

        # Resolve device: prefer requested device if available, fallback to CPU
        if device == "cpu":
            self.device = "cpu"
        elif device == "cuda":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "mps":
            try:
                self.device = "mps" if torch.backends.mps.is_available() else "cpu"
            except Exception:
                self.device = "cpu"
        else:
            self.device = "cpu"

        # Find models directory - primary location is hamal-ai/ai-service/models/
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up from services/reid/ to ai-service/
        ai_service_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
        models_dir = os.path.join(ai_service_dir, "models")

        # Search locations (prioritize ai-service/models)
        search_locations = [
            os.path.join(models_dir, model_name),  # Primary: ai-service/models/
            model_name,                             # Direct path if absolute
        ]

        checkpoint_path = None
        for candidate in search_locations:
            if os.path.exists(candidate):
                checkpoint_path = candidate
                break

        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"OSNet checkpoint '{model_name}' not found. "
                f"Expected at: {os.path.join(models_dir, model_name)}"
            )

        # Build a torchreid FeatureExtractor using an OSNet backbone
        # Note: Classifier layers are discarded (expected for feature extraction)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.extractor = FeatureExtractor(
                model_name="osnet_x0_5",
                device=self.device,
                model_path=checkpoint_path,
            )

        logger.info(f"✅ OSNet ReID initialized: {checkpoint_path} on {self.device}")

    def extract_features(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Extract L2-normalized embeddings for each box.

        Args:
            frame: (H, W, 3) BGR uint8 image as produced by OpenCV
            boxes: (N, 4) array of [x1, y1, x2, y2] in pixel coords

        Returns:
            (N, 512) numpy array of L2-normalized feature vectors (float32)
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
            feats = self.extractor(crops_rgb)
        except Exception as e:
            raise RuntimeError(f"OSNet extractor failed: {e}") from e

        # The extractor may return a torch.Tensor or a numpy array (or a list).
        # Convert to numpy if necessary.
        if isinstance(feats, torch.Tensor):
            feats = feats.detach().cpu().numpy()
        elif isinstance(feats, list):
            feats = np.asarray(feats)

        # L2-normalize and return numpy array (float32)
        feats = feats.astype(np.float32)
        norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-6
        feats = feats / norms.astype(np.float32)

        return feats
