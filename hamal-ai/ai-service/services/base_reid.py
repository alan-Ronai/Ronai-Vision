"""Base class for ReID (Re-Identification) feature extractors.

All ReID encoders should inherit from BaseReID and implement extract_features().
"""

import numpy as np
from abc import ABC, abstractmethod


class BaseReID(ABC):
    """Abstract base class for ReID feature extractors.

    All ReID models (OSNet, TransReID, CLIP, etc.) should inherit from this class
    and implement the extract_features method.
    """

    @abstractmethod
    def extract_features(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Extract L2-normalized feature vectors for detected objects.

        Args:
            frame: (H, W, 3) BGR uint8 image as produced by OpenCV
            boxes: (N, 4) array of [x1, y1, x2, y2] bounding boxes in pixel coords

        Returns:
            (N, D) numpy array of float32 features (L2-normalized)
            where D is the feature dimension (e.g., 512 for OSNet, 768 for TransReID)
        """
        pass
