"""Abstract base class for ReID (Re-Identification) feature extraction.

Concrete implementations should implement `extract_features(frame, boxes)` and
return a NumPy array of shape (N, D) with L2-normalized feature vectors.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseReID(ABC):
    """Abstract interface for ReID feature extraction."""

    @abstractmethod
    def extract_features(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """
        Extract ReID features from cropped regions.

        Args:
            frame: (H, W, 3) BGR numpy array
            boxes: (N, 4) array of [x1, y1, x2, y2] in pixel coords

        Returns:
            (N, D) feature vectors (float32)
        """
        raise NotImplementedError()
