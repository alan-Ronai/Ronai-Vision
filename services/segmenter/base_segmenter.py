"""Abstract base class for segmenters."""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import numpy as np


class SegmentationResult:
    """Lightweight container for segmentation results."""

    def __init__(
        self,
        masks: np.ndarray,
        scores: np.ndarray,
        class_ids: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None,
    ):
        """
        Args:
            masks: (N, H, W) binary or float masks [0, 1] or {0, 1}
            scores: (N,) confidence/quality scores [0, 1]
            class_ids: optional (N,) integer class IDs
            class_names: optional list of class name strings
        """
        self.masks = masks
        self.scores = scores
        self.class_ids = class_ids or np.array([], dtype=np.int32)
        self.class_names = class_names or []

    def __len__(self):
        return len(self.masks)


class BaseSegmenter(ABC):
    """Abstract interface for image segmenters."""

    @abstractmethod
    def segment(
        self,
        frame: np.ndarray,
        boxes: Optional[np.ndarray] = None,
        points: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
    ) -> SegmentationResult:
        """
        Run segmentation on a frame.

        Args:
            frame: (H, W, 3) BGR numpy array
            boxes: optional (N, 4) array of [x1, y1, x2, y2] prompt boxes
            points: optional (N, 2) array of [x, y] prompt points
            labels: optional (N,) array of labels (1=foreground, 0=background)

        Returns:
            SegmentationResult with masks, scores, and optional class info
        """
        pass
