"""Abstract base class for object detectors."""

from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np


class DetectionResult:
    """Lightweight container for detection results."""

    def __init__(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        class_names: List[str] = None,
    ):
        """
        Args:
            boxes: (N, 4) array of [x1, y1, x2, y2] normalized or pixel coords
            scores: (N,) confidence scores [0, 1]
            class_ids: (N,) integer class IDs
            class_names: optional list of class name strings
        """
        self.boxes = boxes
        self.scores = scores
        self.class_ids = class_ids
        self.class_names = class_names or []

    def __len__(self):
        return len(self.boxes)


class BaseDetector(ABC):
    """Abstract interface for object detectors."""

    @abstractmethod
    def predict(self, frame: np.ndarray, confidence: float = 0.5) -> DetectionResult:
        """
        Run detection on a single frame.

        Args:
            frame: (H, W, 3) BGR numpy array
            confidence: confidence threshold [0, 1]

        Returns:
            DetectionResult with boxes, scores, class_ids
        """
        pass

    @abstractmethod
    def get_class_names(self) -> List[str]:
        """Return list of class names."""
        pass
