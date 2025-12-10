"""Abstract base class for object trackers."""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np


class Track:
    """Lightweight container for a tracked object."""

    def __init__(
        self, track_id: int, box: np.ndarray, class_id: int, confidence: float
    ):
        """
        Args:
            track_id: unique integer ID for this track
            box: [x1, y1, x2, y2] bounding box in pixels
            class_id: integer class ID
            confidence: detection confidence [0, 1]
        """
        self.track_id = track_id
        self.box = box
        self.class_id = class_id
        self.confidence = confidence
        self.age = 0  # frames since first detection
        self.hits = 0  # consecutive successful detections


class BaseTracker(ABC):
    """Abstract interface for object trackers."""

    @abstractmethod
    def update(
        self, boxes: np.ndarray, class_ids: np.ndarray, confidences: np.ndarray
    ) -> List[Track]:
        """
        Update tracker with new detections.

        Args:
            boxes: (N, 4) array of [x1, y1, x2, y2] in pixel coords
            class_ids: (N,) integer class IDs
            confidences: (N,) confidence scores [0, 1]

        Returns:
            List of Track objects with assigned IDs
        """
        pass

    @abstractmethod
    def get_active_tracks(self) -> List[Track]:
        """Return currently active tracks."""
        pass
