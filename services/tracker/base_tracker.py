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
        self.confidence_history = []  # Track confidence over time for validation

    def update_confidence(self, confidence: float, history_size: int = 10):
        """Update confidence history for track validation.

        Args:
            confidence: New confidence value
            history_size: Maximum number of samples to keep
        """
        self.confidence = confidence
        self.confidence_history.append(confidence)

        # Keep only last N samples
        if len(self.confidence_history) > history_size:
            self.confidence_history = self.confidence_history[-history_size:]

    def get_avg_confidence(self) -> float:
        """Get average confidence over history.

        Returns:
            Average confidence, or current confidence if no history
        """
        if not self.confidence_history:
            return self.confidence
        return float(np.mean(self.confidence_history))


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
