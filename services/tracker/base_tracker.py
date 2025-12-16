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

        # Metadata system for storing custom context/annotations
        self.metadata = {
            "created_at": None,  # Timestamp when track was created
            "updated_at": None,  # Last update timestamp
            "notes": [],  # User notes/annotations
            "tags": [],  # Custom tags
            "attributes": {},  # Custom key-value attributes
            "alerts": [],  # Alert history
            "zones_visited": set(),  # Zones/areas visited
            "behavior": None,  # Detected behavior
            "custom": {},  # Flexible custom metadata
        }

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

    # ========================================================================
    # METADATA MANAGEMENT METHODS
    # ========================================================================

    def set_metadata(self, key: str, value, category: str = "custom"):
        """Set metadata value.

        Args:
            key: Metadata key
            value: Metadata value
            category: Category (custom, attributes, etc.)
        """
        import time

        if category == "custom" or category not in self.metadata:
            self.metadata["custom"][key] = value
        else:
            self.metadata[category][key] = value
        self.metadata["updated_at"] = time.time()

    def get_metadata(self, key: str, default=None, category: str = "custom"):
        """Get metadata value.

        Args:
            key: Metadata key
            default: Default value if key not found
            category: Category to search in

        Returns:
            Metadata value or default
        """
        if category == "custom" or category not in self.metadata:
            return self.metadata["custom"].get(key, default)
        return self.metadata.get(category, {}).get(key, default)

    def add_note(self, note: str, author: str = "system"):
        """Add a note/annotation to the track.

        Args:
            note: Note text
            author: Author of the note
        """
        import time

        self.metadata["notes"].append(
            {"text": note, "author": author, "timestamp": time.time()}
        )
        self.metadata["updated_at"] = time.time()

    def add_tag(self, tag: str):
        """Add a tag to the track.

        Args:
            tag: Tag to add
        """
        import time

        if tag not in self.metadata["tags"]:
            self.metadata["tags"].append(tag)
            self.metadata["updated_at"] = time.time()

    def remove_tag(self, tag: str):
        """Remove a tag from the track.

        Args:
            tag: Tag to remove
        """
        import time

        if tag in self.metadata["tags"]:
            self.metadata["tags"].remove(tag)
            self.metadata["updated_at"] = time.time()

    def add_alert(self, alert_type: str, message: str, severity: str = "info"):
        """Add an alert to the track.

        Args:
            alert_type: Type of alert (e.g., 'weapon_detected', 'zone_violation')
            message: Alert message
            severity: Severity level (info, warning, critical)
        """
        import time

        self.metadata["alerts"].append(
            {
                "type": alert_type,
                "message": message,
                "severity": severity,
                "timestamp": time.time(),
            }
        )
        self.metadata["updated_at"] = time.time()

    def set_attribute(self, key: str, value):
        """Set a custom attribute.

        Args:
            key: Attribute key
            value: Attribute value
        """
        import time

        self.metadata["attributes"][key] = value
        self.metadata["updated_at"] = time.time()

    def get_attribute(self, key: str, default=None):
        """Get a custom attribute.

        Args:
            key: Attribute key
            default: Default value if not found

        Returns:
            Attribute value or default
        """
        return self.metadata["attributes"].get(key, default)

    def add_zone(self, zone_name: str):
        """Add a zone to visited zones.

        Args:
            zone_name: Name of the zone
        """
        import time

        self.metadata["zones_visited"].add(zone_name)
        self.metadata["updated_at"] = time.time()

    def set_behavior(self, behavior: str, confidence: float = 1.0):
        """Set detected behavior.

        Args:
            behavior: Behavior type (e.g., 'walking', 'running', 'loitering')
            confidence: Confidence in behavior detection
        """
        import time

        self.metadata["behavior"] = {
            "type": behavior,
            "confidence": confidence,
            "detected_at": time.time(),
        }
        self.metadata["updated_at"] = time.time()

    def get_metadata_summary(self) -> dict:
        """Get a summary of all metadata.

        Returns:
            Dictionary with metadata summary
        """
        # Convert sets to lists for JSON serialization
        summary = self.metadata.copy()
        summary["zones_visited"] = list(summary["zones_visited"])
        return summary

    def clear_metadata(self, preserve_timestamps: bool = True):
        """Clear all metadata.

        Args:
            preserve_timestamps: Whether to preserve created_at timestamp
        """
        import time

        created_at = self.metadata.get("created_at") if preserve_timestamps else None

        self.metadata = {
            "created_at": created_at,
            "updated_at": time.time(),
            "notes": [],
            "tags": [],
            "attributes": {},
            "alerts": [],
            "zones_visited": set(),
            "behavior": None,
            "custom": {},
        }


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
