"""ReID module: multi-class appearance-based re-identification.

This module supports specialized ReID encoders for different object classes:
- OSNet for people (lightweight, fast, high accuracy)
- TransReID-Vehicle for cars/trucks/motorcycles
- Universal encoder for other object types (animals, equipment, etc.)

The pipeline dynamically selects the right encoder based on detected class.
"""

import os
from typing import Dict, Optional

from services.reid.osnet_reid import OSNetReID
from services.reid.transreid_vehicle import TransReIDVehicle
from services.reid.universal_reid import UniversalReID


class MultiClassReID:
    """Multi-class ReID dispatcher.

    Routes detection crops to the appropriate ReID encoder based on class:
    - person → OSNet
    - vehicle classes (car, truck, motorcycle, bus) → TransReID-Vehicle
    - other classes → Universal encoder (fallback)
    """

    VEHICLE_CLASSES = {
        "car",
        "truck",
        "motorcycle",
        "bus",
        "van",
        "vehicle",  # generic
    }

    def __init__(self, device: Optional[str] = None):
        """Initialize multi-class ReID encoders.

        Args:
            device: Inference device ("cpu", "cuda", "mps")
        """
        self.device = device or os.environ.get("DEVICE", "cpu")
        self.encoders: Dict[str, Optional[BaseReID]] = {
            "person": None,
            "vehicle": None,
            "universal": None,
        }

        # Initialize person encoder (OSNet)
        try:
            self.encoders["person"] = OSNetReID(device=self.device)
        except Exception as e:
            print(f"[WARNING] OSNet person ReID failed: {e}")

        # Initialize vehicle encoder (TransReID-Vehicle)
        try:
            self.encoders["vehicle"] = TransReIDVehicle(device=self.device)
        except Exception as e:
            print(f"[WARNING] TransReID Vehicle failed: {e}")

        # Initialize universal fallback encoder
        try:
            self.encoders["universal"] = UniversalReID(device=self.device)
        except Exception as e:
            print(f"[WARNING] Universal ReID failed: {e}")

        # Verify at least one encoder is available
        available = [k for k, v in self.encoders.items() if v is not None]
        if not available:
            raise RuntimeError(
                "No ReID encoders available. Please install OSNet or check model availability."
            )
        print(f"[INFO] Multi-class ReID initialized with encoders: {available}")

    def get_encoder_for_class(self, class_name: str) -> Optional[BaseReID]:
        """Get the appropriate ReID encoder for a class.

        Args:
            class_name: YOLO class name (e.g., "person", "car", "dog")

        Returns:
            ReID encoder instance or None if unavailable
        """
        class_lower = class_name.lower()

        if class_lower == "person":
            return self.encoders["person"]
        elif class_lower in self.VEHICLE_CLASSES:
            return self.encoders["vehicle"]
        else:
            return self.encoders["universal"]

    def extract_features(self, frame, boxes, class_ids, class_names) -> Optional[dict]:
        """Extract features for detections, dispatching by class.

        Args:
            frame: (H, W, 3) BGR image
            boxes: (N, 4) detection boxes
            class_ids: (N,) class IDs from YOLO
            class_names: list of class name strings from YOLO

        Returns:
            Dict mapping class_name -> (indices, features) for that class
            or None if no valid encoders available
        """
        if boxes is None or len(boxes) == 0:
            return None

        results = {}

        # Group detections by class
        for class_name in set([class_names[int(cid)] for cid in class_ids]):
            class_mask = [class_names[int(cid)] == class_name for cid in class_ids]
            class_indices = [i for i, keep in enumerate(class_mask) if keep]
            class_boxes = boxes[class_mask]

            if len(class_boxes) == 0:
                continue

            encoder = self.get_encoder_for_class(class_name)
            if encoder is None:
                continue

            try:
                feats = encoder.extract_features(frame, class_boxes)
                results[class_name] = {
                    "indices": class_indices,
                    "features": feats,
                    "encoder": class_name
                    if class_name == "person"
                    else (
                        "vehicle" if class_name in self.VEHICLE_CLASSES else "universal"
                    ),
                }
            except Exception as e:
                print(f"[WARNING] Feature extraction failed for {class_name}: {e}")

        return results if results else None


def get_reid(model_name: str | None = None, device: str | None = None) -> BaseReID:
    """Factory: return an instantiated OSNet ReID extractor (single-class).

    **Deprecated**: Use MultiClassReID for multi-class support.

    Args:
        model_name: OSNet checkpoint filename (default: osnet_x0_5_imagenet.pth)
        device: Device to run on (default: from DEVICE env var or 'cpu')

    Returns:
        OSNetReID instance

    Raises:
        ImportError: If OSNet is not available
    """
    device = device or os.environ.get("DEVICE", "cpu")
    model_name = model_name or "osnet_x0_5_imagenet.pth"
    return OSNetReID(model_name, device=device)


def get_multi_class_reid(device: str | None = None) -> MultiClassReID:
    """Factory: return a multi-class ReID dispatcher.

    This is the recommended approach for multi-object tracking.

    Args:
        device: Device to run on (default: from DEVICE env var or 'cpu')

    Returns:
        MultiClassReID instance with specialized encoders for different classes
    """
    device = device or os.environ.get("DEVICE", "cpu")
    return MultiClassReID(device=device)


__all__ = [
    "BaseReID",
    "OSNetReID",
    "MultiClassReID",
    "get_reid",
    "get_multi_class_reid",
]
