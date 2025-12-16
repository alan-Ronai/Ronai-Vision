"""Multi-detector system that combines results from multiple YOLO models.

Allows running multiple specialized detectors (e.g., COCO + weapons) in parallel.
"""

import numpy as np
from typing import List, Dict, Optional
from services.detector.base_detector import BaseDetector, DetectionResult
from services.detector.yolo_detector import YOLODetector
import logging

logger = logging.getLogger(__name__)


class MultiDetector(BaseDetector):
    """Combines multiple detectors to detect different object categories.

    Example use cases:
    - COCO model (people, vehicles) + Weapons model (guns, knives)
    - General detection + Specialized detection (medical equipment, etc.)

    Each detector can have different confidence thresholds and class filtering.
    """

    def __init__(self, detectors: Dict[str, Dict], device: str = "cpu"):
        """Initialize multi-detector system.

        Args:
            detectors: Dict of detector configs, e.g.:
                {
                    "coco": {
                        "model": "yolo12n.pt",
                        "confidence": 0.25,
                        "classes": ["person", "car", "truck"],
                        "enabled": True
                    },
                    "weapons": {
                        "model": "weapon-detection.pt",
                        "confidence": 0.5,
                        "classes": None,  # None = all classes
                        "enabled": True
                    }
                }
            device: inference device ("cpu", "cuda", or "mps")
        """
        self.device = device
        self.detector_instances = {}
        self.detector_configs = detectors

        # Initialize each detector
        for name, config in detectors.items():
            if config.get("enabled", True):
                try:
                    logger.info(f"Loading detector: {name} ({config['model']})")
                    detector = YOLODetector(model_name=config["model"], device=device)
                    self.detector_instances[name] = {
                        "detector": detector,
                        "confidence": config.get("confidence", 0.5),
                        "classes": config.get("classes", None),
                        "priority": config.get(
                            "priority", 0
                        ),  # Higher = more important
                    }
                    logger.info(
                        f"✅ Loaded {name}: {len(detector.get_class_names())} classes"
                    )
                except Exception as e:
                    logger.error(f"Failed to load detector {name}: {e}")

    def predict(self, frame: np.ndarray, confidence: float = 0.5) -> DetectionResult:
        """Run all detectors and combine results.

        Args:
            frame: (H, W, 3) BGR numpy array
            confidence: default confidence (overridden by per-detector config)

        Returns:
            Combined DetectionResult with all detections
        """
        all_boxes = []
        all_scores = []
        all_class_ids = []
        all_class_names = []
        class_id_offset = 0

        # Run each detector
        for name, instance in self.detector_instances.items():
            detector = instance["detector"]
            conf = instance["confidence"]
            allowed_classes = instance["classes"]

            # Run detection with profiling
            from services.profiler import profiler

            with profiler.profile(f"yolo_detection_{name}"):
                result = detector.predict(frame, confidence=conf)

            if len(result) == 0:
                continue

            # Filter by allowed classes if specified
            if allowed_classes is not None:
                mask = self._filter_classes(
                    result, allowed_classes, detector.get_class_names()
                )
                if not mask.any():
                    continue

                result.boxes = result.boxes[mask]
                result.scores = result.scores[mask]
                result.class_ids = result.class_ids[mask]

            # Remap class IDs to avoid conflicts
            remapped_class_ids = result.class_ids + class_id_offset

            # Store results
            all_boxes.append(result.boxes)
            all_scores.append(result.scores)
            all_class_ids.append(remapped_class_ids)

            # Build combined class names list
            detector_class_names = detector.get_class_names()
            all_class_names.extend(detector_class_names)

            class_id_offset += len(detector_class_names)

            logger.debug(f"{name}: detected {len(result)} objects")

        # Combine all results
        if not all_boxes:
            return DetectionResult(
                boxes=np.zeros((0, 4), dtype=np.float32),
                scores=np.zeros(0, dtype=np.float32),
                class_ids=np.zeros(0, dtype=np.int32),
                class_names=all_class_names,
            )

        combined_boxes = np.vstack(all_boxes)
        combined_scores = np.concatenate(all_scores)
        combined_class_ids = np.concatenate(all_class_ids)

        return DetectionResult(
            boxes=combined_boxes,
            scores=combined_scores,
            class_ids=combined_class_ids,
            class_names=all_class_names,
        )

    def get_class_names(self) -> List[str]:
        """Return combined list of all class names from all detectors."""
        all_names = []
        for instance in self.detector_instances.values():
            all_names.extend(instance["detector"].get_class_names())
        return all_names

    @staticmethod
    def _filter_classes(
        result: DetectionResult, allowed_classes: List[str], class_names: List[str]
    ) -> np.ndarray:
        """Create boolean mask for allowed classes.

        Args:
            result: DetectionResult to filter
            allowed_classes: List of allowed class names
            class_names: Full list of class names from detector

        Returns:
            Boolean mask (N,) where True = keep detection
        """
        # Build set of allowed class IDs
        allowed_ids = set()
        for class_name in allowed_classes:
            if class_name in class_names:
                allowed_ids.add(class_names.index(class_name))

        # Create mask
        mask = np.isin(result.class_ids, list(allowed_ids))
        return mask

    def get_detector(self, name: str) -> Optional[YOLODetector]:
        """Get a specific detector by name."""
        instance = self.detector_instances.get(name)
        return instance["detector"] if instance else None

    def enable_detector(self, name: str):
        """Enable a detector."""
        if name in self.detector_configs:
            config = self.detector_configs[name]
            config["enabled"] = True

            if name not in self.detector_instances:
                detector = YOLODetector(model_name=config["model"], device=self.device)
                self.detector_instances[name] = {
                    "detector": detector,
                    "confidence": config.get("confidence", 0.5),
                    "classes": config.get("classes", None),
                    "priority": config.get("priority", 0),
                }
                logger.info(f"Enabled detector: {name}")

    def disable_detector(self, name: str):
        """Disable a detector."""
        if name in self.detector_instances:
            del self.detector_instances[name]
            logger.info(f"Disabled detector: {name}")
