"""Detector factory: creates appropriate detector based on configuration.

This module provides a unified interface for creating detectors, supporting both:
- Single YOLO detector (default)
- Multi-detector with weapon detection (when enabled)
"""

import os
from typing import Optional

from services.detector import YOLODetector, MultiDetector
from config.pipeline_config import PipelineConfig


def create_detector(device: str = "cpu", force_weapon_detection: Optional[bool] = None):
    """Create detector based on configuration.

    Args:
        device: Device to run models on ('cpu', 'cuda', 'mps')
        force_weapon_detection: Override config weapon detection setting (None = use config)

    Returns:
        BaseDetector instance (either YOLODetector or MultiDetector)
    """
    enable_weapon = (
        force_weapon_detection
        if force_weapon_detection is not None
        else PipelineConfig.ENABLE_WEAPON_DETECTION
    )

    if not enable_weapon:
        # Single detector mode (default)
        print(f"[INFO] Initializing single detector: {PipelineConfig.YOLO_MODEL}")
        return YOLODetector(model_name=PipelineConfig.YOLO_MODEL, device=device)

    # Multi-detector mode with weapon detection
    print(f"[INFO] Initializing multi-detector with weapon detection")

    # Check if weapon model exists
    weapon_model_path = PipelineConfig.WEAPON_MODEL
    if not os.path.exists(weapon_model_path):
        print(
            f"[WARNING] Weapon model not found at {weapon_model_path}, downloading..."
        )
        print(f"[INFO] Falling back to single detector mode")
        return YOLODetector(model_name=PipelineConfig.YOLO_MODEL, device=device)

    # Configure multi-detector
    detector_config = {
        "primary": {
            "model": PipelineConfig.YOLO_MODEL,
            "confidence": PipelineConfig.YOLO_CONFIDENCE,
            "classes": PipelineConfig.ALLOWED_CLASSES,  # Filter classes if configured
            "priority": 1,
        },
        "weapon": {
            "model": weapon_model_path,
            "confidence": PipelineConfig.WEAPON_CONFIDENCE,
            "classes": None,  # Detect all weapon classes
            "priority": 2,  # Higher priority for weapon alerts
        },
    }

    print(
        f"  - Primary: {PipelineConfig.YOLO_MODEL} (conf={PipelineConfig.YOLO_CONFIDENCE})"
    )
    print(f"  - Weapon: {weapon_model_path} (conf={PipelineConfig.WEAPON_CONFIDENCE})")

    return MultiDetector(detector_config, device=device)


def get_detector_info(detector) -> dict:
    """Get information about the detector configuration.

    Args:
        detector: BaseDetector instance

    Returns:
        dict with detector type and configuration details
    """
    if isinstance(detector, MultiDetector):
        return {
            "type": "multi",
            "detectors": list(detector.detector_instances.keys()),
            "weapon_detection_enabled": True,
        }
    else:
        return {
            "type": "single",
            "model": detector.model_name
            if hasattr(detector, "model_name")
            else "unknown",
            "weapon_detection_enabled": False,
        }
