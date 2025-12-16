"""Object detection module."""

from services.detector.base_detector import BaseDetector, DetectionResult
from services.detector.yolo_detector import YOLODetector
from services.detector.multi_detector import MultiDetector
from services.detector.detector_factory import create_detector, get_detector_info

__all__ = [
    "BaseDetector",
    "DetectionResult",
    "YOLODetector",
    "MultiDetector",
    "create_detector",
    "get_detector_info",
]
