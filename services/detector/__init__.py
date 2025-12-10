"""Object detection module."""

from services.detector.base_detector import BaseDetector, DetectionResult
from services.detector.yolo_detector import YOLODetector

__all__ = ["BaseDetector", "DetectionResult", "YOLODetector"]
