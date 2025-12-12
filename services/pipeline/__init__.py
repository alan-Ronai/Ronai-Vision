"""Pipeline services for multi-camera processing.

This package contains the core pipeline processor that orchestrates
detection, segmentation, ReID, and tracking.
"""

from services.pipeline.processor import FrameProcessor

__all__ = ["FrameProcessor"]
