"""Segmentation module."""

from services.segmenter.base_segmenter import BaseSegmenter, SegmentationResult

try:
    from services.segmenter.sam2_segmenter import SAM2Segmenter

    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    SAM2Segmenter = None

__all__ = ["BaseSegmenter", "SegmentationResult", "SAM2Segmenter", "SAM2_AVAILABLE"]
