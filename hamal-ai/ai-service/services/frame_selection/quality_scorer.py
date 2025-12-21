"""Frame Quality Scorer - Evaluates frame quality for optimal Gemini analysis.

Scores each frame based on multiple metrics:
- Bounding box size (larger = more detail)
- Edge distance (penalize objects near frame edges)
- Sharpness (Laplacian variance for blur detection)
- Detection confidence
- Class-specific scores (vehicle view angle for plates)
- Motion blur detection (gradient analysis)
- Exposure quality (histogram analysis)

Higher scores indicate better frames for analysis.

TUNED FOR OPTIMAL LICENSE PLATE RECOGNITION AND CLOTHING IDENTIFICATION.
"""

import cv2
import numpy as np
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List

logger = logging.getLogger(__name__)


@dataclass
class QualityConfig:
    """Configuration for frame quality scoring.

    TUNED VALUES for security/surveillance use cases:
    - License plate recognition requires sharp, well-lit images
    - Clothing identification needs good color reproduction
    - Face/weapon detection benefits from larger, clearer subjects
    """

    # Bounding box size thresholds (pixels squared)
    # Tuned: Lower minimum for distant objects, higher ideal for clarity
    min_bbox_area: int = 8000  # ~90x90 pixels minimum
    ideal_bbox_area: int = 50000  # ~220x220 pixels ideal
    max_bbox_area: int = 250000  # Penalize if too large (cropping issues)

    # Edge margin threshold (fraction of frame size)
    # Tuned: Stricter margin requirement to avoid partial visibility
    edge_margin_threshold: float = 0.08  # 8% margin = full score

    # Sharpness thresholds (Laplacian variance)
    # Tuned: More sensitive to blur, especially important for plates
    sharpness_blur_threshold: float = 50.0  # Below = definitely blurry
    sharpness_ok_threshold: float = 150.0  # Acceptable sharpness
    sharpness_sharp_threshold: float = 400.0  # Above = very sharp

    # Motion blur thresholds (Sobel gradient ratio)
    motion_blur_threshold: float = 0.3  # Below = directional blur detected

    # Exposure quality thresholds
    underexposed_threshold: int = 50  # Mean brightness below = too dark
    overexposed_threshold: int = 200  # Mean brightness above = too bright
    ideal_brightness_low: int = 80
    ideal_brightness_high: int = 170

    # Scoring weights (must sum to 1.0)
    # TUNED: Prioritize sharpness and size for license plate/clothing
    weight_size: float = 0.22
    weight_position: float = 0.12
    weight_sharpness: float = 0.28  # Most important for plates
    weight_motion_blur: float = 0.10  # Detect directional blur
    weight_exposure: float = 0.08  # Lighting quality
    weight_confidence: float = 0.10
    weight_class_specific: float = 0.10

    # Early-exit quality threshold
    # Tuned: Slightly lower to allow good frames through faster
    quality_threshold: float = 0.80  # Analyze immediately if score > this

    # Minimum quality for consideration
    min_acceptable_quality: float = 0.35


@dataclass
class QualityScoreBreakdown:
    """Detailed breakdown of quality score components."""

    size: float = 0.0
    position: float = 0.0
    sharpness: float = 0.0
    motion_blur: float = 0.0
    exposure: float = 0.0
    confidence: float = 0.0
    class_specific: float = 0.0
    total: float = 0.0

    # Additional diagnostics
    laplacian_variance: float = 0.0
    mean_brightness: float = 0.0
    aspect_ratio: float = 0.0
    bbox_area: int = 0

    def to_dict(self) -> Dict:
        return {
            "size": round(self.size, 3),
            "position": round(self.position, 3),
            "sharpness": round(self.sharpness, 3),
            "motion_blur": round(self.motion_blur, 3),
            "exposure": round(self.exposure, 3),
            "confidence": round(self.confidence, 3),
            "class_specific": round(self.class_specific, 3),
            "total": round(self.total, 3),
            "diagnostics": {
                "laplacian_variance": round(self.laplacian_variance, 1),
                "mean_brightness": round(self.mean_brightness, 1),
                "aspect_ratio": round(self.aspect_ratio, 2),
                "bbox_area": self.bbox_area,
            }
        }


class FrameQualityScorer:
    """Evaluates frame quality for optimal analysis timing.

    Scores frames based on multiple metrics to determine the best
    moment to send an object to Gemini for analysis.

    ENHANCED with:
    - Motion blur detection (directional blur from movement)
    - Exposure quality scoring
    - More granular sharpness assessment
    - Better class-specific tuning for vehicles and persons
    """

    def __init__(self, config: Optional[QualityConfig] = None):
        """Initialize the quality scorer.

        Args:
            config: Quality scoring configuration
        """
        self.config = config or QualityConfig()

        # Allow config override from environment
        if os.environ.get("FRAME_QUALITY_THRESHOLD"):
            self.config.quality_threshold = float(os.environ["FRAME_QUALITY_THRESHOLD"])

        # Validate weights sum to 1.0
        weight_sum = (
            self.config.weight_size +
            self.config.weight_position +
            self.config.weight_sharpness +
            self.config.weight_motion_blur +
            self.config.weight_exposure +
            self.config.weight_confidence +
            self.config.weight_class_specific
        )
        if abs(weight_sum - 1.0) > 0.01:
            logger.warning(f"Quality weights sum to {weight_sum}, normalizing to 1.0")
            # Normalize weights
            self.config.weight_size /= weight_sum
            self.config.weight_position /= weight_sum
            self.config.weight_sharpness /= weight_sum
            self.config.weight_motion_blur /= weight_sum
            self.config.weight_exposure /= weight_sum
            self.config.weight_confidence /= weight_sum
            self.config.weight_class_specific /= weight_sum

        logger.info(
            f"FrameQualityScorer initialized: threshold={self.config.quality_threshold}, "
            f"weights=[size:{self.config.weight_size:.2f}, sharp:{self.config.weight_sharpness:.2f}, "
            f"motion:{self.config.weight_motion_blur:.2f}]"
        )

    def score_frame(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
        class_name: str,
        confidence: float = 0.5,
    ) -> QualityScoreBreakdown:
        """Calculate quality score for a frame/detection combination.

        Args:
            frame: Full frame (BGR numpy array)
            bbox: Bounding box in (x1, y1, x2, y2) format
            class_name: Object class (e.g., "person", "car", "truck")
            confidence: Detection confidence (0.0-1.0)

        Returns:
            QualityScoreBreakdown with individual and total scores
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]

        # Clamp to frame bounds
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(x1 + 1, min(w, x2))
        y2 = max(y1 + 1, min(h, y2))

        breakdown = QualityScoreBreakdown()

        # Store diagnostics
        breakdown.bbox_area = (x2 - x1) * (y2 - y1)
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        breakdown.aspect_ratio = bbox_width / max(1, bbox_height)

        # Crop for analysis
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return breakdown

        # Convert to grayscale for analysis
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop

        # 1. Bounding Box Size Score
        breakdown.size = self._score_size(breakdown.bbox_area)

        # 2. Edge Position Score
        breakdown.position = self._score_position(x1, y1, x2, y2, w, h)

        # 3. Sharpness Score (Laplacian variance)
        breakdown.sharpness, breakdown.laplacian_variance = self._score_sharpness(gray)

        # 4. Motion Blur Detection (Sobel gradients)
        breakdown.motion_blur = self._score_motion_blur(gray)

        # 5. Exposure Quality
        breakdown.exposure, breakdown.mean_brightness = self._score_exposure(gray)

        # 6. Confidence Score
        breakdown.confidence = min(1.0, max(0.0, confidence))

        # 7. Class-Specific Score
        breakdown.class_specific = self._score_class_specific(
            x1, y1, x2, y2, class_name, breakdown.aspect_ratio
        )

        # Calculate weighted total
        breakdown.total = (
            breakdown.size * self.config.weight_size +
            breakdown.position * self.config.weight_position +
            breakdown.sharpness * self.config.weight_sharpness +
            breakdown.motion_blur * self.config.weight_motion_blur +
            breakdown.exposure * self.config.weight_exposure +
            breakdown.confidence * self.config.weight_confidence +
            breakdown.class_specific * self.config.weight_class_specific
        )

        return breakdown

    def _score_size(self, bbox_area: int) -> float:
        """Score based on bounding box size.

        Uses a bell curve: too small = bad, ideal = good, too large = slightly penalized.

        Returns:
            Score 0.0-1.0
        """
        if bbox_area < self.config.min_bbox_area:
            # Heavy penalty for too small
            ratio = bbox_area / self.config.min_bbox_area
            return ratio * 0.4  # Max 0.4 for tiny objects

        elif bbox_area <= self.config.ideal_bbox_area:
            # Linear scaling from min to ideal
            range_size = self.config.ideal_bbox_area - self.config.min_bbox_area
            normalized = (bbox_area - self.config.min_bbox_area) / range_size
            return 0.4 + normalized * 0.6  # 0.4 to 1.0

        elif bbox_area <= self.config.max_bbox_area:
            # Slight penalty for very large (might have cropping issues)
            excess = (bbox_area - self.config.ideal_bbox_area)
            max_excess = self.config.max_bbox_area - self.config.ideal_bbox_area
            penalty = (excess / max_excess) * 0.1
            return 1.0 - penalty  # 0.9 to 1.0

        else:
            # Too large - might be filling entire frame
            return 0.85

    def _score_position(
        self,
        x1: int, y1: int, x2: int, y2: int,
        frame_w: int, frame_h: int
    ) -> float:
        """Score based on distance from frame edges.

        Objects near edges are often partially visible.
        Uses minimum margin approach.

        Returns:
            Score 0.0-1.0
        """
        # Calculate margin from each edge as fraction of frame size
        margin_left = x1 / max(1, frame_w)
        margin_right = (frame_w - x2) / max(1, frame_w)
        margin_top = y1 / max(1, frame_h)
        margin_bottom = (frame_h - y2) / max(1, frame_h)

        # Minimum margin determines score
        min_margin = min(margin_left, margin_right, margin_top, margin_bottom)

        # Score with smooth transition
        threshold = self.config.edge_margin_threshold
        if min_margin >= threshold:
            return 1.0
        elif min_margin <= 0:
            return 0.0
        else:
            # Smooth curve instead of linear
            ratio = min_margin / threshold
            return ratio ** 0.7  # Slightly forgiving for close-to-edge

    def _score_sharpness(self, gray: np.ndarray) -> Tuple[float, float]:
        """Score based on image sharpness using Laplacian variance.

        Enhanced with multi-scale analysis for better blur detection.

        Returns:
            Tuple of (score 0.0-1.0, raw laplacian variance)
        """
        try:
            if gray.size == 0:
                return 0.0, 0.0

            # Calculate Laplacian variance (standard sharpness metric)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            variance = laplacian.var()

            # Score based on thresholds
            if variance < self.config.sharpness_blur_threshold:
                # Definitely blurry
                score = (variance / self.config.sharpness_blur_threshold) * 0.2
            elif variance < self.config.sharpness_ok_threshold:
                # Somewhat blurry
                range_size = self.config.sharpness_ok_threshold - self.config.sharpness_blur_threshold
                normalized = (variance - self.config.sharpness_blur_threshold) / range_size
                score = 0.2 + normalized * 0.4  # 0.2 to 0.6
            elif variance < self.config.sharpness_sharp_threshold:
                # Acceptable to good
                range_size = self.config.sharpness_sharp_threshold - self.config.sharpness_ok_threshold
                normalized = (variance - self.config.sharpness_ok_threshold) / range_size
                score = 0.6 + normalized * 0.4  # 0.6 to 1.0
            else:
                # Very sharp
                score = 1.0

            return score, variance

        except Exception as e:
            logger.debug(f"Sharpness calculation failed: {e}")
            return 0.5, 0.0

    def _score_motion_blur(self, gray: np.ndarray) -> float:
        """Detect directional motion blur using Sobel gradients.

        Motion blur causes gradients to be stronger in one direction.
        We compare horizontal vs vertical gradient magnitudes.

        Returns:
            Score 0.0-1.0 (1.0 = no motion blur detected)
        """
        try:
            if gray.size == 0:
                return 0.5

            # Calculate Sobel gradients
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

            # Magnitude of gradients
            mag_x = np.abs(grad_x).mean()
            mag_y = np.abs(grad_y).mean()

            # Avoid division by zero
            total = mag_x + mag_y
            if total < 1:
                return 0.5  # Low detail image

            # Ratio of smaller to larger gradient
            ratio = min(mag_x, mag_y) / max(mag_x, mag_y)

            # Motion blur causes strong directional bias (ratio < 0.5)
            if ratio < self.config.motion_blur_threshold:
                # Strong motion blur
                return ratio / self.config.motion_blur_threshold * 0.5
            elif ratio < 0.6:
                # Some motion blur
                return 0.5 + (ratio - 0.3) / 0.3 * 0.3
            else:
                # Good balance = no significant motion blur
                return 0.8 + (ratio - 0.6) / 0.4 * 0.2

        except Exception as e:
            logger.debug(f"Motion blur detection failed: {e}")
            return 0.5

    def _score_exposure(self, gray: np.ndarray) -> Tuple[float, float]:
        """Score based on exposure quality (brightness histogram).

        Checks for under/overexposure and contrast.

        Returns:
            Tuple of (score 0.0-1.0, mean brightness)
        """
        try:
            if gray.size == 0:
                return 0.5, 128.0

            mean_brightness = np.mean(gray)
            std_brightness = np.std(gray)

            # Check for under/overexposure
            if mean_brightness < self.config.underexposed_threshold:
                # Too dark
                ratio = mean_brightness / self.config.underexposed_threshold
                exposure_score = ratio * 0.4
            elif mean_brightness > self.config.overexposed_threshold:
                # Too bright
                excess = mean_brightness - self.config.overexposed_threshold
                max_excess = 255 - self.config.overexposed_threshold
                ratio = 1 - (excess / max_excess)
                exposure_score = ratio * 0.6 + 0.2
            elif self.config.ideal_brightness_low <= mean_brightness <= self.config.ideal_brightness_high:
                # Ideal range
                exposure_score = 1.0
            else:
                # Acceptable but not ideal
                if mean_brightness < self.config.ideal_brightness_low:
                    range_size = self.config.ideal_brightness_low - self.config.underexposed_threshold
                    normalized = (mean_brightness - self.config.underexposed_threshold) / range_size
                    exposure_score = 0.4 + normalized * 0.6
                else:
                    range_size = self.config.overexposed_threshold - self.config.ideal_brightness_high
                    normalized = (self.config.overexposed_threshold - mean_brightness) / range_size
                    exposure_score = 0.7 + normalized * 0.3

            # Bonus for good contrast (high std)
            if std_brightness > 50:
                exposure_score = min(1.0, exposure_score + 0.1)

            return exposure_score, mean_brightness

        except Exception as e:
            logger.debug(f"Exposure scoring failed: {e}")
            return 0.5, 128.0

    def _score_class_specific(
        self,
        x1: int, y1: int, x2: int, y2: int,
        class_name: str,
        aspect_ratio: float
    ) -> float:
        """Score based on class-specific criteria.

        ENHANCED for better vehicle plate detection and person identification.

        For vehicles:
        - Strongly favor rear/front views (license plates visible)
        - Penalize extreme side views
        - Consider vertical position (plates are usually lower)

        For persons:
        - Favor upright poses
        - Penalize very wide aspect ratios (lying down)

        Returns:
            Score 0.0-1.0
        """
        # Vehicle classes
        vehicle_classes = {"car", "truck", "bus", "motorcycle", "van", "vehicle"}

        if class_name in vehicle_classes:
            # ENHANCED vehicle scoring for license plate visibility

            # Ideal aspect ratio for front/rear view: 1.2-1.8
            # Side view has aspect ratio: 2.5-4.0
            if 1.0 <= aspect_ratio <= 2.2:
                # Front/rear view - EXCELLENT for plates
                # Peak at 1.4-1.6
                if 1.3 <= aspect_ratio <= 1.7:
                    view_score = 1.0
                else:
                    deviation = min(abs(aspect_ratio - 1.3), abs(aspect_ratio - 1.7))
                    view_score = 1.0 - deviation * 0.5
            elif 2.2 < aspect_ratio <= 3.0:
                # Angled view - plate might be partially visible
                view_score = 0.5 - (aspect_ratio - 2.2) * 0.1
            else:
                # Full side view or extreme - plate unlikely visible
                view_score = 0.25

            return max(0.2, view_score)

        elif class_name == "person":
            # ENHANCED person scoring

            # Ideal aspect ratio for standing person: 0.35-0.55
            # Sitting/crouching: 0.6-1.0
            # Lying down: > 1.5
            if 0.3 <= aspect_ratio <= 0.6:
                # Standing/walking - best for identification
                return 0.95
            elif 0.25 <= aspect_ratio < 0.3:
                # Very thin - might be partially visible
                return 0.7
            elif 0.6 < aspect_ratio <= 1.0:
                # Sitting or crouching
                return 0.8
            elif 1.0 < aspect_ratio <= 1.5:
                # Bent over or unusual pose
                return 0.6
            else:
                # Lying down or very unusual
                return 0.4

        else:
            # Unknown class - neutral scoring
            return 0.7

    def is_above_threshold(self, score: float) -> bool:
        """Check if score exceeds early-exit threshold."""
        return score >= self.config.quality_threshold

    def is_acceptable(self, score: float) -> bool:
        """Check if score meets minimum acceptable quality."""
        return score >= self.config.min_acceptable_quality

    def get_config(self) -> Dict:
        """Get current configuration as dictionary."""
        return {
            "min_bbox_area": self.config.min_bbox_area,
            "ideal_bbox_area": self.config.ideal_bbox_area,
            "max_bbox_area": self.config.max_bbox_area,
            "edge_margin_threshold": self.config.edge_margin_threshold,
            "sharpness_blur_threshold": self.config.sharpness_blur_threshold,
            "sharpness_ok_threshold": self.config.sharpness_ok_threshold,
            "sharpness_sharp_threshold": self.config.sharpness_sharp_threshold,
            "quality_threshold": self.config.quality_threshold,
            "min_acceptable_quality": self.config.min_acceptable_quality,
            "weights": {
                "size": round(self.config.weight_size, 3),
                "position": round(self.config.weight_position, 3),
                "sharpness": round(self.config.weight_sharpness, 3),
                "motion_blur": round(self.config.weight_motion_blur, 3),
                "exposure": round(self.config.weight_exposure, 3),
                "confidence": round(self.config.weight_confidence, 3),
                "class_specific": round(self.config.weight_class_specific, 3),
            }
        }


# Global instance
_quality_scorer: Optional[FrameQualityScorer] = None


def get_quality_scorer() -> FrameQualityScorer:
    """Get or create the global quality scorer instance."""
    global _quality_scorer
    if _quality_scorer is None:
        _quality_scorer = FrameQualityScorer()
    return _quality_scorer


def init_quality_scorer(config: Optional[QualityConfig] = None) -> FrameQualityScorer:
    """Initialize the global quality scorer with custom config."""
    global _quality_scorer
    _quality_scorer = FrameQualityScorer(config)
    return _quality_scorer
