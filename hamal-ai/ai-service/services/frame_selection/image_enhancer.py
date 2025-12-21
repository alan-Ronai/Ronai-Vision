"""Image Enhancer - Enhances frames before Gemini analysis.

Provides multiple enhancement techniques:
- Adaptive contrast enhancement (CLAHE)
- Smart sharpening (unsharp mask)
- Denoising (Non-local Means or bilateral filter)
- Auto white balance
- Brightness/gamma correction
- License plate region enhancement

These enhancements improve Gemini's ability to read license plates,
identify clothing colors, and detect weapons.
"""

import cv2
import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class EnhancementLevel(Enum):
    """Preset enhancement levels."""
    NONE = "none"
    LIGHT = "light"  # Subtle improvements
    MODERATE = "moderate"  # Default, balanced
    AGGRESSIVE = "aggressive"  # Maximum clarity


@dataclass
class EnhancementConfig:
    """Configuration for image enhancement."""

    # Master enable
    enabled: bool = True
    level: EnhancementLevel = EnhancementLevel.MODERATE

    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe_enabled: bool = True
    clahe_clip_limit: float = 2.5  # Higher = more contrast (1.0-4.0)
    clahe_grid_size: int = 8  # Grid size for local adaptation

    # Sharpening
    sharpen_enabled: bool = True
    sharpen_amount: float = 1.2  # Sharpening strength (0.5-2.0)
    sharpen_radius: float = 1.0  # Gaussian blur radius for unsharp mask
    sharpen_threshold: int = 5  # Only sharpen if difference > threshold

    # Denoising
    denoise_enabled: bool = True
    denoise_strength: int = 7  # Filter strength (3-15)
    denoise_color_strength: int = 7  # Color component strength

    # Brightness/Gamma
    auto_brightness: bool = True
    target_brightness: int = 128  # Target mean brightness (0-255)
    gamma_correction: float = 1.0  # Gamma (< 1 = brighter, > 1 = darker)

    # White balance
    auto_white_balance: bool = True

    # License plate enhancement (for vehicles)
    plate_region_boost: bool = True
    plate_region_sharpen_extra: float = 0.5  # Additional sharpening for plate region


class ImageEnhancer:
    """Enhances images for optimal Gemini analysis.

    Uses multiple techniques to improve image quality:
    - Contrast enhancement for low-light scenes
    - Sharpening for motion blur
    - Denoising for compression artifacts
    - Color correction for accurate identification
    """

    def __init__(self, config: Optional[EnhancementConfig] = None):
        """Initialize the image enhancer.

        Args:
            config: Enhancement configuration
        """
        self.config = config or EnhancementConfig()

        # Pre-create CLAHE object for efficiency
        self._clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=(self.config.clahe_grid_size, self.config.clahe_grid_size)
        )

        # Stats
        self._stats = {
            "images_enhanced": 0,
            "clahe_applied": 0,
            "sharpen_applied": 0,
            "denoise_applied": 0,
            "brightness_adjusted": 0,
            "white_balance_applied": 0,
        }

        logger.info(
            f"ImageEnhancer initialized: level={self.config.level.value}, "
            f"clahe={self.config.clahe_enabled}, sharpen={self.config.sharpen_enabled}"
        )

    def enhance(
        self,
        image: np.ndarray,
        class_name: str = "unknown",
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> np.ndarray:
        """Enhance an image for Gemini analysis.

        Args:
            image: Input image (BGR numpy array)
            class_name: Object class for class-specific enhancements
            bbox: Original bbox in full frame (for context, may be None for crops)

        Returns:
            Enhanced image (BGR numpy array)
        """
        if not self.config.enabled or self.config.level == EnhancementLevel.NONE:
            return image

        if image is None or image.size == 0:
            return image

        try:
            enhanced = image.copy()

            # Apply enhancements based on level
            if self.config.level == EnhancementLevel.LIGHT:
                enhanced = self._enhance_light(enhanced, class_name)
            elif self.config.level == EnhancementLevel.MODERATE:
                enhanced = self._enhance_moderate(enhanced, class_name)
            elif self.config.level == EnhancementLevel.AGGRESSIVE:
                enhanced = self._enhance_aggressive(enhanced, class_name)

            self._stats["images_enhanced"] += 1
            return enhanced

        except Exception as e:
            logger.warning(f"Image enhancement failed: {e}")
            return image

    def _enhance_light(self, image: np.ndarray, class_name: str) -> np.ndarray:
        """Light enhancement - subtle improvements."""
        # Just auto brightness and light sharpening
        if self.config.auto_brightness:
            image = self._adjust_brightness(image, strength=0.5)

        if self.config.sharpen_enabled:
            image = self._sharpen(image, amount=0.5)

        return image

    def _enhance_moderate(self, image: np.ndarray, class_name: str) -> np.ndarray:
        """Moderate enhancement - balanced quality improvement."""
        # 1. Denoise first (removes noise before sharpening amplifies it)
        if self.config.denoise_enabled:
            image = self._denoise(image, strength=0.7)

        # 2. Auto white balance
        if self.config.auto_white_balance:
            image = self._white_balance(image)

        # 3. CLAHE for contrast
        if self.config.clahe_enabled:
            image = self._apply_clahe(image)

        # 4. Auto brightness
        if self.config.auto_brightness:
            image = self._adjust_brightness(image, strength=0.7)

        # 5. Sharpening
        if self.config.sharpen_enabled:
            image = self._sharpen(image, amount=self.config.sharpen_amount)

        # 6. Extra enhancement for vehicle license plate region
        if class_name in {"car", "truck", "bus", "van", "vehicle"} and self.config.plate_region_boost:
            image = self._enhance_plate_region(image)

        return image

    def _enhance_aggressive(self, image: np.ndarray, class_name: str) -> np.ndarray:
        """Aggressive enhancement - maximum clarity."""
        # 1. Strong denoising
        if self.config.denoise_enabled:
            image = self._denoise(image, strength=1.0)

        # 2. White balance
        if self.config.auto_white_balance:
            image = self._white_balance(image)

        # 3. Strong CLAHE
        if self.config.clahe_enabled:
            # Use stronger clip limit for aggressive mode
            clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            self._stats["clahe_applied"] += 1

        # 4. Strong brightness correction
        if self.config.auto_brightness:
            image = self._adjust_brightness(image, strength=1.0)

        # 5. Strong sharpening
        if self.config.sharpen_enabled:
            image = self._sharpen(image, amount=self.config.sharpen_amount * 1.5)

        # 6. Extra enhancement for vehicles
        if class_name in {"car", "truck", "bus", "van", "vehicle"} and self.config.plate_region_boost:
            image = self._enhance_plate_region(image, strength=1.5)

        return image

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

        Enhances local contrast while preventing over-amplification.
        Works in LAB color space to preserve colors.
        """
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

            # Apply CLAHE to L channel only
            lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])

            # Convert back to BGR
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            self._stats["clahe_applied"] += 1
            return enhanced

        except Exception as e:
            logger.debug(f"CLAHE failed: {e}")
            return image

    def _sharpen(self, image: np.ndarray, amount: float = 1.0) -> np.ndarray:
        """Apply unsharp mask sharpening.

        Uses Gaussian blur to create unsharp mask, then blends with original.
        """
        try:
            if amount <= 0:
                return image

            # Create Gaussian blur
            radius = max(1, int(self.config.sharpen_radius * 2) | 1)  # Must be odd
            blurred = cv2.GaussianBlur(image, (radius, radius), 0)

            # Unsharp mask: original + amount * (original - blurred)
            # Using addWeighted for proper blending
            sharpened = cv2.addWeighted(
                image, 1.0 + amount,
                blurred, -amount,
                0
            )

            # Clip to valid range
            sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

            self._stats["sharpen_applied"] += 1
            return sharpened

        except Exception as e:
            logger.debug(f"Sharpening failed: {e}")
            return image

    def _denoise(self, image: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """Apply denoising using fast bilateral filter.

        Bilateral filter preserves edges while smoothing noise.
        Faster than Non-local Means for real-time use.
        """
        try:
            if strength <= 0:
                return image

            # Scale denoise strength
            h = int(self.config.denoise_strength * strength)
            h_color = int(self.config.denoise_color_strength * strength)

            # Use bilateral filter (faster than fastNlMeans)
            # d=9 is the diameter, sigmaColor and sigmaSpace control smoothing
            denoised = cv2.bilateralFilter(
                image,
                d=9,
                sigmaColor=h_color * 10,
                sigmaSpace=h * 10
            )

            self._stats["denoise_applied"] += 1
            return denoised

        except Exception as e:
            logger.debug(f"Denoising failed: {e}")
            return image

    def _adjust_brightness(self, image: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """Auto-adjust brightness to target level.

        Uses gamma correction to adjust brightness while preserving contrast.
        """
        try:
            # Calculate current mean brightness
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            current_brightness = np.mean(gray)

            # Calculate needed gamma correction
            if current_brightness < 1:
                current_brightness = 1

            target = self.config.target_brightness
            # Blend with 1.0 based on strength
            gamma = (target / current_brightness) ** 0.5
            gamma = 1.0 + (gamma - 1.0) * strength
            gamma = max(0.5, min(2.0, gamma))  # Clamp to reasonable range

            if abs(gamma - 1.0) < 0.05:
                return image  # No adjustment needed

            # Apply gamma correction
            inv_gamma = 1.0 / gamma
            table = np.array([
                ((i / 255.0) ** inv_gamma) * 255
                for i in np.arange(0, 256)
            ]).astype("uint8")

            corrected = cv2.LUT(image, table)

            self._stats["brightness_adjusted"] += 1
            return corrected

        except Exception as e:
            logger.debug(f"Brightness adjustment failed: {e}")
            return image

    def _white_balance(self, image: np.ndarray) -> np.ndarray:
        """Apply automatic white balance using gray world assumption.

        Adjusts color channels so their means are equal (gray world).
        """
        try:
            # Calculate mean of each channel
            b, g, r = cv2.split(image)
            b_mean, g_mean, r_mean = np.mean(b), np.mean(g), np.mean(r)

            # Gray world assumption: all channels should have same mean
            gray_mean = (b_mean + g_mean + r_mean) / 3

            if b_mean < 1 or g_mean < 1 or r_mean < 1:
                return image

            # Scale factors
            b_scale = gray_mean / b_mean
            g_scale = gray_mean / g_mean
            r_scale = gray_mean / r_mean

            # Limit scaling to prevent over-correction
            b_scale = max(0.8, min(1.2, b_scale))
            g_scale = max(0.8, min(1.2, g_scale))
            r_scale = max(0.8, min(1.2, r_scale))

            # Apply scaling
            b = np.clip(b * b_scale, 0, 255).astype(np.uint8)
            g = np.clip(g * g_scale, 0, 255).astype(np.uint8)
            r = np.clip(r * r_scale, 0, 255).astype(np.uint8)

            balanced = cv2.merge([b, g, r])

            self._stats["white_balance_applied"] += 1
            return balanced

        except Exception as e:
            logger.debug(f"White balance failed: {e}")
            return image

    def _enhance_plate_region(self, image: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """Extra enhancement for license plate region (bottom 40% of vehicle).

        License plates are typically in the lower portion of vehicle images.
        We apply extra sharpening and contrast to this region.
        """
        try:
            h, w = image.shape[:2]

            # License plate region: bottom 40% of image
            plate_y_start = int(h * 0.6)
            plate_region = image[plate_y_start:, :].copy()

            # Extra sharpening for plate region
            extra_amount = self.config.plate_region_sharpen_extra * strength
            sharpened_plate = self._sharpen(plate_region, amount=extra_amount)

            # Extra CLAHE for plate region
            lab = cv2.cvtColor(sharpened_plate, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced_plate = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            # Replace plate region in original
            result = image.copy()
            result[plate_y_start:, :] = enhanced_plate

            return result

        except Exception as e:
            logger.debug(f"Plate region enhancement failed: {e}")
            return image

    def enhance_for_display(
        self,
        image: np.ndarray,
        max_size: int = 400,
        jpeg_quality: int = 90,
    ) -> Tuple[np.ndarray, Optional[str]]:
        """Enhance image and prepare for frontend display.

        Args:
            image: Input image
            max_size: Maximum dimension for output
            jpeg_quality: JPEG quality for base64 encoding

        Returns:
            Tuple of (enhanced image, base64 encoded JPEG or None)
        """
        import base64

        if image is None or image.size == 0:
            return image, None

        try:
            # Enhance the image
            enhanced = self.enhance(image)

            # Resize if needed
            h, w = enhanced.shape[:2]
            if w > max_size or h > max_size:
                scale = max_size / max(w, h)
                new_w, new_h = int(w * scale), int(h * scale)
                enhanced = cv2.resize(enhanced, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Encode to base64 JPEG
            _, buffer = cv2.imencode('.jpg', enhanced, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            base64_str = base64.b64encode(buffer).decode('utf-8')

            return enhanced, base64_str

        except Exception as e:
            logger.warning(f"Display enhancement failed: {e}")
            return image, None

    def get_stats(self) -> dict:
        """Get enhancement statistics."""
        return self._stats.copy()

    def set_level(self, level: EnhancementLevel):
        """Change enhancement level dynamically."""
        self.config.level = level
        logger.info(f"Enhancement level changed to: {level.value}")


# Global instance
_image_enhancer: Optional[ImageEnhancer] = None


def get_image_enhancer() -> ImageEnhancer:
    """Get or create the global image enhancer instance."""
    global _image_enhancer
    if _image_enhancer is None:
        _image_enhancer = ImageEnhancer()
    return _image_enhancer


def init_image_enhancer(config: Optional[EnhancementConfig] = None) -> ImageEnhancer:
    """Initialize the global image enhancer with custom config."""
    global _image_enhancer
    _image_enhancer = ImageEnhancer(config)
    return _image_enhancer
