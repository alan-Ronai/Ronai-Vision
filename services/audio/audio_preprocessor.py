"""Audio preprocessing utilities for optimal transcription quality.

Provides audio normalization, amplification, and enhancement to improve
transcription accuracy, especially for quiet or low-volume audio.
"""

import numpy as np
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """Preprocess audio for optimal transcription quality.

    Features:
    - Volume normalization (peak/RMS)
    - Amplification with clipping prevention
    - Noise gate for very quiet sections
    - DC offset removal
    """

    def __init__(
        self,
        target_peak: float = -3.0,  # dB - target peak level (higher = louder)
        target_rms: float = -20.0,  # dB - target RMS level for normalization
        normalization_mode: str = "rms",  # "peak" or "rms"
        noise_gate_threshold: float = -60.0,  # dB - silence below this
        apply_noise_gate: bool = False,  # Usually not needed with VAD in transcriber
        remove_dc_offset: bool = True,
        safety_headroom: float = 0.95,  # Prevent clipping (0.0-1.0)
    ):
        """Initialize audio preprocessor.

        Args:
            target_peak: Target peak level in dB (-3 to 0 dB recommended)
            target_rms: Target RMS level in dB (-20 to -10 dB recommended)
            normalization_mode: "peak" for peak normalization, "rms" for RMS normalization
                - "peak": Maximizes volume without clipping (good for general use)
                - "rms": Normalizes average loudness (better for consistent volume)
            noise_gate_threshold: Threshold in dB below which to silence audio
            apply_noise_gate: Whether to apply noise gate (usually false with VAD)
            remove_dc_offset: Remove DC bias from audio
            safety_headroom: Safety factor to prevent clipping (0.95 = 95% of max)
        """
        self.target_peak_db = target_peak
        self.target_rms_db = target_rms
        self.normalization_mode = normalization_mode
        self.noise_gate_threshold_db = noise_gate_threshold
        self.apply_noise_gate = apply_noise_gate
        self.remove_dc_offset = remove_dc_offset
        self.safety_headroom = min(max(safety_headroom, 0.1), 1.0)

        logger.info(f"AudioPreprocessor initialized:")
        logger.info(
            f"  Normalization: {normalization_mode} (target: {target_rms if normalization_mode == 'rms' else target_peak} dB)"
        )
        logger.info(f"  Noise gate: {'enabled' if apply_noise_gate else 'disabled'}")
        logger.info(
            f"  DC offset removal: {'enabled' if remove_dc_offset else 'disabled'}"
        )

    def process(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Preprocess audio for optimal transcription.

        Args:
            audio: Audio samples (float32, -1.0 to 1.0 range)
            sample_rate: Sample rate in Hz

        Returns:
            Preprocessed audio with improved volume and quality
        """
        if audio is None or len(audio) == 0:
            return audio

        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Make a copy to avoid modifying original
        processed = audio.copy()

        # Step 1: Remove DC offset (constant bias)
        if self.remove_dc_offset:
            processed = self._remove_dc_offset(processed)

        # Step 2: Apply noise gate if enabled
        if self.apply_noise_gate:
            processed = self._apply_noise_gate(processed)

        # Step 3: Normalize volume
        if self.normalization_mode == "rms":
            processed = self._normalize_rms(processed)
        else:  # peak
            processed = self._normalize_peak(processed)

        # Step 4: Ensure no clipping
        processed = np.clip(processed, -1.0, 1.0)

        # Log statistics
        original_rms = self._compute_rms_db(audio)
        processed_rms = self._compute_rms_db(processed)
        original_peak = self._compute_peak_db(audio)
        processed_peak = self._compute_peak_db(processed)

        logger.debug(f"Audio preprocessing:")
        logger.debug(
            f"  Original: RMS={original_rms:.1f}dB, Peak={original_peak:.1f}dB"
        )
        logger.debug(
            f"  Processed: RMS={processed_rms:.1f}dB, Peak={processed_peak:.1f}dB"
        )
        logger.debug(f"  Gain applied: {processed_rms - original_rms:.1f}dB")

        return processed

    def _remove_dc_offset(self, audio: np.ndarray) -> np.ndarray:
        """Remove DC offset (constant bias) from audio."""
        mean = np.mean(audio)
        if abs(mean) > 0.001:  # Only remove if significant
            logger.debug(f"Removing DC offset: {mean:.4f}")
            return audio - mean
        return audio

    def _apply_noise_gate(self, audio: np.ndarray) -> np.ndarray:
        """Apply noise gate to silence very quiet sections."""
        # Compute envelope (smoothed amplitude)
        window_size = 2048
        envelope = np.convolve(
            np.abs(audio), np.ones(window_size) / window_size, mode="same"
        )

        # Threshold in linear scale
        threshold = self._db_to_linear(self.noise_gate_threshold_db)

        # Create gate mask (1 = pass, 0 = silence)
        gate_mask = (envelope > threshold).astype(np.float32)

        # Smooth the gate to avoid clicks
        gate_mask = np.convolve(gate_mask, np.ones(512) / 512, mode="same")

        return audio * gate_mask

    def _normalize_peak(self, audio: np.ndarray) -> np.ndarray:
        """Normalize by peak level."""
        current_peak = np.abs(audio).max()

        if current_peak < 1e-6:  # Silence
            return audio

        # Target peak in linear scale
        target_peak_linear = self._db_to_linear(self.target_peak_db)

        # Calculate gain
        gain = (target_peak_linear * self.safety_headroom) / current_peak

        # Limit gain to prevent excessive amplification
        max_gain = 100.0  # 40 dB max gain
        gain = min(gain, max_gain)

        return audio * gain

    def _normalize_rms(self, audio: np.ndarray) -> np.ndarray:
        """Normalize by RMS (average) level."""
        # Compute RMS of non-silent parts
        rms = np.sqrt(np.mean(audio**2))

        if rms < 1e-6:  # Silence
            return audio

        # Target RMS in linear scale
        target_rms_linear = self._db_to_linear(self.target_rms_db)

        # Calculate gain
        gain = target_rms_linear / rms

        # Limit gain
        max_gain = 100.0  # 40 dB max gain
        gain = min(gain, max_gain)

        # Apply gain but ensure we don't clip
        normalized = audio * gain

        # If we're clipping, scale back
        peak = np.abs(normalized).max()
        if peak > self.safety_headroom:
            normalized *= self.safety_headroom / peak

        return normalized

    def _compute_rms_db(self, audio: np.ndarray) -> float:
        """Compute RMS level in dB."""
        rms = np.sqrt(np.mean(audio**2))
        if rms < 1e-10:
            return -100.0
        return 20 * np.log10(rms)

    def _compute_peak_db(self, audio: np.ndarray) -> float:
        """Compute peak level in dB."""
        peak = np.abs(audio).max()
        if peak < 1e-10:
            return -100.0
        return 20 * np.log10(peak)

    def _db_to_linear(self, db: float) -> float:
        """Convert dB to linear scale."""
        return 10 ** (db / 20.0)

    def get_audio_stats(self, audio: np.ndarray) -> dict:
        """Get audio statistics for debugging.

        Returns:
            Dictionary with RMS, peak, duration, clipping info
        """
        if audio is None or len(audio) == 0:
            return {}

        rms_db = self._compute_rms_db(audio)
        peak_db = self._compute_peak_db(audio)
        peak_linear = np.abs(audio).max()
        clipping_samples = np.sum(np.abs(audio) >= 0.99)
        clipping_pct = (clipping_samples / len(audio)) * 100

        return {
            "rms_db": rms_db,
            "peak_db": peak_db,
            "peak_linear": peak_linear,
            "clipping_samples": clipping_samples,
            "clipping_percentage": clipping_pct,
            "is_clipping": clipping_pct > 0.1,
        }


# Convenience function for simple use
def normalize_audio(
    audio: np.ndarray,
    target_level: float = -20.0,
    mode: str = "rms",
    sample_rate: int = 16000,
) -> np.ndarray:
    """Quick normalization function.

    Args:
        audio: Audio samples (float32)
        target_level: Target level in dB (-20 for RMS, -3 for peak)
        mode: "rms" or "peak"
        sample_rate: Sample rate in Hz

    Returns:
        Normalized audio
    """
    preprocessor = AudioPreprocessor(
        target_rms=target_level if mode == "rms" else -20.0,
        target_peak=target_level if mode == "peak" else -3.0,
        normalization_mode=mode,
    )
    return preprocessor.process(audio, sample_rate)
