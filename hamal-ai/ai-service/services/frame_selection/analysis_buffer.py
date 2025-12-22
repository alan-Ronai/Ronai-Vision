"""Analysis Buffer - Buffers frames per-track for optimal Gemini analysis.

When a new track is created, instead of immediately analyzing the first frame,
we buffer frames until we find an optimal one (or timeout).

This dramatically improves analysis quality for license plates, clothing, etc.
since early frames are often blurry or partially visible.
"""

import time
import threading
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple, Callable, Any
from collections import deque

from .quality_scorer import (
    FrameQualityScorer,
    QualityScoreBreakdown,
    get_quality_scorer,
)

logger = logging.getLogger(__name__)


@dataclass
class BufferConfig:
    """Configuration for analysis buffering."""

    # Buffer timing
    max_buffer_frames: int = 30  # Maximum frames to buffer (~1-2 sec at 15-30fps)
    min_buffer_frames: int = 10  # Minimum frames before analysis allowed
    buffer_timeout_ms: float = 3000.0  # Max time to wait before forcing analysis

    # Quality thresholds
    quality_threshold: float = 0.85  # Analyze immediately if score > this
    min_acceptable_quality: float = 0.30  # Minimum quality to consider

    # Memory management
    max_concurrent_buffers: int = 20  # Maximum tracks being buffered simultaneously
    store_full_frames: bool = False  # Store full frames (False = store crops only)
    max_crop_size: int = 400  # Max dimension for stored crops


@dataclass
class BufferedFrame:
    """A single buffered frame with quality score."""

    frame_index: int  # Global frame index
    timestamp: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    quality_score: float
    score_breakdown: Dict
    # Either full frame or crop depending on config
    frame_data: np.ndarray  # Cropped region or full frame
    is_crop: bool = True  # True if frame_data is a crop


@dataclass
class TrackBuffer:
    """Buffer for a single track's frames."""

    track_id: int
    object_class: str
    camera_id: str
    created_at: float = field(default_factory=time.time)

    # Frame storage (deque for efficient pop from front)
    frames: deque = field(default_factory=lambda: deque(maxlen=50))

    # Best frame tracking
    best_frame_index: int = -1
    best_score: float = 0.0
    best_frame: Optional[BufferedFrame] = None

    # State
    analysis_triggered: bool = False
    analysis_pending: bool = False

    # Stats
    frames_received: int = 0
    frames_dropped: int = 0


class AnalysisBuffer:
    """Manages per-track frame buffering for optimal analysis timing.

    When a new track is detected, instead of immediately sending to Gemini,
    we buffer frames and track quality scores. Analysis is triggered when:
    1. A frame exceeds the quality threshold (early exit)
    2. The buffer is full (max_buffer_frames reached)
    3. Timeout is reached (buffer_timeout_ms)

    The best frame from the buffer is then sent to Gemini.
    """

    def __init__(
        self,
        config: Optional[BufferConfig] = None,
        quality_scorer: Optional[FrameQualityScorer] = None,
    ):
        """Initialize the analysis buffer.

        Args:
            config: Buffer configuration
            quality_scorer: Frame quality scorer instance
        """
        self.config = config or BufferConfig()
        self.quality_scorer = quality_scorer or get_quality_scorer()

        # Per-track buffers
        self._buffers: Dict[int, TrackBuffer] = {}  # track_id -> TrackBuffer
        self._lock = threading.Lock()

        # Callbacks for when analysis is ready
        self._analysis_callbacks: List[Callable] = []

        # Frame index counter
        self._frame_index = 0

        # Stats
        self._stats = {
            "tracks_buffered": 0,
            "frames_buffered": 0,
            "early_triggers": 0,  # Triggered by quality threshold
            "timeout_triggers": 0,  # Triggered by timeout
            "buffer_full_triggers": 0,  # Triggered by buffer full
            "avg_best_score": 0.0,
            "avg_frames_to_trigger": 0.0,
        }
        self._trigger_scores: List[float] = []
        self._trigger_frame_counts: List[int] = []

        logger.info(
            f"AnalysisBuffer initialized: max_frames={self.config.max_buffer_frames}, "
            f"timeout={self.config.buffer_timeout_ms}ms, threshold={self.config.quality_threshold}"
        )

    def register_callback(self, callback: Callable[[int, str, np.ndarray, Tuple, Dict], None]):
        """Register callback for when analysis should be triggered.

        Callback signature: (track_id, class_name, frame, bbox, metadata) -> None

        Args:
            callback: Function to call when optimal frame is ready
        """
        self._analysis_callbacks.append(callback)
        logger.debug(f"Registered analysis callback, total: {len(self._analysis_callbacks)}")

    def start_buffer(
        self,
        track_id: int,
        object_class: str,
        camera_id: str,
    ) -> bool:
        """Start buffering frames for a new track.

        Args:
            track_id: Unique track identifier
            object_class: Object class (e.g., "person", "car")
            camera_id: Camera identifier

        Returns:
            True if buffer was created, False if already exists or at capacity
        """
        with self._lock:
            # Check if already buffering this track
            if track_id in self._buffers:
                return False

            # Check capacity
            if len(self._buffers) >= self.config.max_concurrent_buffers:
                # Find oldest buffer and force trigger
                oldest_id = min(self._buffers.keys(), key=lambda k: self._buffers[k].created_at)
                self._trigger_analysis(oldest_id, "capacity")
                del self._buffers[oldest_id]

            # Create new buffer
            self._buffers[track_id] = TrackBuffer(
                track_id=track_id,
                object_class=object_class,
                camera_id=camera_id,
            )
            self._stats["tracks_buffered"] += 1

            logger.debug(f"Started buffer for track {track_id} ({object_class}) on {camera_id}")
            return True

    def add_frame(
        self,
        track_id: int,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
        confidence: float = 0.5,
    ) -> Optional[Tuple[np.ndarray, Tuple, Dict]]:
        """Add a frame to a track's buffer.

        Args:
            track_id: Track identifier
            frame: Full frame (BGR numpy array)
            bbox: Bounding box (x1, y1, x2, y2)
            confidence: Detection confidence

        Returns:
            If analysis should trigger: (best_frame, best_bbox, metadata)
            Otherwise: None
        """
        with self._lock:
            buffer = self._buffers.get(track_id)
            if buffer is None:
                return None

            if buffer.analysis_triggered:
                return None

            # Increment frame index
            self._frame_index += 1
            buffer.frames_received += 1

            # Score this frame
            score_result = self.quality_scorer.score_frame(
                frame, bbox, buffer.object_class, confidence
            )

            # Store frame (crop or full based on config)
            if self.config.store_full_frames:
                frame_data = frame.copy()
                is_crop = False
            else:
                frame_data = self._crop_frame(frame, bbox)
                is_crop = True

            buffered = BufferedFrame(
                frame_index=self._frame_index,
                timestamp=time.time(),
                bbox=bbox,
                confidence=confidence,
                quality_score=score_result.total,
                score_breakdown=score_result.to_dict(),
                frame_data=frame_data,
                is_crop=is_crop,
            )

            # Update best frame tracking
            if score_result.total > buffer.best_score:
                buffer.best_score = score_result.total
                buffer.best_frame_index = self._frame_index
                buffer.best_frame = buffered

            # Add to buffer
            buffer.frames.append(buffered)
            self._stats["frames_buffered"] += 1

            # Check trigger conditions
            trigger_reason = self._check_trigger_conditions(buffer, score_result.total)

            if trigger_reason:
                return self._trigger_analysis(track_id, trigger_reason)

            return None

    def _crop_frame(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
        margin_percent: float = 0.25,
    ) -> np.ndarray:
        """Crop frame to bbox with margin.

        Args:
            frame: Full frame
            bbox: Bounding box (x1, y1, x2, y2)
            margin_percent: Margin to add around bbox

        Returns:
            Cropped region
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]

        # Calculate bbox dimensions
        bbox_w = x2 - x1
        bbox_h = y2 - y1

        # Add margin
        margin_x = int(bbox_w * margin_percent)
        margin_y = int(bbox_h * margin_percent)

        # Expand with margin, clamped to frame bounds
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(w, x2 + margin_x)
        y2 = min(h, y2 + margin_y)

        if x2 <= x1 or y2 <= y1:
            return frame[0:10, 0:10].copy()  # Fallback tiny crop

        crop = frame[y1:y2, x1:x2].copy()

        # Resize if too large
        crop_h, crop_w = crop.shape[:2]
        max_size = self.config.max_crop_size
        if crop_w > max_size or crop_h > max_size:
            import cv2
            scale = max_size / max(crop_w, crop_h)
            new_w, new_h = int(crop_w * scale), int(crop_h * scale)
            crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

        return crop

    def _check_trigger_conditions(
        self,
        buffer: TrackBuffer,
        current_score: float,
    ) -> Optional[str]:
        """Check if analysis should be triggered.

        Args:
            buffer: Track buffer
            current_score: Current frame's quality score

        Returns:
            Trigger reason string, or None if not triggered
        """
        # Check minimum frames requirement
        if buffer.frames_received < self.config.min_buffer_frames:
            return None

        # Check quality threshold (early exit)
        if current_score >= self.config.quality_threshold:
            return "quality"

        # Check buffer full
        if buffer.frames_received >= self.config.max_buffer_frames:
            return "buffer_full"

        # Check timeout
        elapsed_ms = (time.time() - buffer.created_at) * 1000
        if elapsed_ms >= self.config.buffer_timeout_ms:
            return "timeout"

        return None

    def _trigger_analysis(
        self,
        track_id: int,
        reason: str,
    ) -> Optional[Tuple[np.ndarray, Tuple, Dict]]:
        """Trigger analysis for a track and return best frame.

        Args:
            track_id: Track identifier
            reason: Trigger reason

        Returns:
            (best_frame, best_bbox, metadata) or None
        """
        buffer = self._buffers.get(track_id)
        if buffer is None or buffer.analysis_triggered:
            return None

        buffer.analysis_triggered = True

        # Update stats
        if reason == "quality":
            self._stats["early_triggers"] += 1
        elif reason == "timeout":
            self._stats["timeout_triggers"] += 1
        elif reason == "buffer_full":
            self._stats["buffer_full_triggers"] += 1

        # Get best frame
        best_frame = buffer.best_frame
        if best_frame is None:
            logger.warning(f"No best frame for track {track_id}, using last frame")
            if buffer.frames:
                best_frame = buffer.frames[-1]
            else:
                return None

        # Track stats
        self._trigger_scores.append(best_frame.quality_score)
        self._trigger_frame_counts.append(buffer.frames_received)

        # Update rolling averages
        if len(self._trigger_scores) > 100:
            self._trigger_scores = self._trigger_scores[-100:]
            self._trigger_frame_counts = self._trigger_frame_counts[-100:]

        self._stats["avg_best_score"] = sum(self._trigger_scores) / len(self._trigger_scores)
        self._stats["avg_frames_to_trigger"] = sum(self._trigger_frame_counts) / len(self._trigger_frame_counts)

        logger.info(
            f"Analysis triggered for track {track_id} ({buffer.object_class}): "
            f"reason={reason}, score={best_frame.quality_score:.3f}, "
            f"frames={buffer.frames_received}, camera={buffer.camera_id}"
        )

        # Build metadata
        metadata = {
            "track_id": track_id,
            "object_class": buffer.object_class,
            "camera_id": buffer.camera_id,
            "trigger_reason": reason,
            "quality_score": best_frame.quality_score,
            "score_breakdown": best_frame.score_breakdown,
            "frames_buffered": buffer.frames_received,
            "buffer_time_ms": (time.time() - buffer.created_at) * 1000,
            "is_crop": best_frame.is_crop,  # True if frame_data is pre-cropped
        }

        # Call registered callbacks
        for callback in self._analysis_callbacks:
            try:
                callback(
                    track_id,
                    buffer.object_class,
                    best_frame.frame_data,
                    best_frame.bbox,
                    metadata,
                )
            except Exception as e:
                logger.error(f"Analysis callback error: {e}")

        return (best_frame.frame_data, best_frame.bbox, metadata)

    def force_trigger(self, track_id: int) -> Optional[Tuple[np.ndarray, Tuple, Dict]]:
        """Force trigger analysis for a track.

        Use when track is about to be lost or deleted.

        Args:
            track_id: Track identifier

        Returns:
            (best_frame, best_bbox, metadata) or None
        """
        with self._lock:
            if track_id not in self._buffers:
                return None
            return self._trigger_analysis(track_id, "forced")

    def is_buffering(self, track_id: int) -> bool:
        """Check if a track is currently being buffered.

        Args:
            track_id: Track identifier

        Returns:
            True if buffering
        """
        with self._lock:
            buffer = self._buffers.get(track_id)
            return buffer is not None and not buffer.analysis_triggered

    def has_been_analyzed(self, track_id: int) -> bool:
        """Check if a track's analysis has been triggered.

        Args:
            track_id: Track identifier

        Returns:
            True if analysis was triggered
        """
        with self._lock:
            buffer = self._buffers.get(track_id)
            return buffer is not None and buffer.analysis_triggered

    def cleanup_track(self, track_id: int):
        """Clean up buffer for a deleted track.

        Args:
            track_id: Track identifier
        """
        with self._lock:
            if track_id in self._buffers:
                buffer = self._buffers[track_id]
                # Force trigger if not yet analyzed
                if not buffer.analysis_triggered:
                    self._trigger_analysis(track_id, "cleanup")
                del self._buffers[track_id]

    def get_buffer_status(self, track_id: int) -> Optional[Dict]:
        """Get status of a track's buffer.

        Args:
            track_id: Track identifier

        Returns:
            Status dict or None
        """
        with self._lock:
            buffer = self._buffers.get(track_id)
            if buffer is None:
                return None

            return {
                "track_id": track_id,
                "object_class": buffer.object_class,
                "camera_id": buffer.camera_id,
                "frames_received": buffer.frames_received,
                "best_score": buffer.best_score,
                "analysis_triggered": buffer.analysis_triggered,
                "elapsed_ms": (time.time() - buffer.created_at) * 1000,
            }

    def get_stats(self) -> Dict:
        """Get buffer manager statistics."""
        with self._lock:
            return {
                **self._stats,
                "active_buffers": len([b for b in self._buffers.values() if not b.analysis_triggered]),
                "total_buffers": len(self._buffers),
                "config": {
                    "max_buffer_frames": self.config.max_buffer_frames,
                    "min_buffer_frames": self.config.min_buffer_frames,
                    "buffer_timeout_ms": self.config.buffer_timeout_ms,
                    "quality_threshold": self.config.quality_threshold,
                    "max_concurrent_buffers": self.config.max_concurrent_buffers,
                }
            }

    def clear(self):
        """Clear all buffers."""
        with self._lock:
            # Force trigger all pending analyses
            for track_id in list(self._buffers.keys()):
                buffer = self._buffers[track_id]
                if not buffer.analysis_triggered:
                    self._trigger_analysis(track_id, "clear")

            self._buffers.clear()
            logger.info("Analysis buffer cleared")


# Global instance
_analysis_buffer: Optional[AnalysisBuffer] = None


def get_analysis_buffer() -> AnalysisBuffer:
    """Get or create the global analysis buffer instance."""
    global _analysis_buffer
    if _analysis_buffer is None:
        _analysis_buffer = AnalysisBuffer()
    return _analysis_buffer


def init_analysis_buffer(
    config: Optional[BufferConfig] = None,
    quality_scorer: Optional[FrameQualityScorer] = None,
) -> AnalysisBuffer:
    """Initialize the global analysis buffer with custom config."""
    global _analysis_buffer
    _analysis_buffer = AnalysisBuffer(config, quality_scorer)
    return _analysis_buffer
