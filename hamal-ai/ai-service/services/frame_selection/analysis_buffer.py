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
from collections import deque, OrderedDict

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
    max_crop_size: int = 600  # Max dimension for stored crops (increased for better quality)


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
        # Track IDs that have been analyzed (for has_been_analyzed check)
        # Use OrderedDict to maintain insertion order for proper LRU eviction
        self._analyzed_tracks: OrderedDict = OrderedDict()  # track_id -> timestamp
        self._max_analyzed_history = 1000  # Max number of analyzed track IDs to remember

        # GIDs that have been analyzed (persists across track_id changes)
        # When a vehicle returns and gets a new track_id but same GID, we skip re-analysis
        self._analyzed_gids: OrderedDict = OrderedDict()  # gid -> timestamp
        self._max_analyzed_gids = 5000  # Max GIDs to remember (larger than tracks)

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
                # Note: _trigger_analysis already deletes the buffer, no need to delete again

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
        object_class: Optional[str] = None,
    ) -> Optional[Tuple[np.ndarray, Tuple, Dict]]:
        """Add a frame to a track's buffer.

        Args:
            track_id: Track identifier
            frame: Full frame (BGR numpy array)
            bbox: Bounding box (x1, y1, x2, y2)
            confidence: Detection confidence
            object_class: Object class (e.g., "person", "car") for consistency check

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

            # CLASS CONSISTENCY CHECK: Verify object class matches buffer's class
            # This prevents buffering wrong object type if tracker misassigns track_id
            if object_class is not None:
                # Normalize class names for comparison
                # "car", "truck", "bus", "motorcycle", "bicycle" are all "vehicle" for buffering purposes
                vehicle_classes = {"car", "truck", "bus", "motorcycle", "bicycle", "vehicle"}
                expected_type = "vehicle" if buffer.object_class in vehicle_classes else buffer.object_class
                current_type = "vehicle" if object_class in vehicle_classes else object_class

                if expected_type != current_type:
                    logger.warning(
                        f"CLASS MISMATCH for track {track_id}: buffer expects {buffer.object_class} "
                        f"({expected_type}) but got {object_class} ({current_type}). "
                        f"Resetting buffer."
                    )
                    # Reset the buffer with the new class
                    # This handles cases where tracker reassigned track to different object type
                    old_frames = buffer.frames_received
                    camera_id = buffer.camera_id
                    del self._buffers[track_id]
                    self._buffers[track_id] = TrackBuffer(
                        track_id=track_id,
                        object_class=object_class,
                        camera_id=camera_id,
                    )
                    buffer = self._buffers[track_id]
                    logger.info(
                        f"Buffer reset for track {track_id}: was {expected_type} ({old_frames} frames), "
                        f"now {current_type}"
                    )

            # Increment frame index
            self._frame_index += 1
            buffer.frames_received += 1

            # CPU OPTIMIZATION: Skip quality scoring every other frame if we already have a decent frame
            # This reduces CPU load by ~50% for quality scoring operations
            skip_scoring = (
                buffer.best_score >= 0.5 and  # Already have a decent frame
                buffer.frames_received % 2 == 0  # Skip every other frame
            )

            if skip_scoring:
                # Just check timeout without expensive scoring
                elapsed_ms = (time.time() - buffer.created_at) * 1000
                if elapsed_ms >= self.config.buffer_timeout_ms:
                    logger.info(f"Trigger: timeout for track {buffer.track_id} after {elapsed_ms:.0f}ms, {buffer.frames_received} frames (skipped scoring)")
                    return self._trigger_analysis(track_id, "timeout")
                return None

            # Score this frame (CPU intensive - Laplacian, Sobel, histogram)
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
        margin_percent: float = 0.60,  # Increased from 0.25 to avoid cutting off heads/feet
    ) -> np.ndarray:
        """Crop frame to bbox with margin.

        IMPORTANT: Use generous margin (60%) to ensure full person/vehicle is captured.
        This crop will be used directly for analysis and display - no further cropping needed.

        Args:
            frame: Full frame
            bbox: Bounding box (x1, y1, x2, y2)
            margin_percent: Margin to add around bbox (default 60%)

        Returns:
            Cropped region with sufficient context around the subject
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
        # CRITICAL: Check timeout FIRST - timeout should fire regardless of frame count
        # This ensures analysis happens even if the object leaves quickly or frames are sparse
        elapsed_ms = (time.time() - buffer.created_at) * 1000
        if elapsed_ms >= self.config.buffer_timeout_ms:
            # Must have at least 1 frame to trigger
            if buffer.frames_received >= 1:
                logger.info(f"Trigger: timeout for track {buffer.track_id} after {elapsed_ms:.0f}ms, {buffer.frames_received} frames")
                return "timeout"
            else:
                logger.debug(f"Timeout reached but no frames for track {buffer.track_id}")

        # Check minimum frames requirement for other triggers
        if buffer.frames_received < self.config.min_buffer_frames:
            return None

        # Check quality threshold (early exit)
        if current_score >= self.config.quality_threshold:
            return "quality"

        # Check buffer full
        if buffer.frames_received >= self.config.max_buffer_frames:
            return "buffer_full"

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

        # CRITICAL: Clear frame data from buffer to prevent memory leak
        # Keep only minimal metadata, not the heavy numpy arrays
        buffer.frames.clear()
        if buffer.best_frame:
            # Keep a reference to best_frame for return, but clear stored reference after
            pass  # best_frame is already extracted above

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

        # Store result for return before clearing
        result = (best_frame.frame_data, best_frame.bbox, metadata)

        # CRITICAL: Clear the best_frame reference to free memory
        buffer.best_frame = None

        # Delete the buffer entirely after analysis to prevent memory buildup
        # The track has been analyzed, we don't need to keep its data
        del self._buffers[track_id]

        # Remember that this track was analyzed (for has_been_analyzed)
        # Store with timestamp for potential future use
        self._analyzed_tracks[track_id] = time.time()

        # Limit the size of analyzed tracks history to prevent unbounded growth
        # Remove oldest entries (OrderedDict maintains insertion order)
        while len(self._analyzed_tracks) > self._max_analyzed_history:
            # popitem(last=False) removes the oldest (first inserted) item
            self._analyzed_tracks.popitem(last=False)

        logger.debug(f"Buffer cleaned up for track {track_id}")

        return result

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

    def check_stale_buffers(self) -> List[Tuple[int, np.ndarray, Tuple, Dict]]:
        """Check all buffers for timeout and trigger stale ones.

        CRITICAL: This must be called periodically to handle buffers where the
        tracked object has disappeared. Without this, buffers for lost tracks
        would never trigger their timeout.

        Returns:
            List of (track_id, best_frame, best_bbox, metadata) for triggered buffers
        """
        triggered = []
        now = time.time()

        with self._lock:
            # Collect stale track_ids (can't modify dict during iteration)
            stale_track_ids = []
            for track_id, buffer in self._buffers.items():
                if buffer.analysis_triggered:
                    continue

                elapsed_ms = (now - buffer.created_at) * 1000
                if elapsed_ms >= self.config.buffer_timeout_ms and buffer.frames_received >= 1:
                    stale_track_ids.append(track_id)

            # Trigger analysis for stale buffers
            for track_id in stale_track_ids:
                logger.info(
                    f"Stale buffer timeout: track {track_id} "
                    f"(buffer had no new frames, forcing analysis)"
                )
                result = self._trigger_analysis(track_id, "stale_timeout")
                if result:
                    best_frame, best_bbox, metadata = result
                    triggered.append((track_id, best_frame, best_bbox, metadata))

        if triggered:
            logger.info(f"Triggered {len(triggered)} stale buffers")

        return triggered

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

        This checks the local cache of analyzed track IDs. The actual track data
        is stored in the backend (MongoDB/trackedStore), not here.

        Args:
            track_id: Track identifier

        Returns:
            True if analysis was triggered for this track
        """
        with self._lock:
            # Check if track is in the analyzed dict (buffer was already cleaned up)
            if track_id in self._analyzed_tracks:
                return True
            # Or check if buffer exists and was triggered (shouldn't happen after fix)
            buffer = self._buffers.get(track_id)
            return buffer is not None and buffer.analysis_triggered

    def has_gid_been_analyzed(self, gid: int) -> bool:
        """Check if a GID has been analyzed (persists across track_id changes).

        When a vehicle returns and gets matched to an existing GID from the gallery,
        this check prevents re-analyzing the same object.

        Args:
            gid: Global identifier (extracted from track_id like v_1_5 -> 5)

        Returns:
            True if this GID was already analyzed
        """
        with self._lock:
            return gid in self._analyzed_gids

    def mark_gid_analyzed(self, gid: int):
        """Mark a GID as analyzed.

        Called after successful Gemini analysis to prevent re-analysis
        when the same object returns and matches via gallery.

        Args:
            gid: Global identifier to mark as analyzed
        """
        with self._lock:
            self._analyzed_gids[gid] = time.time()
            # Limit size - evict oldest
            while len(self._analyzed_gids) > self._max_analyzed_gids:
                self._analyzed_gids.popitem(last=False)
            logger.debug(f"Marked GID {gid} as analyzed (total: {len(self._analyzed_gids)})")

    def extract_gid_from_track_id(self, track_id) -> Optional[int]:
        """Extract GID number from track_id string.

        Args:
            track_id: Track identifier like 'v_1_5' or 'p_2_10'

        Returns:
            GID number (e.g., 5) or None if not extractable
        """
        import re
        if isinstance(track_id, str):
            match = re.search(r'[vpt]_\d+_(\d+)', track_id)
            if match:
                return int(match.group(1))
        return None

    def get_all_analyzed_gids(self) -> set:
        """Get all GIDs that have been analyzed.

        Returns:
            Set of GID integers that have been analyzed by Gemini
        """
        with self._lock:
            return set(self._analyzed_gids.keys())

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

    def cleanup_orphan_buffers(self, active_track_ids: set) -> int:
        """Clean up buffers for tracks that no longer exist.

        Prevents memory leak from buffers accumulating for deleted tracks.

        Args:
            active_track_ids: Set of currently active track IDs

        Returns:
            Number of orphan buffers cleaned up
        """
        with self._lock:
            orphan_ids = []
            for track_id in self._buffers:
                if track_id not in active_track_ids:
                    orphan_ids.append(track_id)

            for track_id in orphan_ids:
                buffer = self._buffers.get(track_id)
                if buffer is None:
                    continue  # Already deleted (race condition or previous cleanup)
                # Force trigger if not yet analyzed (gives it one last chance)
                if not buffer.analysis_triggered:
                    self._trigger_analysis(track_id, "orphan_cleanup")
                # Note: _trigger_analysis already deletes the buffer, but check just in case
                if track_id in self._buffers:
                    del self._buffers[track_id]

            if orphan_ids:
                logger.debug(f"Cleaned up {len(orphan_ids)} orphan analysis buffers")

            return len(orphan_ids)

    def clear(self, clear_analyzed_gids: bool = False):
        """Clear all buffers and reset state.

        Note: This does NOT affect tracks stored in the backend.
        Only clears local buffering state.

        Args:
            clear_analyzed_gids: If True, also clear the analyzed GIDs tracking.
                               Set to True for full reset (e.g., "clear all" button).
        """
        with self._lock:
            # Force trigger all pending analyses before clearing
            for track_id in list(self._buffers.keys()):
                buffer = self._buffers[track_id]
                if not buffer.analysis_triggered:
                    self._trigger_analysis(track_id, "clear")

            self._buffers.clear()
            self._analyzed_tracks.clear()

            if clear_analyzed_gids:
                self._analyzed_gids.clear()
                logger.info("Analysis buffer fully cleared (including analyzed GIDs)")
            else:
                logger.info("Analysis buffer cleared (backend track data preserved)")


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
