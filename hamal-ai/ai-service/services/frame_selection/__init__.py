"""Frame Selection Module - Optimal frame selection for Gemini analysis.

This module provides intelligent frame selection to ensure the best possible
frames are sent to Gemini for analysis, rather than always using the first frame.

Components:
- FrameQualityScorer: Scores frames based on size, position, sharpness, etc.
- AnalysisBuffer: Buffers frames per-track and selects the best one
- ImageEnhancer: Enhances images before analysis (CLAHE, sharpening, denoising)

Usage:
    from services.frame_selection import (
        get_analysis_buffer,
        get_quality_scorer,
        get_image_enhancer,
        BufferConfig,
        QualityConfig,
        EnhancementConfig,
    )

    # Initialize (usually done once at startup)
    buffer = get_analysis_buffer()
    enhancer = get_image_enhancer()

    # When new track detected
    buffer.start_buffer(track_id=123, object_class="car", camera_id="cam-1")

    # For each frame with this track
    result = buffer.add_frame(track_id=123, frame=frame, bbox=bbox, confidence=0.9)
    if result:
        best_frame, best_bbox, metadata = result
        # Enhance the frame before analysis
        enhanced_frame = enhancer.enhance(best_frame, class_name="car")
        # Send to Gemini for analysis
        analysis = await gemini.analyze_vehicle(enhanced_frame, best_bbox)
"""

from .quality_scorer import (
    FrameQualityScorer,
    QualityConfig,
    QualityScoreBreakdown,
    get_quality_scorer,
    init_quality_scorer,
)

from .analysis_buffer import (
    AnalysisBuffer,
    BufferConfig,
    BufferedFrame,
    TrackBuffer,
    get_analysis_buffer,
    init_analysis_buffer,
)

from .image_enhancer import (
    ImageEnhancer,
    EnhancementConfig,
    EnhancementLevel,
    get_image_enhancer,
    init_image_enhancer,
)

__all__ = [
    # Quality Scorer
    "FrameQualityScorer",
    "QualityConfig",
    "QualityScoreBreakdown",
    "get_quality_scorer",
    "init_quality_scorer",
    # Analysis Buffer
    "AnalysisBuffer",
    "BufferConfig",
    "BufferedFrame",
    "TrackBuffer",
    "get_analysis_buffer",
    "init_analysis_buffer",
    # Image Enhancer
    "ImageEnhancer",
    "EnhancementConfig",
    "EnhancementLevel",
    "get_image_enhancer",
    "init_image_enhancer",
]
