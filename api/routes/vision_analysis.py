"""
API routes for AI vision analysis (Gemini).

Provides endpoints for:
- Triggering Gemini analysis on tracked objects
- Querying analysis results
- Managing analysis history per track
"""

import time
import logging
from typing import Optional, Dict
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from services.tracker.metadata_manager import get_metadata_manager
from services.gemini.analyzer import GeminiAnalyzer
from services.logging.operational_logger import get_operational_logger

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/vision", tags=["vision"])

# Global analyzer instance (lazy-loaded)
_analyzer: Optional[GeminiAnalyzer] = None


def get_analyzer() -> Optional[GeminiAnalyzer]:
    """Get or initialize Gemini analyzer."""
    global _analyzer
    if _analyzer is None:
        try:
            _analyzer = GeminiAnalyzer()
        except Exception as e:
            logger.error(f"Failed to initialize Gemini analyzer: {e}")
            return None
    return _analyzer


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================


class AnalysisRequest(BaseModel):
    """Request to analyze a track."""

    track_id: int
    camera_id: Optional[str] = None
    force: bool = False  # Force analysis even if already analyzed


class AnalysisResponse(BaseModel):
    """Response from analysis."""

    track_id: int
    status: str  # "pending", "completed", "error"
    analysis: Optional[Dict] = None
    error: Optional[str] = None
    timestamp: float


# ============================================================================
# ANALYSIS DEDUPLICATION LOGIC
# ============================================================================


def _should_analyze_track(track_metadata: Dict, class_name: str) -> bool:
    """Determine if track should be analyzed (deduplication logic).

    Max 2 analyses per track:
    - First analysis: when first detected
    - Second analysis: 3-5 seconds later for verification

    Args:
        track_metadata: Track metadata dictionary
        class_name: Class name (person, car, etc)

    Returns:
        True if should analyze, False if already analyzed enough times
    """
    # Get analysis history
    analyses = track_metadata.get("gemini_analyses", [])

    # Max 2 analyses per track
    if len(analyses) >= 2:
        return False

    # First analysis: always do it
    if len(analyses) == 0:
        return True

    # Second analysis: only if >3 seconds since first
    if len(analyses) == 1:
        first_analysis_time = analyses[0].get("timestamp", 0)
        time_since_first = time.time() - first_analysis_time
        if time_since_first >= 3.0:  # 3 seconds minimum between analyses
            return True

    return False


def _add_analysis_to_track(track_id: int, analysis_result: Dict, class_name: str):
    """Add analysis result to track metadata.

    Args:
        track_id: Track ID
        analysis_result: Analysis result from Gemini
        class_name: Class name (person, car, etc)
    """
    manager = get_metadata_manager()
    metadata = manager.get_track_metadata(track_id)

    if metadata is None:
        logger.warning(f"Track {track_id} not found in metadata manager")
        return

    # Initialize analyses list if needed
    if "gemini_analyses" not in metadata:
        metadata["gemini_analyses"] = []

    # Add analysis with timestamp and class info
    analysis_entry = {
        "timestamp": time.time(),
        "class": class_name,
        "result": analysis_result,
    }
    metadata["gemini_analyses"].append(analysis_entry)

    # Update metadata
    manager.update_track_metadata(track_id, metadata.get("class_id", -1), metadata)

    logger.info(f"Added Gemini analysis to track {track_id}: {class_name}")


# ============================================================================
# BACKGROUND ANALYSIS TASK
# ============================================================================


async def _analyze_track_background(
    track_id: int,
    frame_data: bytes,
    bbox: Optional[tuple],
    class_name: str,
):
    """Background task to analyze a track.

    Args:
        track_id: Track ID
        frame_data: JPEG frame data (bytes)
        bbox: Bounding box (x1, y1, x2, y2)
        class_name: Class name (person, car)
    """
    try:
        analyzer = get_analyzer()
        if analyzer is None:
            logger.error("Gemini analyzer not available")
            return

        # Convert JPEG bytes to numpy array for analysis
        import cv2
        import numpy as np

        nparr = np.frombuffer(frame_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            logger.error("Failed to decode frame for analysis")
            return

        # Convert bbox from tuple to numpy array if provided
        bbox_arr = None
        if bbox:
            bbox_arr = np.array(bbox)

        # Call appropriate analyzer
        if class_name.lower() == "person":
            result = analyzer.analyze_person(image, bbox_arr)
        elif class_name.lower() == "car":
            result = analyzer.analyze_car(image, bbox_arr)
        else:
            logger.warning(f"Unknown class for analysis: {class_name}")
            return

        # Add to track metadata
        _add_analysis_to_track(track_id, result, class_name)

        # Log to operational logger
        op_logger = get_operational_logger()
        manager = get_metadata_manager()
        md = manager.get_track_metadata(track_id) or {}
        cam_id = md.get("camera_id")
        if cam_id:
            op_logger.log_analysis(
                camera_id=cam_id,
                track_id=track_id,
                class_name=class_name,
                analysis_result=result,
            )

        logger.info(f"Background analysis completed for track {track_id}")

    except Exception as e:
        logger.error(f"Background analysis failed for track {track_id}: {e}")


# ============================================================================
# API ENDPOINTS
# ============================================================================


@router.post("/analyze", response_model=AnalysisResponse)
async def trigger_analysis(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
) -> AnalysisResponse:
    """Trigger Gemini analysis on a track.

    Automatically deduplicates: max 2 analyses per track ID.
    First analysis happens immediately, second analysis can be triggered
    after 3+ seconds for verification.

    Note: This endpoint requires frame data to be provided.
    In production, frames are stored when armed person is detected.

    Args:
        request: Analysis request with track_id
        background_tasks: FastAPI background tasks

    Returns:
        Analysis response with status and results
    """
    manager = get_metadata_manager()
    metadata = manager.get_track_metadata(request.track_id)

    if metadata is None:
        raise HTTPException(
            status_code=404, detail=f"Track {request.track_id} not found"
        )

    class_name = metadata.get("class_name", "unknown")

    # Check deduplication (unless force=True)
    if not request.force and not _should_analyze_track(metadata, class_name):
        analyses = metadata.get("gemini_analyses", [])
        return AnalysisResponse(
            track_id=request.track_id,
            status="skipped",
            analysis=analyses[-1]["result"] if analyses else None,
            error=f"Already analyzed {len(analyses)} times",
            timestamp=time.time(),
        )

    # For now, analysis requires manual frame upload or event-based capture
    # In production, frames are captured during armed person detection
    raise HTTPException(
        status_code=400,
        detail="Frame data not available. Analysis is triggered automatically when armed person is detected.",
    )


@router.get("/track/{track_id}/analyses")
async def get_track_analyses(track_id: int):
    """Get all analyses for a track.

    Args:
        track_id: Track ID

    Returns:
        List of analyses with timestamps and results
    """
    manager = get_metadata_manager()
    metadata = manager.get_track_metadata(track_id)

    if metadata is None:
        raise HTTPException(status_code=404, detail=f"Track {track_id} not found")

    analyses = metadata.get("gemini_analyses", [])

    return {
        "track_id": track_id,
        "total_analyses": len(analyses),
        "analyses": analyses,
    }


@router.get("/status")
async def get_analysis_status():
    """Get Gemini analyzer status.

    Returns:
        Status information
    """
    analyzer = get_analyzer()

    return {
        "available": analyzer is not None,
        "model": analyzer.model_name if analyzer else None,
        "status": "ready" if analyzer else "unavailable",
    }
