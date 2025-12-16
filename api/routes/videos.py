"""
API routes for video recording and retrieval.

Provides endpoints for:
- Retrieving recorded videos
- Manually starting/stopping recordings
- Listing active recording sessions
- Downloading video files
"""

import logging
import json
from typing import Optional
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from services.video.encoder import get_video_encoder
from services.video.recording_integration import (
    start_recording_for_track,
    stop_recording_for_track,
    is_track_recording,
    get_active_recordings,
)
from services.tracker.metadata_manager import get_metadata_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/videos", tags=["videos"])


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================


class RecordingRequest(BaseModel):
    """Request to start/stop recording."""

    track_id: Optional[int] = None
    camera_id: str
    event_type: Optional[str] = None


class RecordingResponse(BaseModel):
    """Recording response."""

    status: str  # "started", "stopped", "error"
    session_id: Optional[str] = None
    message: str


# ============================================================================
# API ENDPOINTS
# ============================================================================


@router.post("/record/start")
async def start_recording(request: RecordingRequest) -> RecordingResponse:
    """Manually start a recording session.

    Can record a specific track or entire camera feed.

    Args:
        request: Recording request

    Returns:
        Recording response with session ID
    """
    try:
        if request.track_id is None:
            return RecordingResponse(
                status="error",
                message="track_id required",
            )

        session_id = start_recording_for_track(
            track_id=request.track_id,
            camera_id=request.camera_id,
            event_type=request.event_type or "manual",
        )

        return RecordingResponse(
            status="started",
            session_id=session_id,
            message=f"Recording started for track {request.track_id}",
        )

    except Exception as e:
        logger.error(f"Failed to start recording: {e}")
        return RecordingResponse(
            status="error",
            message=str(e),
        )


@router.post("/record/stop")
async def stop_recording(request: RecordingRequest) -> RecordingResponse:
    """Manually stop a recording session.

    Queues video for encoding.

    Args:
        request: Recording request with track_id

    Returns:
        Recording response
    """
    try:
        if request.track_id is None:
            return RecordingResponse(
                status="error",
                message="track_id required",
            )

        success = stop_recording_for_track(
            track_id=request.track_id,
            camera_id=request.camera_id,
        )

        if not success:
            return RecordingResponse(
                status="error",
                message=f"Track {request.track_id} not recording",
            )

        return RecordingResponse(
            status="stopped",
            message=f"Recording stopped for track {request.track_id}",
        )

    except Exception as e:
        logger.error(f"Failed to stop recording: {e}")
        return RecordingResponse(
            status="error",
            message=str(e),
        )


@router.get("/recordings/active")
async def get_active_recordings_endpoint(
    camera_id: Optional[str] = Query(None, description="Filter by camera"),
) -> dict:
    """Get list of active recording sessions.

    Args:
        camera_id: Filter by camera ID (optional)

    Returns:
        List of active recordings
    """
    recordings = get_active_recordings(camera_id)

    return {
        "total": len(recordings),
        "recordings": recordings,
    }


@router.get("/recordings/track/{track_id}")
async def get_track_recordings(
    track_id: int,
) -> dict:
    """Get all recordings for a specific track.

    Args:
        track_id: Track ID

    Returns:
        List of recordings with metadata
    """
    encoder = get_video_encoder()
    manager = get_metadata_manager()

    metadata = manager.get_track_metadata(track_id)
    if metadata is None:
        raise HTTPException(status_code=404, detail=f"Track {track_id} not found")

    # Look for video files for this track
    videos = []
    for video_file in encoder.output_dir.glob("*.mp4"):
        metadata_file = video_file.with_suffix(".json")
        if metadata_file.exists():
            try:
                with open(metadata_file) as f:
                    video_metadata = json.load(f)
                    if track_id in video_metadata.get("track_ids", []):
                        videos.append(
                            {
                                "filename": video_file.name,
                                "path": str(video_file),
                                "size_bytes": video_file.stat().st_size,
                                "metadata": video_metadata,
                            }
                        )
            except Exception as e:
                logger.error(f"Error reading video metadata: {e}")

    return {
        "track_id": track_id,
        "total_videos": len(videos),
        "videos": videos,
    }


@router.get("/videos/download/{video_id}")
async def download_video(video_id: str) -> FileResponse:
    """Download a video file.

    Args:
        video_id: Video filename (without extension)

    Returns:
        Video file
    """
    encoder = get_video_encoder()

    # Sanitize filename
    video_id = video_id.replace("..", "").replace("/", "")
    video_path = encoder.output_dir / f"{video_id}.mp4"

    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")

    return FileResponse(
        path=video_path,
        media_type="video/mp4",
        filename=f"{video_id}.mp4",
    )


@router.get("/videos/list")
async def list_videos(
    camera_id: Optional[str] = Query(None, description="Filter by camera"),
    limit: int = Query(100, description="Max results"),
) -> dict:
    """List recorded videos with metadata.

    Args:
        camera_id: Filter by camera ID
        limit: Maximum results

    Returns:
        List of video files with metadata
    """
    encoder = get_video_encoder()

    videos = []
    for i, video_file in enumerate(encoder.output_dir.glob("*.mp4")):
        if i >= limit:
            break

        metadata_file = video_file.with_suffix(".json")
        metadata = None

        if metadata_file.exists():
            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)
                    if camera_id and metadata.get("camera_id") != camera_id:
                        continue
            except Exception as e:
                logger.error(f"Error reading metadata: {e}")

        videos.append(
            {
                "filename": video_file.name,
                "size_bytes": video_file.stat().st_size,
                "created": video_file.stat().st_ctime,
                "camera_id": metadata.get("camera_id") if metadata else None,
                "event_type": metadata.get("event_type") if metadata else None,
                "track_ids": metadata.get("track_ids") if metadata else [],
                "metadata": metadata,
            }
        )

    return {
        "total": len(videos),
        "limit": limit,
        "camera_id_filter": camera_id,
        "videos": videos,
    }


@router.get("/encoder/status")
async def get_encoder_status() -> dict:
    """Get video encoder status.

    Returns:
        Encoder information
    """
    encoder = get_video_encoder()

    # Count videos in output directory
    video_count = len(list(encoder.output_dir.glob("*.mp4")))

    return {
        "status": "ready",
        "codec": encoder.codec,
        "fps": encoder.fps,
        "bitrate": encoder.bitrate,
        "output_directory": str(encoder.output_dir),
        "total_videos": video_count,
        "queue_size": encoder.encode_queue.qsize(),
    }
