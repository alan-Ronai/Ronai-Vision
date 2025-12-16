"""API routes for video recording control."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import logging

from services.recorder import get_event_recorder

logger = logging.getLogger(__name__)

router = APIRouter()


class StartRecordingRequest(BaseModel):
    """Request to start recording."""

    camera_id: str
    trigger_reason: Optional[str] = None
    include_buffer: bool = True


class StartRecordingResponse(BaseModel):
    """Response for start recording."""

    session_id: str
    camera_id: str
    output_path: str
    message: str


class StopRecordingResponse(BaseModel):
    """Response for stop recording."""

    session_id: str
    output_path: str
    message: str


class RecordingSessionInfo(BaseModel):
    """Recording session information."""

    session_id: str
    camera_id: str
    trigger_type: str
    trigger_reason: Optional[str]
    track_id: Optional[int]
    duration: float
    frame_count: int
    output_path: str


@router.post("/start", response_model=StartRecordingResponse)
async def start_recording(request: StartRecordingRequest):
    """Start manual video recording for a camera.

    Args:
        request: Recording start request

    Returns:
        Session information
    """
    try:
        recorder = get_event_recorder()

        session_id = recorder.start_recording(
            camera_id=request.camera_id,
            trigger_type="manual",
            trigger_reason=request.trigger_reason,
            include_buffer=request.include_buffer,
        )

        if session_id is None:
            raise HTTPException(status_code=500, detail="Failed to start recording")

        # Get session info
        sessions = recorder.get_active_sessions()
        session = next((s for s in sessions if s["session_id"] == session_id), None)

        if not session:
            raise HTTPException(status_code=500, detail="Session created but not found")

        return StartRecordingResponse(
            session_id=session_id,
            camera_id=request.camera_id,
            output_path=session["output_path"],
            message=f"Recording started for camera {request.camera_id}",
        )

    except Exception as e:
        logger.error(f"Failed to start recording: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop/{session_id}", response_model=StopRecordingResponse)
async def stop_recording(session_id: str):
    """Stop an active recording session.

    Args:
        session_id: Session ID to stop

    Returns:
        Stop confirmation with output path
    """
    try:
        recorder = get_event_recorder()

        output_path = recorder.stop_recording(session_id)

        if output_path is None:
            raise HTTPException(status_code=404, detail="Session not found")

        return StopRecordingResponse(
            session_id=session_id,
            output_path=output_path,
            message=f"Recording stopped: {output_path}",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to stop recording: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions", response_model=List[RecordingSessionInfo])
async def get_active_sessions(camera_id: Optional[str] = None):
    """Get list of active recording sessions.

    Args:
        camera_id: Optional filter by camera

    Returns:
        List of active sessions
    """
    try:
        recorder = get_event_recorder()
        sessions = recorder.get_active_sessions(camera_id=camera_id)

        return [RecordingSessionInfo(**session) for session in sessions]

    except Exception as e:
        logger.error(f"Failed to get sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}", response_model=RecordingSessionInfo)
async def get_session_info(session_id: str):
    """Get information about a specific recording session.

    Args:
        session_id: Session ID

    Returns:
        Session information
    """
    try:
        recorder = get_event_recorder()
        sessions = recorder.get_active_sessions()

        session = next((s for s in sessions if s["session_id"] == session_id), None)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        return RecordingSessionInfo(**session)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cleanup")
async def cleanup_old_sessions(max_age: int = 3600):
    """Clean up old inactive sessions from memory.

    Args:
        max_age: Maximum age for inactive sessions (seconds)

    Returns:
        Cleanup confirmation
    """
    try:
        recorder = get_event_recorder()
        recorder.cleanup_old_sessions(max_age=max_age)

        return {"message": f"Cleaned up sessions older than {max_age} seconds"}

    except Exception as e:
        logger.error(f"Failed to cleanup sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))
