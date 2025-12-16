"""
Video recording integration service.

Connects armed person detection to video recording and encoding.
Automatically starts/stops recording sessions and triggers background encoding.
"""

import logging
import time
from typing import Optional, List, Dict
from services.video.encoder import get_video_encoder
from services.tracker.metadata_manager import get_metadata_manager
from services.logging.operational_logger import get_operational_logger

logger = logging.getLogger(__name__)

# Track active recording sessions: track_id -> session_id
_track_recording_sessions: Dict[int, str] = {}
_recording_frames_buffer: Dict[str, List] = {}


def start_recording_for_track(
    track_id: int,
    camera_id: str,
    event_type: str = "armed_person_detected",
    context: Optional[Dict] = None,
) -> str:
    """Start video recording for a track.

    Args:
        track_id: Track ID
        camera_id: Camera ID
        event_type: Event type (e.g., "armed_person_detected")
        context: Additional context (optional)

    Returns:
        Session ID
    """
    encoder = get_video_encoder()

    # Generate unique session ID
    session_id = f"{camera_id}_track{track_id}_{int(time.time() * 1000)}"

    # Start recording session
    encoder.start_recording(
        session_id=session_id,
        camera_id=camera_id,
        event_type=event_type,
    )

    # Track the session
    _track_recording_sessions[track_id] = session_id
    _recording_frames_buffer[session_id] = []

    logger.info(f"Recording started for track {track_id}: {session_id}")

    return session_id


def add_frame_to_recording(
    track_id: int,
    frame,
    visible_track_ids: Optional[List[int]] = None,
) -> bool:
    """Add frame to the recording for a track.

    Args:
        track_id: Primary track ID
        frame: Video frame
        visible_track_ids: All track IDs visible in this frame

    Returns:
        True if added, False if track not recording
    """
    if track_id not in _track_recording_sessions:
        return False

    session_id = _track_recording_sessions[track_id]
    encoder = get_video_encoder()

    # Add frame to encoder
    success = encoder.add_frame(
        session_id=session_id,
        frame=frame,
        track_ids=visible_track_ids,
    )

    return success


def stop_recording_for_track(
    track_id: int,
    camera_id: str,
) -> bool:
    """Stop recording for a track and queue for encoding.

    Args:
        track_id: Track ID
        camera_id: Camera ID (for logging)

    Returns:
        True if stopped, False if not recording
    """
    if track_id not in _track_recording_sessions:
        return False

    session_id = _track_recording_sessions.pop(track_id)
    encoder = get_video_encoder()

    # Stop recording (triggers background encoding)
    encoder.stop_recording(session_id)

    # Clean up buffer
    _recording_frames_buffer.pop(session_id, None)

    # Get track metadata for logging
    manager = get_metadata_manager()
    metadata = manager.get_track_metadata(track_id)

    op_logger = get_operational_logger()
    op_logger.log_recording(
        camera_id=camera_id,
        action="stopped",
        track_id=track_id,
    )

    logger.info(f"Recording stopped for track {track_id}: {session_id}")

    return True


def is_track_recording(track_id: int) -> bool:
    """Check if a track is currently being recorded.

    Args:
        track_id: Track ID

    Returns:
        True if recording
    """
    return track_id in _track_recording_sessions


def get_recording_session_id(track_id: int) -> Optional[str]:
    """Get the session ID for a track's recording.

    Args:
        track_id: Track ID

    Returns:
        Session ID or None if not recording
    """
    return _track_recording_sessions.get(track_id)


def get_active_recordings(camera_id: Optional[str] = None) -> List[Dict]:
    """Get list of active recording sessions.

    Args:
        camera_id: Filter by camera (optional)

    Returns:
        List of active recording info
    """
    active = []
    for track_id, session_id in _track_recording_sessions.items():
        if camera_id is None or camera_id in session_id:
            manager = get_metadata_manager()
            metadata = manager.get_track_metadata(track_id)

            active.append(
                {
                    "track_id": track_id,
                    "session_id": session_id,
                    "camera_id": camera_id,
                    "class_name": metadata.get("class_name") if metadata else None,
                    "start_time": time.time(),
                }
            )

    return active
