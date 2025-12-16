"""Event-based video recording service.

This module provides event-triggered video recording with:
- Circular frame buffer for pre-event footage
- Automatic recording on events (e.g., armed person detection)
- Manual recording control via API
- Per-camera and per-event recording sessions
"""

import cv2
import time
import threading
import logging
from typing import Optional, Dict, List
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RecordingSession:
    """Active recording session."""

    session_id: str
    camera_id: str
    trigger_type: str  # 'manual' or 'auto'
    trigger_reason: Optional[str]  # e.g., 'armed_person_detected'
    track_id: Optional[int]  # Track ID that triggered the recording
    start_time: float
    video_writer: Optional[cv2.VideoWriter]
    output_path: str
    frame_count: int = 0
    fps: float = 30.0
    is_active: bool = True


@dataclass
class RecorderConfig:
    """Configuration for event recorder."""

    output_dir: str = "output/recordings"
    buffer_size: int = 150  # Pre-event buffer (5 seconds at 30fps)
    max_duration: int = 300  # Max recording duration in seconds
    fps: float = 30.0
    codec: str = "mp4v"  # or 'avc1' for H.264
    resolution: tuple = (1920, 1080)  # Default resolution


class EventRecorder:
    """Event-based video recorder with circular buffer and automatic triggers."""

    def __init__(self, config: Optional[RecorderConfig] = None):
        """Initialize event recorder.

        Args:
            config: Recorder configuration
        """
        self.config = config or RecorderConfig()

        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        # Per-camera circular frame buffers
        self._buffers: Dict[str, deque] = {}

        # Active recording sessions
        self._sessions: Dict[str, RecordingSession] = {}

        # Track-based recording state (track_id -> session_id)
        self._track_recordings: Dict[int, str] = {}

        # Thread lock for thread-safe operations
        self._lock = threading.RLock()

        # Session counter for unique IDs
        self._session_counter = 0

        logger.info(
            f"EventRecorder initialized: buffer={self.config.buffer_size} frames, "
            f"output={self.config.output_dir}"
        )

    def add_frame(self, camera_id: str, frame: np.ndarray, timestamp: Optional[float] = None):
        """Add frame to circular buffer and active recordings.

        Args:
            camera_id: Camera identifier
            frame: Frame to add (H, W, 3) BGR uint8
            timestamp: Optional timestamp (uses current time if None)
        """
        if timestamp is None:
            timestamp = time.time()

        with self._lock:
            # Initialize buffer for this camera if needed
            if camera_id not in self._buffers:
                self._buffers[camera_id] = deque(maxlen=self.config.buffer_size)

            # Add to circular buffer
            self._buffers[camera_id].append((timestamp, frame.copy()))

            # Write to all active sessions for this camera
            for session in self._sessions.values():
                if session.camera_id == camera_id and session.is_active:
                    self._write_frame_to_session(session, frame, timestamp)

    def start_recording(
        self,
        camera_id: str,
        trigger_type: str = "manual",
        trigger_reason: Optional[str] = None,
        track_id: Optional[int] = None,
        include_buffer: bool = True,
    ) -> str:
        """Start a new recording session.

        Args:
            camera_id: Camera to record
            trigger_type: 'manual' or 'auto'
            trigger_reason: Reason for recording (e.g., 'armed_person_detected')
            track_id: Optional track ID that triggered recording
            include_buffer: Include pre-event buffer frames

        Returns:
            Session ID
        """
        with self._lock:
            # Generate session ID
            self._session_counter += 1
            session_id = f"{camera_id}_{int(time.time())}_{self._session_counter}"

            # Create output filename
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            reason_str = f"_{trigger_reason}" if trigger_reason else ""
            track_str = f"_track{track_id}" if track_id is not None else ""
            output_path = (
                f"{self.config.output_dir}/{camera_id}_{timestamp_str}{reason_str}{track_str}.mp4"
            )

            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*self.config.codec)
            video_writer = cv2.VideoWriter(
                output_path,
                fourcc,
                self.config.fps,
                self.config.resolution,
            )

            if not video_writer.isOpened():
                logger.error(f"Failed to open video writer for {output_path}")
                return None

            # Create session
            session = RecordingSession(
                session_id=session_id,
                camera_id=camera_id,
                trigger_type=trigger_type,
                trigger_reason=trigger_reason,
                track_id=track_id,
                start_time=time.time(),
                video_writer=video_writer,
                output_path=output_path,
                fps=self.config.fps,
            )

            self._sessions[session_id] = session

            # Link track to session if provided
            if track_id is not None:
                self._track_recordings[track_id] = session_id

            # Write buffered frames if requested
            if include_buffer and camera_id in self._buffers:
                buffer = list(self._buffers[camera_id])
                for timestamp, frame in buffer:
                    self._write_frame_to_session(session, frame, timestamp)

            logger.info(
                f"Started recording: session={session_id}, camera={camera_id}, "
                f"trigger={trigger_type}, reason={trigger_reason}, track={track_id}, "
                f"buffer_frames={len(buffer) if include_buffer else 0}"
            )

            return session_id

    def stop_recording(self, session_id: str) -> Optional[str]:
        """Stop a recording session.

        Args:
            session_id: Session to stop

        Returns:
            Output video path or None if session not found
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                logger.warning(f"Session not found: {session_id}")
                return None

            if not session.is_active:
                logger.warning(f"Session already stopped: {session_id}")
                return session.output_path

            # Mark as inactive
            session.is_active = False

            # Release video writer
            if session.video_writer:
                session.video_writer.release()

            # Remove track recording link
            if session.track_id is not None:
                self._track_recordings.pop(session.track_id, None)

            duration = time.time() - session.start_time

            logger.info(
                f"Stopped recording: session={session_id}, frames={session.frame_count}, "
                f"duration={duration:.1f}s, output={session.output_path}"
            )

            return session.output_path

    def stop_recording_by_track(self, track_id: int) -> Optional[str]:
        """Stop recording associated with a track ID.

        Args:
            track_id: Track ID

        Returns:
            Output video path or None if no recording found
        """
        with self._lock:
            session_id = self._track_recordings.get(track_id)
            if session_id:
                return self.stop_recording(session_id)
            return None

    def is_track_recording(self, track_id: int) -> bool:
        """Check if a track is currently being recorded.

        Args:
            track_id: Track ID

        Returns:
            True if track has active recording
        """
        with self._lock:
            return track_id in self._track_recordings

    def get_active_sessions(self, camera_id: Optional[str] = None) -> List[Dict]:
        """Get list of active recording sessions.

        Args:
            camera_id: Optional filter by camera

        Returns:
            List of session info dictionaries
        """
        with self._lock:
            sessions = []
            for session in self._sessions.values():
                if not session.is_active:
                    continue
                if camera_id and session.camera_id != camera_id:
                    continue

                duration = time.time() - session.start_time
                sessions.append(
                    {
                        "session_id": session.session_id,
                        "camera_id": session.camera_id,
                        "trigger_type": session.trigger_type,
                        "trigger_reason": session.trigger_reason,
                        "track_id": session.track_id,
                        "duration": duration,
                        "frame_count": session.frame_count,
                        "output_path": session.output_path,
                    }
                )
            return sessions

    def cleanup_old_sessions(self, max_age: int = 3600):
        """Remove old inactive sessions from memory.

        Args:
            max_age: Max age for inactive sessions (seconds)
        """
        with self._lock:
            current_time = time.time()
            to_remove = []

            for session_id, session in self._sessions.items():
                if not session.is_active:
                    age = current_time - session.start_time
                    if age > max_age:
                        to_remove.append(session_id)

            for session_id in to_remove:
                del self._sessions[session_id]

            if to_remove:
                logger.info(f"Cleaned up {len(to_remove)} old sessions")

    def _write_frame_to_session(
        self, session: RecordingSession, frame: np.ndarray, timestamp: float
    ):
        """Write a frame to a recording session (internal).

        Args:
            session: Recording session
            frame: Frame to write
            timestamp: Frame timestamp
        """
        if not session.is_active or not session.video_writer:
            return

        # Check duration limit
        duration = time.time() - session.start_time
        if duration > self.config.max_duration:
            logger.warning(
                f"Session {session.session_id} exceeded max duration ({self.config.max_duration}s), stopping"
            )
            self.stop_recording(session.session_id)
            return

        # Resize frame to match output resolution if needed
        if frame.shape[:2][::-1] != self.config.resolution:
            frame = cv2.resize(frame, self.config.resolution)

        # Write frame
        try:
            session.video_writer.write(frame)
            session.frame_count += 1
        except Exception as e:
            logger.error(f"Failed to write frame to session {session.session_id}: {e}")

    def shutdown(self):
        """Shutdown recorder and release all resources."""
        with self._lock:
            # Stop all active sessions
            active_sessions = [sid for sid, s in self._sessions.items() if s.is_active]
            for session_id in active_sessions:
                self.stop_recording(session_id)

            # Clear buffers
            self._buffers.clear()

            logger.info("EventRecorder shutdown complete")


# Global singleton instance
_event_recorder: Optional[EventRecorder] = None
_recorder_lock = threading.Lock()


def get_event_recorder(config: Optional[RecorderConfig] = None) -> EventRecorder:
    """Get or create global event recorder instance.

    Args:
        config: Optional configuration (only used on first call)

    Returns:
        EventRecorder instance
    """
    global _event_recorder
    with _recorder_lock:
        if _event_recorder is None:
            _event_recorder = EventRecorder(config)
        return _event_recorder


def reset_event_recorder():
    """Reset global event recorder (for testing)."""
    global _event_recorder
    with _recorder_lock:
        if _event_recorder is not None:
            _event_recorder.shutdown()
        _event_recorder = None
