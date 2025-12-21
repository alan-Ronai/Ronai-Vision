"""Recording Manager - Manages video recordings with FFmpeg encoding.

Handles:
- Starting recordings with pre-buffer frames
- Continuous frame capture during recording
- FFmpeg encoding to MP4
- Emitting video events when recording completes
"""

import asyncio
import logging
import os
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Callable
import numpy as np
import cv2
import httpx

from .frame_buffer import get_frame_buffer, BufferedFrame

logger = logging.getLogger(__name__)

# Configuration
RECORDINGS_DIR = os.environ.get("RECORDINGS_DIR", "/tmp/hamal_recordings")
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:3000")


@dataclass
class ActiveRecording:
    """Represents an active recording in progress."""
    recording_id: str
    camera_id: str
    start_time: float
    duration: float  # Total duration in seconds
    pre_buffer: float  # Pre-buffer duration in seconds
    output_path: str
    frames: List[np.ndarray] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    trigger_reason: str = ""
    metadata: Dict = field(default_factory=dict)
    _stop_event: threading.Event = field(default_factory=threading.Event)


class RecordingManager:
    """Manages video recordings across cameras.

    Coordinates with FrameBuffer for pre-buffer frames and handles
    FFmpeg encoding when recordings complete.
    """

    def __init__(
        self,
        recordings_dir: str = RECORDINGS_DIR,
        default_fps: float = 15.0,
        on_recording_complete: Optional[Callable] = None
    ):
        """Initialize recording manager.

        Args:
            recordings_dir: Directory to store recordings
            default_fps: Default FPS for encoding
            on_recording_complete: Callback when recording finishes
        """
        self.recordings_dir = Path(recordings_dir)
        self.recordings_dir.mkdir(parents=True, exist_ok=True)

        self.default_fps = default_fps
        self.on_recording_complete = on_recording_complete

        # Active recordings by camera
        self._recordings: Dict[str, ActiveRecording] = {}
        self._lock = threading.Lock()

        # Background tasks - use thread-safe queue for cross-thread communication
        self._encoding_queue: asyncio.Queue = None  # Will be created in start()
        self._encoder_task: Optional[asyncio.Task] = None
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

        logger.info(f"RecordingManager initialized, output dir: {self.recordings_dir}")

    async def start(self):
        """Start the recording manager background tasks."""
        # Store event loop reference for thread-safe queue access
        self._event_loop = asyncio.get_running_loop()
        self._encoding_queue = asyncio.Queue()
        self._encoder_task = asyncio.create_task(self._encoding_loop())
        logger.info("RecordingManager started")

    async def stop(self):
        """Stop the recording manager."""
        # Stop all active recordings
        with self._lock:
            for recording in list(self._recordings.values()):
                recording._stop_event.set()

        # Stop encoder task
        if self._encoder_task:
            self._encoder_task.cancel()
            try:
                await self._encoder_task
            except asyncio.CancelledError:
                pass

        logger.info("RecordingManager stopped")

    def start_recording(
        self,
        camera_id: str,
        duration: float = 30.0,
        pre_buffer: float = 5.0,
        trigger_reason: str = "",
        metadata: Optional[Dict] = None
    ) -> Optional[str]:
        """Start a new recording for a camera.

        Args:
            camera_id: Camera to record
            duration: Recording duration in seconds (after trigger)
            pre_buffer: Seconds of pre-buffer to include
            trigger_reason: Why the recording was triggered
            metadata: Additional metadata to store

        Returns:
            Recording ID if started, None if already recording
        """
        with self._lock:
            # Check if already recording this camera
            if camera_id in self._recordings:
                logger.warning(f"Camera {camera_id} is already recording")
                return None

            # Generate recording ID and output path
            recording_id = f"{camera_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            output_path = str(self.recordings_dir / f"{recording_id}.mp4")

            # Create recording object
            recording = ActiveRecording(
                recording_id=recording_id,
                camera_id=camera_id,
                start_time=time.time(),
                duration=duration,
                pre_buffer=pre_buffer,
                output_path=output_path,
                trigger_reason=trigger_reason,
                metadata=metadata or {}
            )

            # Get pre-buffer frames
            frame_buffer = get_frame_buffer()
            pre_frames = frame_buffer.get_frames(camera_id, pre_buffer)

            for bf in pre_frames:
                recording.frames.append(bf.frame)
                recording.timestamps.append(bf.timestamp)

            logger.info(
                f"Started recording {recording_id}: camera={camera_id}, "
                f"duration={duration}s, pre_buffer={pre_buffer}s ({len(pre_frames)} frames)"
            )

            self._recordings[camera_id] = recording

            # Start recording thread
            thread = threading.Thread(
                target=self._recording_thread,
                args=(recording,),
                daemon=True
            )
            thread.start()

            return recording_id

    def add_frame(self, camera_id: str, frame: np.ndarray):
        """Add a frame to an active recording.

        Should be called for every frame received while recording is active.

        Args:
            camera_id: Camera identifier
            frame: Video frame
        """
        with self._lock:
            recording = self._recordings.get(camera_id)
            if recording and not recording._stop_event.is_set():
                recording.frames.append(frame.copy())
                recording.timestamps.append(time.time())

    def is_recording(self, camera_id: str) -> bool:
        """Check if a camera is currently recording.

        Args:
            camera_id: Camera identifier

        Returns:
            True if recording
        """
        with self._lock:
            return camera_id in self._recordings

    def stop_recording(self, camera_id: str) -> bool:
        """Stop a recording early.

        Args:
            camera_id: Camera identifier

        Returns:
            True if recording was stopped
        """
        with self._lock:
            recording = self._recordings.get(camera_id)
            if recording:
                recording._stop_event.set()
                return True
            return False

    def _recording_thread(self, recording: ActiveRecording):
        """Background thread that manages a single recording's lifecycle."""
        camera_id = recording.camera_id
        end_time = recording.start_time + recording.duration

        logger.info(f"Recording thread started for {recording.recording_id}, duration={recording.duration}s")

        # Track frame count to detect stalled streams
        last_frame_count = len(recording.frames)
        last_frame_check = time.time()
        STALL_CHECK_INTERVAL = 2.0  # Check every 2 seconds
        STALL_TIMEOUT = 5.0  # If no new frames for 5 seconds, stop recording

        try:
            # Wait for recording duration to complete
            while time.time() < end_time and not recording._stop_event.is_set():
                time.sleep(0.1)

                # Check for stalled stream (no new frames)
                now = time.time()
                if now - last_frame_check >= STALL_CHECK_INTERVAL:
                    current_frame_count = len(recording.frames)
                    if current_frame_count == last_frame_count:
                        # No new frames - check if stalled too long
                        stall_time = now - last_frame_check
                        if stall_time >= STALL_TIMEOUT:
                            logger.warning(
                                f"Recording {recording.recording_id} stalled - no new frames for {stall_time:.1f}s, "
                                f"stopping early with {current_frame_count} frames"
                            )
                            break
                    else:
                        # Got new frames, reset stall tracking
                        last_frame_count = current_frame_count
                        last_frame_check = now

            # Recording complete - queue for encoding
            logger.info(
                f"Recording {recording.recording_id} complete: "
                f"{len(recording.frames)} frames captured"
            )

            # Remove from active recordings FIRST so new recordings can start
            with self._lock:
                if camera_id in self._recordings:
                    del self._recordings[camera_id]

            # Queue for encoding using the stored event loop reference
            if self._event_loop and self._encoding_queue:
                try:
                    # Thread-safe way to put into asyncio queue
                    self._event_loop.call_soon_threadsafe(
                        self._encoding_queue.put_nowait, recording
                    )
                    logger.info(f"Queued {recording.recording_id} for encoding")
                except Exception as e:
                    logger.error(f"Failed to queue for async encoding: {e}, encoding synchronously")
                    self._encode_recording_sync(recording)
            else:
                # No event loop available - encode synchronously
                logger.info(f"No async loop, encoding {recording.recording_id} synchronously")
                self._encode_recording_sync(recording)

        except Exception as e:
            logger.error(f"Recording thread error for {recording.recording_id}: {e}", exc_info=True)
            with self._lock:
                if camera_id in self._recordings:
                    del self._recordings[camera_id]

    async def _encoding_loop(self):
        """Background loop that encodes completed recordings."""
        logger.info("Encoding loop started")

        while True:
            try:
                recording = await self._encoding_queue.get()
                logger.info(f"Encoding loop processing: {recording.recording_id}")
                await self._encode_recording(recording)
                self._encoding_queue.task_done()
            except asyncio.CancelledError:
                logger.info("Encoding loop cancelled")
                break
            except Exception as e:
                logger.error(f"Encoding loop error: {e}", exc_info=True)

    async def _encode_recording(self, recording: ActiveRecording):
        """Encode a completed recording to MP4 using FFmpeg.

        Args:
            recording: Completed recording to encode
        """
        if not recording.frames:
            logger.warning(f"No frames to encode for {recording.recording_id}")
            return

        logger.info(f"Encoding recording {recording.recording_id}...")

        try:
            # Run encoding in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                None,
                self._encode_recording_sync,
                recording
            )

            if success:
                # Send video event to backend
                await self._send_video_event(recording)

        except Exception as e:
            logger.error(f"Encoding error for {recording.recording_id}: {e}")

    def _encode_recording_sync(self, recording: ActiveRecording) -> bool:
        """Synchronously encode recording using OpenCV.

        Args:
            recording: Recording to encode

        Returns:
            True if encoding succeeded
        """
        if not recording.frames:
            return False

        try:
            # Get frame dimensions from first frame
            height, width = recording.frames[0].shape[:2]

            # Calculate FPS from timestamps
            if len(recording.timestamps) >= 2:
                total_time = recording.timestamps[-1] - recording.timestamps[0]
                if total_time > 0:
                    fps = len(recording.frames) / total_time
                else:
                    fps = self.default_fps
            else:
                fps = self.default_fps

            # Use reasonable FPS bounds
            fps = max(5, min(30, fps))

            logger.info(
                f"Encoding {recording.recording_id}: {len(recording.frames)} frames, "
                f"{width}x{height}, {fps:.1f} fps"
            )

            # Use OpenCV VideoWriter with H264 codec
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            temp_path = recording.output_path + '.temp.mp4'

            writer = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

            if not writer.isOpened():
                logger.error(f"Failed to open video writer for {recording.output_path}")
                return False

            # Write frames
            for frame in recording.frames:
                writer.write(frame)

            writer.release()

            # Use FFmpeg to re-encode for better compatibility
            try:
                ffmpeg_cmd = [
                    'ffmpeg', '-y',
                    '-i', temp_path,
                    '-c:v', 'libx264',
                    '-preset', 'fast',
                    '-crf', '23',
                    '-movflags', '+faststart',
                    recording.output_path
                ]

                result = subprocess.run(
                    ffmpeg_cmd,
                    capture_output=True,
                    timeout=60
                )

                if result.returncode == 0:
                    # Remove temp file
                    os.remove(temp_path)
                    logger.info(f"Encoded {recording.recording_id} -> {recording.output_path}")
                else:
                    # FFmpeg failed, keep OpenCV output
                    os.rename(temp_path, recording.output_path)
                    logger.warning(f"FFmpeg failed, using OpenCV output: {result.stderr.decode()}")

            except (FileNotFoundError, subprocess.TimeoutExpired) as e:
                # FFmpeg not available or timed out, use OpenCV output
                os.rename(temp_path, recording.output_path)
                logger.warning(f"FFmpeg not available, using OpenCV output: {e}")

            # Clear frames to free memory
            recording.frames.clear()

            return True

        except Exception as e:
            logger.error(f"Encoding failed for {recording.recording_id}: {e}", exc_info=True)
            return False

    async def _send_video_event(self, recording: ActiveRecording):
        """Send video event to backend when recording completes.

        Args:
            recording: Completed recording
        """
        try:
            # Calculate video URL (relative to recordings endpoint)
            video_filename = os.path.basename(recording.output_path)
            video_url = f"/recordings/{video_filename}"

            # Build a descriptive title with context
            title_parts = ["🎬"]

            # Add object type if available (e.g., "person", "truck")
            object_type = recording.metadata.get("object_type")
            if object_type:
                type_labels = {
                    "person": "אדם",
                    "car": "רכב",
                    "truck": "משאית",
                    "bus": "אוטובוס",
                    "motorcycle": "אופנוע",
                }
                title_parts.append(type_labels.get(object_type, object_type))

            # Add event type context
            event_type = recording.metadata.get("event_type", "")
            if event_type == "detection":
                if not object_type:
                    title_parts.append("זיהוי")
            elif event_type == "transcription":
                title_parts.append("קשר")

            # Add camera
            title_parts.append(f"מצלמה {recording.camera_id}")

            title = " - ".join(title_parts) if len(title_parts) > 1 else "הקלטה נשמרה"

            # Build description
            description_parts = [f"{recording.duration:.0f} שניות"]
            if recording.pre_buffer > 0:
                description_parts.append(f"+{recording.pre_buffer:.0f}s לפני האירוע")

            description = " | ".join(description_parts)

            # Build event payload
            event = {
                "type": "video",
                "severity": "info",
                "title": title,
                "description": description,
                "cameraId": recording.camera_id,
                "videoClip": video_url,
                "details": {
                    "recording_id": recording.recording_id,
                    "duration": recording.duration,
                    "pre_buffer": recording.pre_buffer,
                    "frame_count": len(recording.timestamps),
                    "trigger_reason": recording.trigger_reason,
                    "video_url": video_url,
                    **recording.metadata
                }
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{BACKEND_URL}/api/events",
                    json=event,
                    timeout=10.0
                )

                if response.status_code in (200, 201):
                    logger.info(f"Video event sent for {recording.recording_id}")
                else:
                    logger.warning(f"Video event failed: {response.status_code}")

            # Call callback if registered
            if self.on_recording_complete:
                self.on_recording_complete(recording)

        except Exception as e:
            logger.error(f"Failed to send video event: {e}", exc_info=True)

    def get_stats(self) -> Dict:
        """Get recording manager statistics.

        Returns:
            Dictionary with stats
        """
        with self._lock:
            active = {}
            for camera_id, recording in self._recordings.items():
                elapsed = time.time() - recording.start_time
                active[camera_id] = {
                    "recording_id": recording.recording_id,
                    "elapsed": round(elapsed, 1),
                    "duration": recording.duration,
                    "frames_captured": len(recording.frames)
                }

            return {
                "recordings_dir": str(self.recordings_dir),
                "active_recordings": active,
                "encoding_queue_size": self._encoding_queue.qsize() if hasattr(self._encoding_queue, 'qsize') else 0
            }


# Global singleton
_recording_manager: Optional[RecordingManager] = None


def get_recording_manager() -> Optional[RecordingManager]:
    """Get the global recording manager instance."""
    return _recording_manager


async def init_recording_manager(
    recordings_dir: str = RECORDINGS_DIR,
    default_fps: float = 15.0
) -> RecordingManager:
    """Initialize and start the global recording manager.

    Args:
        recordings_dir: Directory to store recordings
        default_fps: Default FPS for encoding

    Returns:
        RecordingManager instance
    """
    global _recording_manager

    if _recording_manager is not None:
        logger.warning("Recording manager already initialized")
        return _recording_manager

    _recording_manager = RecordingManager(
        recordings_dir=recordings_dir,
        default_fps=default_fps
    )
    await _recording_manager.start()

    return _recording_manager
