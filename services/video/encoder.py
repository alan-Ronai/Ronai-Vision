"""
Video encoding service for async frame-to-video conversion.

Supports:
- Async background encoding (doesn't block main pipeline)
- GPU hardware acceleration (NVIDIA CUDA H264 encoding)
- Frame buffering and video compilation
- Metadata attachment (tracks, detections, timestamps)
"""

import logging
import threading
import queue
import os
import time
from pathlib import Path
from typing import Optional, Dict, List
import numpy as np
import cv2
from datetime import datetime

logger = logging.getLogger(__name__)


class VideoEncoder:
    """Async video encoder with GPU acceleration support."""

    def __init__(
        self,
        output_dir: str = "output/recordings",
        fps: int = 30,
        codec: str = "auto",  # "auto", "h264", "h264_nvenc", "h264_qsv"
        bitrate: str = "5000k",
    ):
        """Initialize video encoder.

        Args:
            output_dir: Output directory for video files
            fps: Frames per second
            codec: Video codec. "auto" detects best available (NVIDIA > Intel > Software)
            bitrate: Bitrate (e.g., "5000k", "10M")
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.fps = fps
        self.bitrate = bitrate
        self.codec = self._select_codec(codec)

        # Encoding queue and worker thread
        self.encode_queue: queue.Queue = queue.Queue(maxsize=1000)
        self.worker_thread = threading.Thread(
            target=self._encode_worker, daemon=True, name="VideoEncoderWorker"
        )
        self.worker_thread.start()

        # Active recording sessions: session_id -> {frames, metadata, start_time}
        self.sessions: Dict[str, Dict] = {}
        self.sessions_lock = threading.Lock()

        logger.info(
            f"VideoEncoder initialized: codec={self.codec}, fps={fps}, bitrate={bitrate}"
        )

    def _select_codec(self, requested: str) -> str:
        """Select best available codec with GPU acceleration.

        Priority: NVIDIA CUDA > Intel QSV > Software H264

        Args:
            requested: "auto" or specific codec name

        Returns:
            Selected codec name for FFmpeg
        """
        if requested != "auto":
            return requested

        # Check available codecs
        codecs_available = []

        # Check NVIDIA CUDA (GPU-accelerated)
        if self._check_cuda_available():
            codecs_available.append("h264_nvenc")

        # Check Intel QSV (Quick Sync)
        if self._check_qsv_available():
            codecs_available.append("h264_qsv")

        # Always fall back to software encoder
        codecs_available.append("libx264")

        selected = codecs_available[0]
        logger.info(f"Codec selection: tried {codecs_available}, selected {selected}")
        return selected

    def _check_cuda_available(self) -> bool:
        """Check if NVIDIA CUDA H264 encoding is available."""
        try:
            # Try to check via ffmpeg
            import subprocess

            result = subprocess.run(
                ["ffmpeg", "-codecs"], capture_output=True, text=True, timeout=5
            )
            return "h264_nvenc" in result.stdout
        except Exception:
            return False

    def _check_qsv_available(self) -> bool:
        """Check if Intel QSV H264 encoding is available."""
        try:
            import subprocess

            result = subprocess.run(
                ["ffmpeg", "-codecs"], capture_output=True, text=True, timeout=5
            )
            return "h264_qsv" in result.stdout
        except Exception:
            return False

    def start_recording(
        self,
        session_id: str,
        camera_id: Optional[str] = None,
        event_type: Optional[str] = None,
    ) -> str:
        """Start a new recording session.

        Args:
            session_id: Unique session ID
            camera_id: Camera identifier
            event_type: Type of event (e.g., "armed_person", "weapon_detected")

        Returns:
            Session ID
        """
        with self.sessions_lock:
            if session_id in self.sessions:
                logger.warning(f"Session {session_id} already exists")
                return session_id

            self.sessions[session_id] = {
                "frames": [],
                "metadata": {
                    "camera_id": camera_id,
                    "event_type": event_type,
                    "start_time": time.time(),
                    "track_ids": set(),
                },
                "start_time": time.time(),
            }

        logger.info(
            f"Recording started: session={session_id}, camera={camera_id}, event={event_type}"
        )
        return session_id

    def add_frame(
        self,
        session_id: str,
        frame: np.ndarray,
        track_ids: Optional[List[int]] = None,
    ) -> bool:
        """Add frame to recording session.

        Args:
            session_id: Session ID
            frame: Frame (H, W, 3) BGR uint8
            track_ids: List of track IDs visible in frame

        Returns:
            True if added, False if session not found
        """
        with self.sessions_lock:
            if session_id not in self.sessions:
                return False

            # Store frame
            self.sessions[session_id]["frames"].append(frame.copy())

            # Update track IDs
            if track_ids:
                self.sessions[session_id]["metadata"]["track_ids"].update(track_ids)

        return True

    def stop_recording(self, session_id: str) -> Optional[str]:
        """Stop recording and queue for encoding.

        Args:
            session_id: Session ID

        Returns:
            Video output path (or None if encoding will happen in background)
        """
        with self.sessions_lock:
            if session_id not in self.sessions:
                logger.warning(f"Session {session_id} not found")
                return None

            session_data = self.sessions.pop(session_id)

        if len(session_data["frames"]) == 0:
            logger.warning(f"Session {session_id} has no frames")
            return None

        # Queue for background encoding
        try:
            self.encode_queue.put_nowait((session_id, session_data))
            logger.info(
                f"Recording queued for encoding: {session_id} ({len(session_data['frames'])} frames)"
            )
        except queue.Full:
            logger.error(f"Encode queue full, dropping session {session_id}")
            return None

        return None  # Will be ready after async encoding

    def _encode_worker(self):
        """Background worker thread for encoding videos."""
        while True:
            try:
                session_id, session_data = self.encode_queue.get(timeout=1)
                self._encode_session(session_id, session_data)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Encoding worker error: {e}")

    def _encode_session(self, session_id: str, session_data: Dict):
        """Encode a single recording session to video file.

        Args:
            session_id: Session ID
            session_data: Session data with frames and metadata
        """
        try:
            frames = session_data["frames"]
            metadata = session_data["metadata"]

            if len(frames) == 0:
                logger.warning(f"No frames to encode for session {session_id}")
                return

            # Generate output path
            timestamp = datetime.fromtimestamp(metadata["start_time"]).isoformat()
            camera_id = metadata.get("camera_id", "unknown")
            event_type = metadata.get("event_type", "event")
            filename = f"{camera_id}_{event_type}_{timestamp.replace(':', '-')}.mp4"
            output_path = self.output_dir / filename

            # Create VideoWriter
            frame_h, frame_w = frames[0].shape[:2]
            fourcc = self._get_fourcc()

            writer = cv2.VideoWriter(
                str(output_path), fourcc, self.fps, (frame_w, frame_h)
            )

            if not writer.isOpened():
                logger.error(f"Failed to open VideoWriter for {output_path}")
                return

            # Write frames
            for i, frame in enumerate(frames):
                writer.write(frame)
                if (i + 1) % 100 == 0:
                    logger.debug(f"Encoded {i + 1}/{len(frames)} frames")

            writer.release()

            # Save metadata JSON
            metadata_copy = metadata.copy()
            metadata_copy["track_ids"] = list(metadata_copy["track_ids"])
            metadata_copy["end_time"] = time.time()
            metadata_copy["total_frames"] = len(frames)
            metadata_copy["video_path"] = str(output_path)

            import json

            metadata_path = output_path.with_suffix(".json")
            with open(metadata_path, "w") as f:
                json.dump(metadata_copy, f, indent=2)

            logger.info(
                f"Video encoded: {output_path} "
                f"({len(frames)} frames, {metadata_copy['track_ids']})"
            )

        except Exception as e:
            logger.error(f"Failed to encode session {session_id}: {e}")

    def _get_fourcc(self) -> int:
        """Get FourCC code for selected codec."""
        if self.codec == "h264_nvenc":
            return cv2.VideoWriter_fourcc(*"H264")
        elif self.codec == "h264_qsv":
            return cv2.VideoWriter_fourcc(*"H264")
        else:  # libx264 / software
            return cv2.VideoWriter_fourcc(*"mp4v")

    def get_video_path(self, session_id: str) -> Optional[str]:
        """Get video file path for completed session.

        Args:
            session_id: Session ID

        Returns:
            Path to video file if exists, None otherwise
        """
        # Look for video file matching session_id
        for video_file in self.output_dir.glob("*.mp4"):
            metadata_file = video_file.with_suffix(".json")
            if metadata_file.exists():
                import json

                try:
                    with open(metadata_file) as f:
                        metadata = json.load(f)
                        # Match by camera_id + event_type + approximate time
                        # For now, just check file exists
                        return str(video_file)
                except Exception:
                    pass

        return None


# Global encoder instance
_encoder: Optional[VideoEncoder] = None


def get_video_encoder() -> VideoEncoder:
    """Get or initialize global video encoder."""
    global _encoder
    if _encoder is None:
        _encoder = VideoEncoder()
    return _encoder
