"""Output publisher: unified frame rendering and distribution.

Handles rendering detections on frames and publishing to various outputs
(broadcaster, local files, HTTP endpoints).
"""

import os
import json
import cv2
import numpy as np
import requests
import threading
from typing import Optional, Dict, Any

from services.output.renderer import FrameRenderer
from services.output.broadcaster import broadcaster


class OutputPublisher:
    """Unified output publishing pipeline.

    Renders tracks/masks on frames and publishes to multiple outputs:
    - MJPEG broadcaster (for web streaming)
    - Local disk (optional, SAVE_FRAMES=true)
    - HTTP server (optional, STREAM_SERVER_URL=...)

    Shutdown-aware: stops publishing when stop_event is set.
    """

    def __init__(
        self,
        output_dir: str = "output/multi",
        save_frames: bool = False,
        stream_url: Optional[str] = None,
        stream_token: Optional[str] = None,
        stop_event: Optional[threading.Event] = None,
    ):
        """Initialize publisher.

        Args:
            output_dir: Directory for saving frames (if enabled)
            save_frames: Whether to save rendered frames locally
            stream_url: Optional HTTP endpoint for frame publishing
            stream_token: Optional auth token for HTTP publishing
            stop_event: Optional event to signal shutdown (stops HTTP publishing)
        """
        self.renderer = FrameRenderer()
        self.output_dir = output_dir
        self.save_frames = save_frames
        self.stream_url = stream_url
        self.stream_token = stream_token
        self.stop_event = stop_event
        self._http_failed_count = {}  # Track consecutive failures per camera
        self._http_disabled = False  # Disable HTTP after too many failures

        os.makedirs(output_dir, exist_ok=True)

    def publish(
        self,
        frame: np.ndarray,
        camera_id: str,
        frame_idx: int,
        process_result: Dict[str, Any],
    ) -> np.ndarray:
        """Render and publish a frame.

        Args:
            frame: Original BGR frame
            camera_id: Camera identifier
            frame_idx: Frame index (for file naming)
            process_result: Output from FrameProcessor.process_frame()

        Returns:
            Rendered frame (BGR, uint8)
        """
        tracks = process_result["tracks"]
        masks = process_result["masks"]
        class_names = process_result["class_names"]

        # Render masks
        out = self.renderer.render_masks(frame, masks)

        # Render tracks with class names and IDs
        out = self.renderer.render_tracks(out, tracks, class_names)

        # Publish to broadcaster (for MJPEG streaming)
        self._publish_to_broadcaster(out, camera_id)

        # Optionally save to disk
        if self.save_frames:
            self._save_to_disk(out, camera_id, frame_idx, process_result)

        # Optionally publish to HTTP server (skip if shutting down or disabled)
        if self.stream_url and not self._is_shutting_down() and not self._http_disabled:
            self._publish_to_http(out, camera_id)

        return out

    def _is_shutting_down(self) -> bool:
        """Check if shutdown has been requested.

        Returns:
            True if stop_event is set, False otherwise
        """
        return self.stop_event is not None and self.stop_event.is_set()

    def _publish_to_broadcaster(self, frame: np.ndarray, camera_id: str) -> None:
        """Publish frame to in-memory broadcaster for MJPEG streaming.

        Args:
            frame: Rendered BGR frame
            camera_id: Camera identifier
        """
        try:
            broadcaster.publish_frame(camera_id, frame)
        except Exception as e:
            print(f"[WARNING] Broadcaster publish failed for {camera_id}: {e}")

    def _save_to_disk(
        self,
        frame: np.ndarray,
        camera_id: str,
        frame_idx: int,
        process_result: Dict[str, Any],
    ) -> None:
        """Save frame and metadata to disk.

        Args:
            frame: Rendered BGR frame
            camera_id: Camera identifier
            frame_idx: Frame index
            process_result: Processing results with track info
        """
        try:
            cam_dir = os.path.join(self.output_dir, camera_id)
            os.makedirs(cam_dir, exist_ok=True)

            # Save image
            img_path = os.path.join(cam_dir, f"frame_{frame_idx:06d}.jpg")
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ok:
                with open(img_path, "wb") as f:
                    f.write(buf.tobytes())

            # Save metadata
            tracks = process_result["tracks"]
            metadata = {
                "frame_idx": frame_idx,
                "camera_id": camera_id,
                "num_tracks": len(tracks),
                "tracks": [
                    {
                        "track_id": t.track_id,
                        "box": t.box.tolist(),
                        "class_id": int(t.class_id),
                        "confidence": float(t.confidence),
                    }
                    for t in tracks
                ],
            }

            meta_path = os.path.join(cam_dir, f"frame_{frame_idx:06d}.json")
            with open(meta_path, "w") as f:
                json.dump(metadata, f)
        except Exception as e:
            print(f"[WARNING] Disk save failed for {camera_id}: {e}")

    def _publish_to_http(self, frame: np.ndarray, camera_id: str) -> None:
        """Publish frame to HTTP endpoint.

        Automatically disables HTTP publishing after 10 consecutive failures
        to prevent log spam and performance degradation.

        Args:
            frame: Rendered BGR frame
            camera_id: Camera identifier
        """
        try:
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                return

            url = f"{self.stream_url.rstrip('/')}/api/stream/publish/{camera_id}"
            headers = {"Content-Type": "image/jpeg"}
            if self.stream_token:
                headers["x-stream-token"] = self.stream_token

            # Reduced timeout to fail fast during shutdown
            response = requests.post(
                url, data=buf.tobytes(), headers=headers, timeout=2.0
            )

            # Success - reset failure count
            if response.status_code == 200:
                self._http_failed_count[camera_id] = 0
            else:
                self._track_http_failure(camera_id, f"HTTP {response.status_code}")

        except requests.exceptions.Timeout:
            self._track_http_failure(camera_id, "timeout")
        except requests.exceptions.ConnectionError:
            self._track_http_failure(camera_id, "connection refused")
        except Exception as e:
            self._track_http_failure(camera_id, str(e))

    def _track_http_failure(self, camera_id: str, reason: str) -> None:
        """Track HTTP publishing failures and disable after threshold.

        Args:
            camera_id: Camera identifier
            reason: Failure reason for logging
        """
        if camera_id not in self._http_failed_count:
            self._http_failed_count[camera_id] = 0

        self._http_failed_count[camera_id] += 1

        # Only log first few failures to avoid spam
        if self._http_failed_count[camera_id] <= 3:
            print(
                f"[WARNING] HTTP publish {reason} for {camera_id} ({self._http_failed_count[camera_id]}/10)"
            )

        # Disable HTTP publishing after 10 consecutive failures
        if self._http_failed_count[camera_id] >= 10:
            if not self._http_disabled:
                self._http_disabled = True
                print(
                    f"[WARNING] HTTP publishing disabled after 10 consecutive failures"
                )
                print(
                    f"[INFO] Frames still available via MJPEG broadcaster at /api/stream/mjpeg/{camera_id}"
                )
