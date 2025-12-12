"""In-memory frame broadcaster for streaming rendered frames.

Provides a simple publish/subscribe mechanism where producers call
`publish_frame(cam_id, frame)` with a BGR numpy array and consumers (API
endpoints) can stream MJPEG or websocket frames.

Tracks active viewers per camera for on-demand processing.
"""

import asyncio
import cv2
import time
import threading
from typing import Dict, Optional, Set, List


class BroadcastEntry:
    def __init__(self):
        self.jpeg: Optional[bytes] = None
        self.ts: float = 0.0
        self.viewer_count: int = 0  # Active viewers for this camera


class FrameBroadcaster:
    def __init__(self):
        self._store: Dict[str, BroadcastEntry] = {}
        self._cond = asyncio.Condition()
        self._configured_cameras: Set[str] = set()  # All known camera IDs
        self._lock = threading.Lock()  # Thread-safe viewer count operations

    def publish_frame(self, cam_id: str, frame) -> None:
        """Publish a BGR frame (numpy array). Encodes to JPEG and notifies listeners."""
        try:
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                return
            b = buf.tobytes()
        except Exception:
            return

        entry = self._store.setdefault(cam_id, BroadcastEntry())
        entry.jpeg = b
        entry.ts = time.time()

        # notify listeners (thread-safe)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop in current thread - this is expected when called from worker threads
            # Notification will happen on next wait_for_frame() call
            return

        if loop.is_running():
            # schedule notification asynchronously
            asyncio.run_coroutine_threadsafe(self._notify(), loop)

    async def _notify(self):
        async with self._cond:
            self._cond.notify_all()

    async def wait_for_frame(
        self, cam_id: str, timeout: float = 5.0
    ) -> Optional[bytes]:
        """Wait until a new frame is available for cam_id. Returns JPEG bytes or None."""
        entry = self._store.get(cam_id)
        if entry and entry.jpeg:
            return entry.jpeg

        try:
            async with self._cond:
                await asyncio.wait_for(self._cond.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

        entry = self._store.get(cam_id)
        return entry.jpeg if entry else None

    def status(self) -> Dict[str, float]:
        """Return a mapping of cam_id -> last_publish_timestamp (epoch seconds).

        Useful for health/debugging to see whether frames are arriving.
        """
        return {k: v.ts for k, v in self._store.items()}

    def register_camera(self, cam_id: str) -> None:
        """Register a camera as configured (may not be active yet).

        Args:
            cam_id: Camera identifier
        """
        with self._lock:
            self._configured_cameras.add(cam_id)
            if cam_id not in self._store:
                self._store[cam_id] = BroadcastEntry()

    def get_all_camera_ids(self) -> List[str]:
        """Get list of all configured camera IDs.

        Returns:
            List of camera IDs
        """
        with self._lock:
            return list(self._configured_cameras)

    def increment_viewer(self, cam_id: str) -> int:
        """Increment viewer count for a camera.

        Args:
            cam_id: Camera identifier

        Returns:
            New viewer count
        """
        with self._lock:
            entry = self._store.setdefault(cam_id, BroadcastEntry())
            entry.viewer_count += 1
            return entry.viewer_count

    def decrement_viewer(self, cam_id: str) -> int:
        """Decrement viewer count for a camera.

        Args:
            cam_id: Camera identifier

        Returns:
            New viewer count (minimum 0)
        """
        with self._lock:
            entry = self._store.get(cam_id)
            if entry:
                entry.viewer_count = max(0, entry.viewer_count - 1)
                return entry.viewer_count
            return 0

    def get_viewer_count(self, cam_id: str) -> int:
        """Get current viewer count for a camera.

        Args:
            cam_id: Camera identifier

        Returns:
            Number of active viewers
        """
        with self._lock:
            entry = self._store.get(cam_id)
            return entry.viewer_count if entry else 0

    def get_active_cameras(self) -> List[str]:
        """Get list of cameras with active viewers.

        Returns:
            List of camera IDs with viewer_count > 0
        """
        with self._lock:
            return [
                cam_id
                for cam_id, entry in self._store.items()
                if entry.viewer_count > 0
            ]


# module-level singleton
broadcaster = FrameBroadcaster()
