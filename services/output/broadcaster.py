"""In-memory frame broadcaster for streaming rendered frames.

Provides a simple publish/subscribe mechanism where producers call
`publish_frame(cam_id, frame)` with a BGR numpy array and consumers (API
endpoints) can stream MJPEG or websocket frames.
"""

import asyncio
import cv2
import time
from typing import Dict, Optional


class BroadcastEntry:
    def __init__(self):
        self.jpeg: Optional[bytes] = None
        self.ts: float = 0.0


class FrameBroadcaster:
    def __init__(self):
        self._store: Dict[str, BroadcastEntry] = {}
        self._cond = asyncio.Condition()

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

        # notify listeners
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # schedule notification asynchronously
            asyncio.ensure_future(self._notify())

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


# module-level singleton
broadcaster = FrameBroadcaster()
