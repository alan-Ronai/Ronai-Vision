"""WebRTC video track that pulls frames from the in-memory broadcaster.

Requires `aiortc` and `av` packages. The track decodes JPEG bytes published
by `services.output.broadcaster` and yields `av.VideoFrame` instances for
sending over a RTCPeerConnection.
"""

import asyncio
import time
import numpy as np
import cv2
from av import VideoFrame
from aiortc import VideoStreamTrack


class BroadcasterVideoTrack(VideoStreamTrack):
    """A Track that reads JPEG frames from a broadcaster and yields VideoFrame."""

    def __init__(self, broadcaster, cam_id: str, fps: float = 15.0):
        super().__init__()  # don't forget this
        self.broadcaster = broadcaster
        self.cam_id = cam_id
        self.frame_time = 1.0 / float(fps)
        self._last_timestamp = None

    async def recv(self):
        pts, time_base = None, None
        # throttle to target fps
        now = time.time()
        if self._last_timestamp is not None:
            elapsed = now - self._last_timestamp
            to_wait = max(0.0, self.frame_time - elapsed)
            if to_wait > 0:
                await asyncio.sleep(to_wait)

        jpeg = await self.broadcaster.wait_for_frame(self.cam_id, timeout=2.0)
        if jpeg is None:
            # return a blank frame
            w, h = 640, 480
            img = np.zeros((h, w, 3), dtype=np.uint8)
            frame = VideoFrame.from_ndarray(img, format="bgr24")
            frame.pts = 0
            frame.time_base = 1 / 90000
            self._last_timestamp = time.time()
            return frame

        # decode JPEG to ndarray
        arr = np.frombuffer(jpeg, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            # fallback blank
            h, w = 480, 640
            img = np.zeros((h, w, 3), dtype=np.uint8)

        # Convert BGR to RGB for av
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame = VideoFrame.from_ndarray(rgb, format="rgb24")
        frame.pts = None
        frame.time_base = None
        self._last_timestamp = time.time()
        return frame
