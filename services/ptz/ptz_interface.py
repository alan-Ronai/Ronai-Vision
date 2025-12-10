"""Simple PTZ simulator interface.

This provides a tiny in-memory PTZ state machine useful for local
simulation and testing. Real PTZ drivers should implement the same
methods (`move`, `set`, `state`).
"""


class PTZSimulator:
    def __init__(self):
        self.pan = 0.0
        self.tilt = 0.0
        self.zoom = 1.0

    def move(
        self, pan_delta: float = 0.0, tilt_delta: float = 0.0, zoom_delta: float = 0.0
    ):
        self.pan += pan_delta
        self.tilt += tilt_delta
        self.zoom = max(0.1, self.zoom + zoom_delta)

    def set(
        self,
        pan: float | None = None,
        tilt: float | None = None,
        zoom: float | None = None,
    ):
        if pan is not None:
            self.pan = float(pan)
        if tilt is not None:
            self.tilt = float(tilt)
        if zoom is not None:
            self.zoom = max(0.1, float(zoom))

    def state(self):
        return {"pan": self.pan, "tilt": self.tilt, "zoom": self.zoom}
