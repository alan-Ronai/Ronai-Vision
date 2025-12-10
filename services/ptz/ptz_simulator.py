class PTZSimulator:
    def __init__(self):
        self.pan = 0
        self.tilt = 0
        self.zoom = 1.0

    def move_to(self, pan, tilt, zoom):
        self.pan = pan
        self.tilt = tilt
        self.zoom = zoom
        print(f"[PTZ] Moving to pan={pan}, tilt={tilt}, zoom={zoom}")
