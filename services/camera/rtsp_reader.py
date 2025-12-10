"""Simple RTSP/File reader wrapper using OpenCV.

Provides a small synchronous API compatible with the existing
`LocalCameraSimulator.get_frame()` (returns a single frame or None).
"""

import cv2


class RTSPReader:
    def __init__(self, source: str, reopen_on_eof: bool = True):
        self.source = source
        self.reopen_on_eof = reopen_on_eof
        self.cap = cv2.VideoCapture(source)

    def get_frame(self):
        if not self.cap.isOpened():
            self.cap.open(self.source)

        ok, frame = self.cap.read()
        if not ok:
            if self.reopen_on_eof:
                # try to rewind if it's a file
                try:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = self.cap.read()
                except Exception:
                    return None
            else:
                return None
        return frame

    def release(self):
        try:
            self.cap.release()
        except Exception:
            pass
