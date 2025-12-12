"""Simple RTSP/File reader wrapper using OpenCV.

Provides a small synchronous API compatible with the existing
`LocalCameraSimulator.get_frame()` (returns a single frame or None).
"""

import cv2
import time


class RTSPReader:
    def __init__(self, source: str, reopen_on_eof: bool = True):
        self.source = source
        self.reopen_on_eof = reopen_on_eof
        self.cap = self._create_capture(source)
        self._consecutive_failures = 0
        self._max_failures = (
            20  # Wait for 20 failures before reconnect (more tolerant of errors)
        )
        self._last_frame = None  # Cache last valid frame
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 3

    def _create_capture(self, source: str) -> cv2.VideoCapture:
        """Create VideoCapture with error-tolerant settings for RTSP streams."""
        cap = cv2.VideoCapture(source)

        # Enable error concealment for corrupted frames
        # CAP_PROP_FOURCC = 6, but we use backend-specific options
        if source.startswith("rtsp://"):
            # For RTSP streams, use FFmpeg backend options
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer lag

            # OpenCV may expose these via environment or backend settings
            # These help with corrupted H.264 macroblocks:
            # - ec=guess_mvs: Error concealment via motion vector guessing
            # - err_detect=ignore_err: Continue despite errors

        return cap

    def get_frame(self):
        if not self.cap.isOpened():
            self.cap = self._create_capture(self.source)
            self._consecutive_failures = 0
            self._reconnect_attempts = 0

        ok, frame = self.cap.read()
        if not ok:
            self._consecutive_failures += 1

            # If too many failures, try to reconnect
            if self._consecutive_failures >= self._max_failures:
                print(
                    f"[RTSPReader] {self._max_failures} consecutive failures, reconnecting stream (attempt {self._reconnect_attempts + 1})..."
                )
                try:
                    self.cap.release()
                except Exception:
                    pass

                time.sleep(1.0)  # Longer delay before reconnect to let stream stabilize
                self.cap = self._create_capture(self.source)
                self._consecutive_failures = 0
                self._reconnect_attempts += 1

                # Give stream time to stabilize after reconnection
                for retry in range(5):
                    try:
                        ok, frame = self.cap.read()
                        if ok and frame is not None and frame.size > 0:
                            print(
                                f"[RTSPReader] Stream recovered after {retry + 1} attempts"
                            )
                            self._last_frame = frame
                            self._reconnect_attempts = 0
                            return frame
                    except Exception as e:
                        print(f"[RTSPReader] Reconnect attempt {retry + 1} failed: {e}")
                    time.sleep(0.2)

                # If still failing after reconnect, return last valid frame
                return self._last_frame

            # For files, try to rewind
            if self.reopen_on_eof:
                try:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = self.cap.read()
                    if ok and frame is not None and frame.size > 0:
                        self._consecutive_failures = 0
                        self._last_frame = frame
                        return frame
                except Exception:
                    pass

            # Return last valid frame as fallback (prevents gaps)
            return self._last_frame

        # Frame read successfully - validate it
        if frame is None or frame.size == 0:
            self._consecutive_failures += 1
            return self._last_frame

        # Frame is good
        self._consecutive_failures = 0
        self._last_frame = frame
        return frame

    def release(self):
        try:
            self.cap.release()
        except Exception:
            pass
