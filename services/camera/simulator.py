import cv2


class LocalCameraSimulator:
    def __init__(self, file_path):
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {file_path}")

    def get_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            # Try to restart from beginning
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()

        if not ok or frame is None:
            # Return a blank frame if all else fails
            return None

        return frame
