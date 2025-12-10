import cv2
import numpy as np
import os

from services.reid import get_reid


def main():
    # Load a single frame
    cap = cv2.VideoCapture("assets/sample_video.mp4")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Failed to read sample frame")
        return

    # Sample box: center of frame
    h, w = frame.shape[:2]
    box = np.array([[w // 4, h // 4, 3 * w // 4, 3 * h // 4]], dtype=np.float32)

    reid = get_reid()
    feats = reid.extract_features(frame, box)
    print("Features shape:", feats.shape)
    print("First vector norm:", np.linalg.norm(feats[0]))


if __name__ == "__main__":
    main()
