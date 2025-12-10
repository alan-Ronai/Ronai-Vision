import cv2
import numpy as np
import sys
from services.segmenter.sam2_segmenter import SAM2Segmenter


def main():
    cap = cv2.VideoCapture("assets/sample_video.mp4")
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        print("ERROR: failed to read a frame from assets/sample_video.mp4")
        sys.exit(2)

    h, w = frame.shape[:2]
    # a central box prompt
    box = np.array([[w // 4, h // 4, (w * 3) // 4, (h * 3) // 4]], dtype=np.float32)

    try:
        seg = SAM2Segmenter(model_type="small", device="cpu")
    except Exception as e:
        print(f"Failed to initialize SAM2Segmenter: {e}")
        sys.exit(3)

    try:
        res = seg.segment(frame, boxes=box)
        print(f"Masks shape: {res.masks.shape}")
        print(f"Scores: {res.scores}")
        if res.masks.shape[0] > 0:
            mask0 = (res.masks[0] * 255).astype("uint8")
            out_path = "mask0.png"
            cv2.imwrite(out_path, mask0)
            print(f"Wrote mask image to {out_path}")
    except Exception as e:
        print(f"Segmentation failed: {e}")
        sys.exit(4)


if __name__ == "__main__":
    main()
