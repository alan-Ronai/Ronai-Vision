import os
import cv2
import numpy as np

from services.camera.simulator import LocalCameraSimulator
from services.detector import YOLODetector
from services.segmenter.sam2_segmenter import SAM2Segmenter
from services.tracker.centroid_tracker import CentroidTracker
from services.reid.histogram_reid import HistogramReID
from services.output.renderer import FrameRenderer


def main():
    out_dir = "output/samples"
    os.makedirs(out_dir, exist_ok=True)

    cam = LocalCameraSimulator(file_path="assets/sample_video.mp4")

    detector = YOLODetector(model_name="yolo12n.pt", device="cpu")
    segmenter = SAM2Segmenter(model_type="small", device="cpu")
    tracker = CentroidTracker()
    reid = HistogramReID()
    renderer = FrameRenderer()

    for i in range(5):
        frame = cam.get_frame()
        if frame is None:
            print("No more frames")
            break

        print(f"Processing frame {i}")

        # Detection
        det = detector.predict(frame, confidence=0.25)
        boxes = det.boxes
        scores = det.scores
        class_ids = det.class_ids

        # Segmentation (if any boxes)
        masks = np.zeros((0, frame.shape[0], frame.shape[1]), dtype=np.uint8)
        try:
            if len(boxes) > 0:
                seg_res = segmenter.segment(frame, boxes=boxes)
                masks = seg_res.masks
                # Normalize mask shapes: accept (N,H,W) or (N,1,H,W)
                if masks.ndim == 4 and masks.shape[1] == 1:
                    masks = masks[:, 0, :, :]
                elif masks.ndim == 3:
                    pass
                else:
                    # Attempt to coerce to (N,H,W)
                    masks = masks.reshape(
                        (masks.shape[0], frame.shape[0], frame.shape[1])
                    )
        except Exception as e:
            print(f"Segmentation error: {e}")

        # Tracking
        tracks = tracker.update(boxes, class_ids, scores)

        # ReID features (not used for association here, just computed)
        if len(boxes) > 0:
            features = reid.extract_features(frame, boxes)
            print(f"Extracted features shape: {features.shape}")

        # Render masks then tracks
        out = renderer.render_masks(frame, masks)
        out = renderer.render_tracks(out, tracks, det.class_names)

        out_path = os.path.join(out_dir, f"frame_{i:03d}.png")
        cv2.imwrite(out_path, out)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
