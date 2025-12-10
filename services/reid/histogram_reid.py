"""Simple ReID feature extraction using averaged RGB histogram."""

import numpy as np
from services.reid.base_reid import BaseReID


class HistogramReID(BaseReID):
    """Simple ReID using normalized RGB histogram features.

    Fast and CPU-friendly for initial person/object tracking.
    For production, replace with a learned embedding (e.g., OSNet, DinoCLIP).
    """

    def __init__(self, bins: int = 16):
        """
        Args:
            bins: number of histogram bins per channel
        """
        self.bins = bins

    def extract_features(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Extract histogram features from boxes.

        Args:
            frame: (H, W, 3) BGR numpy array
            boxes: (N, 4) array of [x1, y1, x2, y2]

        Returns:
            (N, bins*3) histogram feature vectors normalized to [0, 1]
        """
        features = []

        for box in boxes:
            x1, y1, x2, y2 = box.astype(np.int32)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                # Invalid box: return zero feature
                features.append(np.zeros(self.bins * 3, dtype=np.float32))
                continue

            # Crop region
            crop = frame[y1:y2, x1:x2]

            # Compute histogram for each channel
            hist_b = np.histogram(crop[:, :, 0], bins=self.bins, range=(0, 256))[0]
            hist_g = np.histogram(crop[:, :, 1], bins=self.bins, range=(0, 256))[0]
            hist_r = np.histogram(crop[:, :, 2], bins=self.bins, range=(0, 256))[0]

            # Concatenate and normalize
            hist = np.concatenate([hist_b, hist_g, hist_r]).astype(np.float32)
            hist = hist / (hist.sum() + 1e-5)  # Normalize to [0, 1]

            features.append(hist)

        return np.array(features, dtype=np.float32)
