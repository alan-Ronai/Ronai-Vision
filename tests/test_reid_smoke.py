# tests/test_reid_smoke.py
import os, numpy as np
from services.reid import get_reid

# pick device (optional)
os.environ["DEVICE"] = "auto"  # or 'cpu' or 'mps' or 'cuda'

r = get_reid()  # will raise ImportError with helpful message if missing
import cv2

img = cv2.imread("assets/sample_frame.jpg")  # use an existing sample or create a dummy
if img is None:
    import numpy as np

    img = np.zeros((480, 640, 3), dtype=np.uint8) + 128

# example box: center crop
boxes = np.array([[100, 50, 220, 360]], dtype=np.int32)
feats = r.extract_features(img, boxes)
print("feats.shape =", feats.shape)
print("first vector norm =", np.linalg.norm(feats[0]))
