"""
Concrete code examples: How to add class filtering to your pipeline

Run these patterns directly in scripts/run_multi_camera.py
"""

# ============================================================================
# EXAMPLE 1: CONFIGURE AT TOP OF FILE (easiest for static config)
# ============================================================================
# Add this at the top of run_multi_camera.py after imports:

import os

# Configuration: which object classes to detect and segment
# Set via environment: ALLOWED_CLASSES=person,dog,cat
ALLOWED_CLASSES = os.getenv("ALLOWED_CLASSES", "person").split(",")

# Alternative: confidence threshold (default 0.25)
YOLO_CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", "0.25"))

print(
    f"Pipeline configured: ALLOWED_CLASSES={ALLOWED_CLASSES}, confidence={YOLO_CONFIDENCE}"
)

# Then in run_loop(), use it:
# det = detector.predict(frame, confidence=YOLO_CONFIDENCE)
# seg = segmenter.segment_from_detections(
#     frame, boxes=boxes, class_ids=class_ids,
#     class_names=det.class_names,
#     allowed_class_names=ALLOWED_CLASSES
# )


# ============================================================================
# EXAMPLE 2: ADD CLASS FILTERING TO DETECTION PHASE
# ============================================================================
# Replace the detection section in run_loop() (around line 320-332):

"""
# OLD CODE (segments ALL detections):
det = detector.predict(frame, confidence=0.25)
boxes = det.boxes
scores = det.scores
class_ids = det.class_ids

# Segment (optional)
masks = np.zeros((0, frame.shape[0], frame.shape[1]), dtype=np.uint8)
if len(boxes) > 0:
    seg = segmenter.segment(frame, boxes=boxes)
    masks = seg.masks
"""

# NEW CODE (filters by class before segmentation):
"""
det = detector.predict(frame, confidence=0.25)
boxes = det.boxes
scores = det.scores
class_ids = det.class_ids

# Segment (optional) - WITH CLASS FILTERING
masks = np.zeros((0, frame.shape[0], frame.shape[1]), dtype=np.uint8)
if len(boxes) > 0:
    # Use segment_from_detections to filter by class name
    seg = segmenter.segment_from_detections(
        frame,
        boxes=boxes,
        class_ids=class_ids,
        class_names=det.class_names,
        allowed_class_names=["person"]  # ← CUSTOMIZE THIS
    )
    masks = seg.masks
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0, :, :]
"""


# ============================================================================
# EXAMPLE 3: TRACK ONLY PEOPLE (advanced filtering)
# ============================================================================
# If you want to track ONLY certain classes:

"""
# After detection:
det = detector.predict(frame, confidence=0.25)
boxes = det.boxes
scores = det.scores
class_ids = det.class_ids

# BEFORE: track everything
# tracks = trackers[cam_id].update(boxes, class_ids, scores, feats)

# AFTER: track only people
person_mask = class_ids == 0  # 0 = "person" in YOLO
person_boxes = boxes[person_mask]
person_class_ids = class_ids[person_mask]
person_scores = scores[person_mask]
person_feats = feats[person_mask] if feats is not None and len(feats) > 0 else None

tracks = trackers[cam_id].update(person_boxes, person_class_ids, person_scores, person_feats)
"""


# ============================================================================
# EXAMPLE 4: FILTER BY MULTIPLE CLASSES
# ============================================================================
# Track people AND vehicles:

"""
ALLOWED_CLASSES = ["person", "car", "truck", "bus"]

# After detection:
det = detector.predict(frame, confidence=0.25)

# Find indices of allowed classes
allowed_mask = np.isin(det.class_names[det.class_ids], ALLOWED_CLASSES)

boxes = det.boxes[allowed_mask]
scores = det.scores[allowed_mask]
class_ids = det.class_ids[allowed_mask]
feats = reid.extract_features(frame, boxes)  # Only extract features for allowed boxes

tracks = trackers[cam_id].update(boxes, class_ids, scores, feats)
"""


# ============================================================================
# EXAMPLE 5: EXCLUDE CERTAIN CLASSES (inverse filter)
# ============================================================================
# Track everything EXCEPT backpacks and handbags:

"""
EXCLUDE_CLASSES = ["backpack", "handbag"]

# After detection:
det = detector.predict(frame, confidence=0.25)

# Exclude certain classes
exclude_mask = ~np.isin(det.class_names[det.class_ids], EXCLUDE_CLASSES)

boxes = det.boxes[exclude_mask]
scores = det.scores[exclude_mask]
class_ids = det.class_ids[exclude_mask]
feats = reid.extract_features(frame, boxes)

tracks = trackers[cam_id].update(boxes, class_ids, scores, feats)
"""


# ============================================================================
# EXAMPLE 6: CONFIDENCE + CLASS FILTERING
# ============================================================================
# Different confidence thresholds for different classes:

"""
det = detector.predict(frame, confidence=0.10)  # Get everything

# High confidence required for people
person_mask = (det.class_ids == 0) & (det.scores >= 0.50)

# Lower confidence ok for vehicles
vehicle_classes = [2, 5, 7]
vehicle_mask = np.isin(det.class_ids, vehicle_classes) & (det.scores >= 0.25)

# Combine
final_mask = person_mask | vehicle_mask

boxes = det.boxes[final_mask]
scores = det.scores[final_mask]
class_ids = det.class_ids[final_mask]
feats = reid.extract_features(frame, boxes)

tracks = trackers[cam_id].update(boxes, class_ids, scores, feats)
"""


# ============================================================================
# EXAMPLE 7: YOLO CLASS ID REFERENCE (0-indexed in COCO dataset)
# ============================================================================

YOLO_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "cat",
    15: "dog",
    16: "horse",
    17: "sheep",
    18: "cow",
    19: "elephant",
    20: "bear",
    21: "zebra",
    22: "giraffe",
    23: "backpack",
    24: "umbrella",
    25: "handbag",
    26: "tie",
    27: "suitcase",
    28: "frisbee",
    29: "skis",
    30: "snowboard",
    31: "sports ball",
    32: "kite",
    33: "baseball bat",
    34: "baseball glove",
    35: "skateboard",
    36: "surfboard",
    37: "tennis racket",
    38: "bottle",
    39: "wine glass",
    40: "cup",
    41: "fork",
    42: "knife",
    43: "spoon",
    44: "bowl",
    45: "banana",
    46: "apple",
    47: "sandwich",
    48: "orange",
    49: "broccoli",
    50: "carrot",
    51: "hot dog",
    52: "pizza",
    53: "donut",
    54: "cake",
    55: "chair",
    56: "couch",
    57: "potted plant",
    58: "bed",
    59: "dining table",
    60: "toilet",
    61: "tv",
    62: "laptop",
    63: "mouse",
    64: "remote",
    65: "keyboard",
    66: "microwave",
    67: "oven",
    68: "toaster",
    69: "sink",
    70: "refrigerator",
    71: "book",
    72: "clock",
    73: "vase",
    74: "scissors",
    75: "teddy bear",
    76: "hair drier",
    77: "toothbrush",
}

# Usage:
# allowed_class_ids = [0, 15]  # person and dog
# mask = np.isin(det.class_ids, allowed_class_ids)


# ============================================================================
# QUICK COPY-PASTE TEMPLATES
# ============================================================================

# TEMPLATE 1: Person only (most common)
"""
allowed_class_names=["person"]
seg = segmenter.segment_from_detections(
    frame, boxes=boxes, class_ids=class_ids,
    class_names=det.class_names, allowed_class_names=allowed_class_names
)
"""

# TEMPLATE 2: People + vehicles
"""
allowed_class_names=["person", "car", "truck", "bus", "bicycle", "motorcycle"]
seg = segmenter.segment_from_detections(
    frame, boxes=boxes, class_ids=class_ids,
    class_names=det.class_names, allowed_class_names=allowed_class_names
)
"""

# TEMPLATE 3: Only high-confidence people
"""
person_mask = (det.class_ids == 0) & (det.scores >= 0.50)
if person_mask.sum() > 0:
    boxes = det.boxes[person_mask]
    scores = det.scores[person_mask]
    class_ids = det.class_ids[person_mask]
    # ... rest of pipeline
"""

# TEMPLATE 4: Environment variable controlled
"""
# At top of file:
ALLOWED_CLASSES = os.getenv("ALLOWED_CLASSES", "person").split(",")

# In pipeline:
seg = segmenter.segment_from_detections(
    frame, boxes=boxes, class_ids=class_ids,
    class_names=det.class_names, allowed_class_names=ALLOWED_CLASSES
)

# Run with: ALLOWED_CLASSES="person,car,dog" python scripts/run_multi_camera.py
"""
