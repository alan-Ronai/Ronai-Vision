"""
CONFIGURATION GUIDE: How to Set What to Detect and Segment

This guide shows exactly where and how to configure YOLO detection and SAM2 segmentation
in the Ronai-Vision pipeline.

===================================================================================

1. # DETECTION CLASSES (YOLO)

YOLO12n detects these classes by default:
person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light,
fire hydrant, stop sign, parking meter, bench, cat, dog, horse, sheep, cow, elephant,
bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis,
snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard,
tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple,
sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch,
potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard,
microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors,
teddy bear, hair drier, toothbrush

DETECTION Configuration in scripts/run_multi_camera.py (lines ~55-65):

    detector = YOLODetector(model_name="yolo12n.pt", device="cpu")

    # Later in the pipeline (line ~320+):
    det = detector.predict(frame, confidence=0.25)  # confidence threshold
    boxes = det.boxes
    class_ids = det.class_ids
    class_names = det.class_names

Adjust confidence threshold:
↓↓↓ LOWER = more detections (but more false positives)
det = detector.predict(frame, confidence=0.10) # More sensitive

    ↓↓↓ HIGHER = fewer detections (more conservative)
    det = detector.predict(frame, confidence=0.50)  # More selective

=================================================================================== 2. SEGMENTATION FILTERING (SAM2 + CLASS FILTER)
===================================================================================

By default, SAM2 segments ALL detected objects. To segment ONLY certain classes:

## OPTION A: Use segment_from_detections() to filter by class name (RECOMMENDED)

Location: In scripts/run_multi_camera.py, around line 320-330

BEFORE (segments everything):
seg = segmenter.segment(frame, boxes=boxes)
masks = seg.masks

AFTER (segments only "person" boxes):
seg = segmenter.segment_from_detections(
frame,
boxes=boxes,
class_ids=class_ids,
class_names=det.class_names,
allowed_class_names=["person"] # <-- CUSTOMIZE THIS
)
masks = seg.masks

Example: Segment ONLY people AND cars:
seg = segmenter.segment_from_detections(
frame,
boxes=boxes,
class_ids=class_ids,
class_names=det.class_names,
allowed_class_names=["person", "car"] # <-- Multiple classes
)
masks = seg.masks

Example: Segment all BUT skip certain classes: # Get all class names except "backpack"
all_classes = set(det.class_names)
allowed = list(all_classes - {"backpack", "handbag"})

    seg = segmenter.segment_from_detections(
        frame,
        boxes=boxes,
        class_ids=class_ids,
        class_names=det.class_names,
        allowed_class_names=allowed
    )
    masks = seg.masks

=================================================================================== 3. TRACKING & ReID FILTERING
===================================================================================

By default, ALL detected objects are tracked. To track ONLY certain classes:

Location: After segmentation (around line ~340-350)

## OPTION A: Filter tracks by class before ReID

BEFORE (tracks everything):
tracks = trackers[cam_id].update(boxes, class_ids, scores, feats)

AFTER (track only people): # Filter to only person detections
person_mask = np.array([cls == 0 for cls in class_ids]) # 0 = "person" in YOLO
filtered_boxes = boxes[person_mask]
filtered_class_ids = class_ids[person_mask]
filtered_scores = scores[person_mask]
filtered_feats = feats[person_mask] if feats is not None else None

    # Pass filtered detections to tracker
    tracks = trackers[cam_id].update(
        filtered_boxes, filtered_class_ids, filtered_scores, filtered_feats
    )

=================================================================================== 4. PRACTICAL EXAMPLES
===================================================================================

## EXAMPLE 1: Track ONLY people (common use case)

Location: scripts/run_multi_camera.py, around line ~315-350

    det = detector.predict(frame, confidence=0.25)

    # Filter to persons only (class_id 0)
    person_mask = det.class_ids == 0
    person_boxes = det.boxes[person_mask]
    person_class_ids = det.class_ids[person_mask]
    person_scores = det.scores[person_mask]

    # Segment persons only
    if len(person_boxes) > 0:
        seg = segmenter.segment(frame, boxes=person_boxes)
        masks = seg.masks
    else:
        masks = np.zeros((0, frame.shape[0], frame.shape[1]), dtype=np.uint8)

    # ReID only persons
    feats = None
    if len(person_boxes) > 0:
        feats = reid.extract_features(frame, person_boxes)

    # Track only persons
    tracks = trackers[cam_id].update(person_boxes, person_class_ids, person_scores, feats)

## EXAMPLE 2: Track vehicles (cars, trucks, buses)

    vehicle_classes = [2, 5, 7]  # car, bus, truck in YOLO
    vehicle_mask = np.isin(det.class_ids, vehicle_classes)
    vehicle_boxes = det.boxes[vehicle_mask]
    vehicle_class_ids = det.class_ids[vehicle_mask]
    vehicle_scores = det.scores[vehicle_mask]

    # ... rest of pipeline with vehicle_boxes, vehicle_class_ids, etc.

## EXAMPLE 3: Track everything EXCEPT backpacks and handbags

    exclude_classes = [24, 25]  # backpack, handbag in YOLO
    keep_mask = ~np.isin(det.class_ids, exclude_classes)
    filtered_boxes = det.boxes[keep_mask]
    filtered_class_ids = det.class_ids[keep_mask]
    # ... rest of pipeline

## EXAMPLE 4: Two-tier filtering (detection confidence + class filter)

    # High confidence on people
    det = detector.predict(frame, confidence=0.50)
    person_mask = det.class_ids == 0
    person_boxes = det.boxes[person_mask]

    # Medium confidence on vehicles
    vehicle_classes = [2, 5, 7]
    vehicle_mask = np.isin(det.class_ids, vehicle_classes) & (det.scores >= 0.30)
    vehicle_boxes = det.boxes[vehicle_mask]

    # Combine
    all_boxes = np.vstack([person_boxes, vehicle_boxes])
    # ... rest of pipeline

=================================================================================== 5. WHERE TO MAKE EDITS
===================================================================================

File: /Users/alankantor/Downloads/Ronai/Ronai-Vision/scripts/run_multi_camera.py

Search for these lines to find where to edit:

Line ~55: "detector = YOLODetector(...)" ← Configure model/device
Line ~320: "det = detector.predict(...)" ← Adjust confidence threshold
Line ~330: "seg = segmenter.segment(...)" ← Add class filtering here
Line ~340: "feats = reid.extract_features(...)" ← ReID on filtered boxes
Line ~345: "tracks = trackers[...].update(...)" ← Track filtered detections

You can also create custom versions: - Copy run_loop() and modify it for your use case - Or add environment variables to control filtering (advanced)

=================================================================================== 6. QUICK START RECIPES
===================================================================================

## RECIPE 1: Person detection only (simplest)

Set allowed_class_names=["person"] in segmenter.segment_from_detections()

## RECIPE 2: People + vehicles

Set allowed_class_names=["person", "car", "truck", "bus"]

## RECIPE 3: Everything except small objects

Track only high-confidence detections: confidence=0.40

## RECIPE 4: Custom pipeline script

Copy scripts/run_multi_camera.py to scripts/run_person_only.py
Edit it to filter for persons only, then:
python scripts/run_person_only.py

===================================================================================
"""

print(**doc**)
