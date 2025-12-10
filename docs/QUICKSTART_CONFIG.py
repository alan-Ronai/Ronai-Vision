#!/usr/bin/env python3
"""
QUICK START: How to Configure Detection and Segmentation Classes

The pipeline is now fully configurable via environment variables.
You do NOT need to modify any Python code!

===================================================================================
MOST COMMON USE CASES
===================================================================================

1. DEFAULT BEHAVIOR (Persons only)
   $ python scripts/run_multi_camera.py

   → Detects: all 80 COCO classes
   → Segments: only "person" class with SAM2
   → Tracks: all person detections


2. SEGMENT PERSONS & VEHICLES
   $ ALLOWED_CLASSES="person,car,truck,bus" python scripts/run_multi_camera.py

   → Segments: person, car, truck, bus
   → Other detections (bicycle, dog, etc.) are tracked but not segmented


3. SEGMENT ALL DETECTIONS (no class filtering)
   $ USE_CLASS_FILTERING=false python scripts/run_multi_camera.py

   → Segments: every detection, regardless of class
   → WARNING: Slow! SAM2 will run on all ~300 detected objects per frame


4. HIGHER YOLO CONFIDENCE (fewer false positives)
   $ YOLO_CONFIDENCE=0.50 python scripts/run_multi_camera.py

   → YOLO threshold raised from 0.25 to 0.50
   → Fewer detections, higher precision


5. LOWER YOLO CONFIDENCE (catch more objects)
   $ YOLO_CONFIDENCE=0.10 python scripts/run_multi_camera.py

   → YOLO threshold lowered to 0.10
   → More detections, may have more false positives


6. COMBINED: Multiple settings
   $ ALLOWED_CLASSES="person,dog,cat" YOLO_CONFIDENCE=0.35 python scripts/run_multi_camera.py

===================================================================================
ALL CONFIGURATION OPTIONS
===================================================================================

Option: ALLOWED_CLASSES
  Default: "person"
  Format: comma-separated list of class names
  Example: "person,car,truck,dog"
  Effect: Only these classes are segmented by SAM2

  Available classes (YOLO COCO dataset):
    person, bicycle, car, motorcycle, airplane, bus, train, truck, boat,
    traffic light, fire hydrant, stop sign, parking meter, bench, cat, dog,
    horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella,
    handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite,
    baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle,
    wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange,
    broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant,
    bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, microwave,
    oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear,
    hair drier, toothbrush


Option: YOLO_CONFIDENCE
  Default: 0.25
  Range: 0.0 - 1.0 (lower = more sensitive)
  Format: float
  Example: 0.50, 0.10, 0.75
  Effect: YOLO confidence threshold for detections

  Guidance:
    0.10 = very sensitive (catches small objects, more false positives)
    0.25 = default, good balance
    0.50 = conservative (fewer false positives, misses some objects)
    0.80 = very strict (only very confident detections)


Option: USE_CLASS_FILTERING
  Default: "true"
  Format: "true" or "false"
  Example: "false"
  Effect: Enable/disable SAM2 class filtering

  Guidance:
    true = efficient (only segment specified classes)
    false = segment everything (slow, ~300+ SAM2 runs per frame)


===================================================================================
ENVIRONMENT VARIABLES IN CODE
===================================================================================

These env vars are read at the top of scripts/run_multi_camera.py:

  Line 38: ALLOWED_CLASSES = os.getenv("ALLOWED_CLASSES", "person").split(",")
  Line 42: YOLO_CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", "0.25"))
  Line 47: USE_CLASS_FILTERING = os.getenv("USE_CLASS_FILTERING", "true").lower() in (...)

Then used in both sequential mode (line ~335) and parallel mode (line ~191):

  det = detector.predict(frame, confidence=YOLO_CONFIDENCE)

  if USE_CLASS_FILTERING:
      seg = segmenter.segment_from_detections(
          frame,
          boxes=boxes,
          class_ids=class_ids,
          class_names=det.class_names,
          allowed_class_names=ALLOWED_CLASSES
      )
  else:
      seg = segmenter.segment(frame, boxes=boxes)

===================================================================================
ADVANCED: WHERE TO MODIFY IF YOU WANT HARDCODED VALUES
===================================================================================

If you want to change defaults without using environment variables:

1. Edit the default values at lines 38-47 in scripts/run_multi_camera.py:

   # Current defaults:
   ALLOWED_CLASSES = os.getenv("ALLOWED_CLASSES", "person").split(",")

   # Change to (hardcoded):
   ALLOWED_CLASSES = ["person", "car", "dog"]


2. Then run normally:
   $ python scripts/run_multi_camera.py

===================================================================================
TESTING YOUR CONFIGURATION
===================================================================================

To verify your settings are loaded correctly:

1. Check the run_loop output for debug info (if you add it)

   # Add this after line 37 in run_loop():
   print(f"Pipeline configured: ALLOWED_CLASSES={ALLOWED_CLASSES}, "
         f"confidence={YOLO_CONFIDENCE}, filtering={USE_CLASS_FILTERING}")

2. Watch the console output during the first few frames
   → You should see class names being processed
   → If only "person" segments appear → filtering is working

3. Check performance metrics:
   $ curl http://localhost:8000/api/status/perf

   → segment_ms should be ~1-2s per frame with filtering
   → segment_ms should be ~20+ seconds with USE_CLASS_FILTERING=false

===================================================================================
TROUBLESHOOTING
===================================================================================

Q: I set ALLOWED_CLASSES but nothing changed?
A: Make sure you're using quotes: ALLOWED_CLASSES="person,car,dog" python ...

Q: The pipeline is very slow with USE_CLASS_FILTERING=false
A: Yes, SAM2 is slow. That's why class filtering is ON by default.
   Keep USE_CLASS_FILTERING=true and only add classes you need.

Q: Can I track only certain classes?
A: Not directly yet. Class filtering only affects segmentation.
   Tracking is per-camera and includes all detected objects.
   Feature request: I can add ALLOWED_TRACKING_CLASSES if needed.

Q: How do I see what classes were detected?
A: Add print statements in run_multi_camera.py at line ~325:
   print(f"Detected classes: {set(det.class_names[det.class_ids])}")

Q: What if I make a typo in the class name?
A: segment_from_detections() will silently filter to no boxes.
   You'll see masks become empty in the output.
   Check the spelling against the list above.

===================================================================================
"""

if __name__ == "__main__":
    print(__doc__)
    print("\nExample runs:\n")
    print("  # Default (person only):")
    print("  python scripts/run_multi_camera.py\n")
    print("  # People and vehicles:")
    print(
        '  ALLOWED_CLASSES="person,car,truck,bus" python scripts/run_multi_camera.py\n'
    )
    print("  # Everything (slow!):")
    print(
        "  USE_CLASS_FILTERING=false YOLO_CONFIDENCE=0.25 python scripts/run_multi_camera.py\n"
    )
    print("  # High precision people:")
    print("  YOLO_CONFIDENCE=0.50 python scripts/run_multi_camera.py\n")
