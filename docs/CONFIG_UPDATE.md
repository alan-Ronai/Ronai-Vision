#!/usr/bin/env python3
"""
UPDATED CONFIGURATION: All Classes by Default

# ✅ WHAT CHANGED

Your pipeline now:
• Detects and segments ALL 80 COCO classes by DEFAULT (no filtering)
• Only filters when you set ALLOWED_CLASSES environment variable
• Removed the USE_CLASS_FILTERING toggle (simpler logic)

This is the inverse of the previous behavior!

# ✅ QUICK START

1. RUN WITH ALL CLASSES (default):
   $ python scripts/run_multi_camera.py

    → Detects all 80 classes
    → Segments all detected objects
    → Tracks everything

2. FILTER TO SPECIFIC CLASSES:
   $ ALLOWED_CLASSES="person,car,dog" python scripts/run_multi_camera.py

    → Detects all 80 classes
    → Segments ONLY person, car, dog
    → Tracks everything

3. MULTIPLE FILTERS:
   $ ALLOWED_CLASSES="person,car,truck,bus" python scripts/run_multi_camera.py

# ✅ CODE CHANGES

Location: scripts/run_multi_camera.py (lines 34-45)

BEFORE:
ALLOWED_CLASSES = os.getenv("ALLOWED_CLASSES", "person").split(",")
USE_CLASS_FILTERING = os.getenv("USE_CLASS_FILTERING", "true").lower() ...

AFTER:
ALLOWED_CLASSES_STR = os.getenv("ALLOWED_CLASSES", "")
ALLOWED_CLASSES = ALLOWED_CLASSES_STR.split(",") if ALLOWED_CLASSES_STR else None

Logic (lines 355-371 for sequential, ~195-210 for parallel):

BEFORE:
if USE_CLASS_FILTERING:
seg = segmenter.segment_from_detections(...)
else:
seg = segmenter.segment(...)

AFTER:
if ALLOWED_CLASSES is not None:
seg = segmenter.segment_from_detections(...)
else:
seg = segmenter.segment(...)

# ✅ ENVIRONMENT VARIABLES

ALLOWED_CLASSES
• Default: "" (empty string → all classes)
• Type: comma-separated string
• Examples: - "" or unset → all 80 COCO classes - "person" → only person - "person,car,dog" → person, car, and dog - "person,bicycle,car,motorcycle,airplane,bus,train,truck" → vehicles + people

YOLO_CONFIDENCE
• Default: 0.25
• Type: float (0.0 to 1.0)
• Lower = more sensitive, more false positives
• Higher = more conservative, fewer detections

# ✅ AVAILABLE COCO CLASSES (80 total)

All classes YOLO can detect:
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

# ✅ PERFORMANCE IMPACT

DEFAULT (all classes):
→ Detects all 80 classes
→ SAM2 segments every detection (slow: ~20-30s per frame)
→ Good for exploration/research

WITH FILTER (e.g., "person,car"):
→ Detects all 80 classes (same speed)
→ SAM2 only segments person and car (fast: ~2-4s per frame)
→ Good for production/specific use cases

# ✅ COMMON USE CASES

People detection only:
$ ALLOWED_CLASSES="person" python scripts/run_multi_camera.py

People + vehicles:
$ ALLOWED_CLASSES="person,car,truck,bus,bicycle,motorcycle" python scripts/run_multi_camera.py

Animals only:
$ ALLOWED_CLASSES="cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe" python scripts/run_multi_camera.py

Furniture + appliances:
$ ALLOWED_CLASSES="chair,couch,potted plant,bed,dining table,toilet,tv,laptop,keyboard,microwave,oven" python scripts/run_multi_camera.py

Everything (default):
$ python scripts/run_multi_camera.py # No env var needed

# ✅ HOW THE LOGIC WORKS

1. ALLOWED_CLASSES = None (default)
   → All boxes sent to SAM2
   → All objects segmented
2. ALLOWED_CLASSES = ["person", "car"]
   → Only person/car boxes sent to SAM2
   → Other detections tracked but not segmented

# ✅ TESTING

Run the test suite:
$ bash scripts/test_config.sh

Output shows:
✓ Default is all classes
✓ Environment variable filtering works
✓ YOLO confidence tuning works
✓ SAM2 segmentation method available

# ✅ MIGRATION FROM OLD BEHAVIOR

If you had hardcoded ALLOWED_CLASSES="person" before:
OLD: python scripts/run_multi_camera.py # Segments only person
NEW: ALLOWED_CLASSES="person" python scripts/run_multi_camera.py # Segments only person

If you had hardcoded USE_CLASS_FILTERING=false before:
OLD: python scripts/run_multi_camera.py # Segments everything
NEW: python scripts/run_multi_camera.py # Segments everything (now default!)

# ✅ NEXT STEPS

1. Try the default behavior:
   $ python scripts/run_multi_camera.py
2. Monitor performance (will be slower with all classes):
   $ curl http://localhost:8000/api/status/perf
3. Switch to filtered mode for production:
   $ ALLOWED_CLASSES="person" python scripts/run_multi_camera.py
4. Adjust YOLO confidence if needed:
   $ ALLOWED_CLASSES="person" YOLO_CONFIDENCE=0.50 python scripts/run_multi_camera.py
   """

if **name** == "**main**":
print(**doc**)
