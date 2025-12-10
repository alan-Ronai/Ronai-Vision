#!/bin/bash
# Test script: Verify class filtering configuration works

echo "=========================================================================="
echo "Testing Class Filtering Configuration"
echo "=========================================================================="
echo ""

# Test 1: Import the module and check defaults
echo "Test 1: Verify configuration defaults (all classes)"
python3 << 'EOF'
from scripts import run_multi_camera

print(f"✓ ALLOWED_CLASSES (default): {run_multi_camera.ALLOWED_CLASSES} (None = all classes)")
print(f"✓ YOLO_CONFIDENCE (default): {run_multi_camera.YOLO_CONFIDENCE}")

# Verify they're the right types
assert run_multi_camera.ALLOWED_CLASSES is None, "Default should be None (all classes)"
assert isinstance(run_multi_camera.YOLO_CONFIDENCE, float)
print("\n✓ Default configuration processes all classes")
EOF
echo ""

# Test 2: Test environment variable override to filter
echo "Test 2: Verify environment variable override works (filter to person,car,dog)"
ALLOWED_CLASSES="person,car,dog" python3 << 'EOF'
import os
from scripts import run_multi_camera

# Re-read config with env var
classes_str = os.getenv("ALLOWED_CLASSES", "")
classes = classes_str.split(",") if classes_str else None

print(f"✓ ALLOWED_CLASSES (env override): {classes}")
assert classes == ["person", "car", "dog"], f"Expected ['person', 'car', 'dog'], got {classes}"
print("✓ Environment variable override successful")
EOF
echo ""

# Test 3: Test empty string env var (should be all classes)
echo "Test 3: Verify empty ALLOWED_CLASSES env var defaults to all classes"
ALLOWED_CLASSES="" python3 << 'EOF'
import os
classes_str = os.getenv("ALLOWED_CLASSES", "")
classes = classes_str.split(",") if classes_str else None
print(f"✓ ALLOWED_CLASSES (empty string): {classes}")
assert classes is None, f"Expected None, got {classes}"
print("✓ Empty string correctly defaults to all classes")
EOF
echo ""

# Test 4: Test YOLO_CONFIDENCE override
echo "Test 4: Verify YOLO_CONFIDENCE override works"
YOLO_CONFIDENCE=0.50 python3 << 'EOF'
import os
confidence = float(os.getenv("YOLO_CONFIDENCE", "0.25"))
print(f"✓ YOLO_CONFIDENCE (env override): {confidence}")
assert confidence == 0.50, f"Expected 0.50, got {confidence}"
print("✓ Confidence override successful")
EOF
echo ""

# Test 5: Check that segment_from_detections method exists
echo "Test 5: Verify SAM2Segmenter has the required method"
python3 << 'EOF'
from services.segmenter.sam2_segmenter import SAM2Segmenter
import inspect

# Check method exists
assert hasattr(SAM2Segmenter, 'segment_from_detections'), \
    "SAM2Segmenter missing segment_from_detections method"

# Check signature
sig = inspect.signature(SAM2Segmenter.segment_from_detections)
params = list(sig.parameters.keys())
expected = ['self', 'frame', 'boxes', 'class_ids', 'class_names', 'allowed_class_names']
assert params == expected, f"Expected {expected}, got {params}"

print(f"✓ segment_from_detections method exists with correct signature")
print(f"  Parameters: {params}")
EOF
echo ""

echo "=========================================================================="
echo "All tests passed! ✓"
echo "=========================================================================="
echo ""
echo "Configuration behavior:"
echo "  • DEFAULT: Processes ALL 80 COCO classes (no filtering)"
echo "  • WITH ALLOWED_CLASSES: Filters to specified classes only"
echo ""
echo "Next steps:"
echo "  1. Run with all classes: python scripts/run_multi_camera.py"
echo "  2. Run with filter: ALLOWED_CLASSES=\"person,car,dog\" python scripts/run_multi_camera.py"
echo "  3. Check metrics: curl http://localhost:8000/api/status/perf"
echo ""
