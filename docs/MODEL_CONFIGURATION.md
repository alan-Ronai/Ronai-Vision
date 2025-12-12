# Model Configuration Guide

## Checkpoints You've Provided

### 1. DeiT-TransReID for Vehicle ReID

**Filename**: `deit_transreid_vehicleID.pth`

**Location**: `models/deit_transreid_vehicleID.pth`

**Use Case**:

-   Cars, trucks, motorcycles, buses, vans
-   Highly optimized for vehicle appearance: color, shape, headlights, roof patterns
-   Based on DeiT (Data-efficient Image Transformers) architecture
-   Better generalization than CNN-based methods

**Features**:

-   ~100M parameters
-   512-D embedding space (optimized for vehicle features)
-   Works well on diverse vehicle types and viewing angles

### 2. CLIP Vision for Universal Objects

**Filename**: `clip-vit-base-patch32` (directory with processor files)

**Location**: `models/clip-vit-base-patch32-processor/`

**Use Case**:

-   Animals (dogs, cats, birds, etc.)
-   Equipment (tools, machinery, etc.)
-   Miscellaneous objects
-   Fallback for any class without specialized encoder

**Features**:

-   86M parameters
-   768-D embedding space
-   Pre-trained on 400M image-text pairs
-   Great for zero-shot understanding of new classes
-   Language-grounded (can understand descriptions)

### 3. OSNet for Person ReID (Reference)

**Filename**: `osnet_x0_5_imagenet.pth`

**Location**: `models/osnet_x0_5_imagenet.pth`

**Use Case**:

-   People / Person class only
-   Lightweight and fast
-   512-D embedding space
-   Highly specialized for human appearance

## Architecture Diagram

```
YOLO Detection (80 classes)
    ↓
Class-based Routing:
    ├─ person
    │  └─→ OSNet (512-D)        [person_reid.pth]
    │
    ├─ car, truck, motorcycle, bus, van
    │  └─→ DeiT-TransReID (512-D) [deit_transreid_vehicleID.pth]
    │
    └─ dog, cat, zebra, tool, ... (all others)
       └─→ CLIP Vision (768-D)    [clip-vit-base-patch32-processor/]
    ↓
Feature Alignment (pad to max dimension = 768)
    ↓
BoT-SORT Tracker (uses aligned features + motion)
    ↓
Cross-camera Deduplication (FAISS store)
```

## Performance Characteristics

| Aspect            | OSNet (Person)      | DeiT-TransReID (Vehicle) | CLIP (Universal)  |
| ----------------- | ------------------- | ------------------------ | ----------------- |
| Model Size        | ~4 MB               | ~50 MB                   | ~350 MB (weights) |
| Inference Time\*  | 15ms                | 25ms                     | 35ms              |
| Embedding Dim     | 512                 | 512                      | 768               |
| Specialization    | Person only         | Vehicles                 | Generic           |
| Accuracy (domain) | Excellent           | Excellent                | Good              |
| Generalization    | Poor outside person | Good for vehicle-like    | Excellent         |

\*Batch of 10 objects on MPS (macOS GPU)

## Configuration

### Environment Setup

```bash
# Device selection
export DEVICE=mps  # or cuda, or cpu

# Class filtering (optional - processes ALL classes if not set)
export ALLOWED_CLASSES=person,car,truck,dog

# YOLO confidence
export YOLO_CONFIDENCE=0.3

# Camera configuration
export CAMERA_CONFIG=config/camera_settings.json

# Optional: parallel processing per camera
export PARALLEL=true

# Optional: save frames to disk
export SAVE_FRAMES=true
```

### Testing Models

```python
from services.reid import get_multi_class_reid
import numpy as np

# Initialize
reid = get_multi_class_reid(device="mps")

# Create dummy detections
frame = np.zeros((720, 1280, 3), dtype=np.uint8)
boxes = np.array([
    [100, 100, 200, 300],   # person
    [300, 200, 500, 400],   # car
    [600, 300, 700, 400],   # dog
])
class_ids = np.array([0, 2, 16])  # COCO class IDs
class_names = ["person", "car", "dog"]  # or pull from YOLO

# Extract features
results = reid.extract_features(frame, boxes, class_ids, class_names)

# Results structure:
# {
#     "person": {"indices": [0], "features": (1, 512), "encoder": "person"},
#     "car": {"indices": [1], "features": (1, 512), "encoder": "vehicle"},
#     "dog": {"indices": [2], "features": (1, 768), "encoder": "universal"}
# }
```

## Troubleshooting

### CLIP Not Found

```
ImportError: No module named 'transformers'
```

**Solution**:

```bash
pip install transformers pillow
```

### Processor Path Error

```
FileNotFoundError: clip-vit-base-patch32-processor not found
```

**Solution**:
The system will auto-download from HuggingFace if local processor not found. This is slower on first run but works fine.

### TransReID Vehicle Not Loading

```
FileNotFoundError: deit_transreid_vehicleID.pth not found
```

**Solution**:

1. Verify file is at: `models/deit_transreid_vehicleID.pth`
2. Check file is not corrupted: `ls -lh models/deit_transreid_vehicleID.pth`
3. Vehicle encoder will gracefully fall back to CLIP Universal encoder

### OSNet Not Loading

```
ImportError: OSNet ReID is required but not available
```

**Solution**:

```bash
pip install torchreid
# Place osnet_x0_5_imagenet.pth in models/
```

## Model Comparison

### When to Use Each

**OSNet (Person)**:

-   You're tracking people in crowded scenes
-   Need fast, lightweight inference
-   Human appearance is the key signal

**DeiT-TransReID (Vehicle)**:

-   Tracking cars and vehicles
-   Need high accuracy on vehicle color/shape
-   Performance matters (videos at high FPS)

**CLIP (Universal)**:

-   Tracking animals, equipment, or miscellaneous objects
-   You don't have domain-specific models
-   Can tolerate slightly lower accuracy for flexibility

### Embedding Quality

```
Specialization (accuracy in domain):
OSNet > DeiT-TransReID > CLIP
└─ Each is optimized for its class

Generalization (works for new classes):
CLIP >> DeiT-TransReID > OSNet
└─ CLIP works for ANY class; others are domain-specific

Speed:
OSNet > CLIP > DeiT-TransReID
└─ OSNet is much lighter; TransReID is heavier
```

## Recommendations

### For Best Tracking Stability

1. **Use ALLOWED_CLASSES** to limit to classes you care about
2. **Tune YOLO_CONFIDENCE** to reduce false positives
3. **Enable PARALLEL mode** for multi-camera (much faster)
4. **Increase BoTSortTracker.lambda_app** for appearance-heavy matching

```python
# In services/tracker/bot_sort.py
tracker = BoTSortTracker(
    max_lost=30,
    lambda_app=0.85,  # 85% appearance, 15% motion (increased from 0.7)
    max_cost=0.7
)
```

### For Production Deployment

1. Use `DEVICE=cuda` for nvidia GPUs or `DEVICE=mps` for macOS
2. Set `YOLO_CONFIDENCE=0.4` (reduce false positives)
3. Use `ALLOWED_CLASSES=person,car` (most common uses)
4. Enable `PARALLEL=true` if you have multiple cameras
5. Monitor memory usage (`top` or `nvidia-smi`)

## Next Steps

1. ✅ Place checkpoint files in `models/`
2. ✅ Set environment variables
3. ✅ Run the pipeline
4. 🔍 Monitor tracking stability in MJPEG stream
5. 🎯 Tune confidence and class filters based on results
