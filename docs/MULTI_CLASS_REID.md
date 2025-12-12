# Multi-Class ReID Architecture

## Overview

The pipeline now supports **specialized ReID encoders for different object classes**, rather than using a single encoder for all objects.

### Why Multiple ReID Models?

ReID models are trained on specific object domains:

-   **OSNet**: Trained exclusively on person datasets (Market1501, CUHK03, MSMT17)
-   **TransReID-Vehicle**: Trained on vehicle datasets (VERI-Wild, AI City, etc.)
-   **Universal Encoder**: Pre-trained vision models for arbitrary object classes

Using the right encoder for each class dramatically improves tracking accuracy and stability.

## Architecture

```
YOLO Detection
    ↓ (outputs boxes + class_ids)
    ↓
MultiClassReID Dispatcher
    ├─ [person] → OSNet (lightweight, fast, optimized for humans)
    ├─ [car/truck/motorcycle] → TransReID-Vehicle (specialized for vehicles)
    └─ [other classes] → Universal ViT/CNN encoder (fallback)
    ↓ (outputs class-specific embeddings)
    ↓
BoT-SORT Tracker
    ↓ (uses embeddings + motion for stable IDs)
    ↓
Per-camera & cross-camera Re-identification
```

## Component Details

### 1. OSNet (Person ReID)

-   **Location**: `services/reid/osnet_reid.py`
-   **Model**: OSNet (x0.5 variant)
-   **Input**: Person crops (any aspect ratio)
-   **Output**: 512-dimensional embedding
-   **Performance**: Lightweight, ~100ms for batch of 10
-   **Checkpoint**: `models/osnet_x0_5_imagenet.pth`

### 2. TransReID-Vehicle (Vehicle ReID)

-   **Location**: `services/reid/transreid_vehicle.py`
-   **Model**: TransReID (Transformer-based)
-   **Input**: Vehicle crops (optimized for ~2:1 aspect ratio)
-   **Output**: 768-dimensional embedding
-   **Performance**: ~200ms for batch of 10
-   **Checkpoint**: `models/transreid_vehicle.pth` (user provides)
-   **Supports**: Cars, trucks, vans, motorcycles, buses, and generic "vehicle" class

### 3. Universal ReID (Fallback)

-   **Location**: `services/reid/universal_reid.py`
-   **Model**: Pre-trained Vision Transformer (ViT-Base) from timm
-   **Input**: Any object crop (auto-resized to 224×224)
-   **Output**: 768-dimensional embedding
-   **Performance**: ~150ms for batch of 10
-   **Classes**: Animals, tools, equipment, or any unspecialized class
-   **Note**: Less accurate than domain-specific models, but works for any object

## Usage

### Running the Pipeline

The runner automatically uses multi-class ReID:

```bash
START_RUNNER=true uvicorn api.server:app --host 0.0.0.0 --port 8000
```

### Programmatic Usage

```python
from services.reid import get_multi_class_reid

# Create dispatcher
reid = get_multi_class_reid(device="mps")

# Extract features for mixed detections
# This automatically routes each class to the right encoder
reid_results = reid.extract_features(
    frame=frame,
    boxes=boxes,           # (N, 4) array
    class_ids=class_ids,   # (N,) array
    class_names=class_names  # list of class name strings
)

# reid_results = {
#     "person": {"indices": [0, 1], "features": (2, 512), "encoder": "person"},
#     "car": {"indices": [2, 3, 4], "features": (3, 768), "encoder": "vehicle"},
#     "dog": {"indices": [5], "features": (1, 768), "encoder": "universal"}
# }
```

## Feature Dimension Alignment

Different encoders output different feature dimensions:

-   OSNet → 512-D
-   TransReID-Vehicle → 768-D
-   Universal ViT → 768-D

The runner automatically pads smaller embeddings to match the maximum dimension, ensuring compatibility with the tracker.

## Configuration

### Environment Variables

```bash
# Device selection (cpu, cuda, mps)
export DEVICE=mps

# Class filtering (which classes to detect/segment)
export ALLOWED_CLASSES=person,car,dog

# YOLO confidence threshold
export YOLO_CONFIDENCE=0.3

# Parallel processing per camera
export PARALLEL=true

# Camera config file
export CAMERA_CONFIG=config/camera_settings.json
```

### Required Checkpoints

Place these in the `models/` directory:

1. **`osnet_x0_5_imagenet.pth`** (user provides)

    - Download from: torchreid official releases
    - Size: ~4 MB

2. **`transreid_vehicle.pth`** (user provides)

    - Download from: TransReID official repo or AI City Challenge
    - Size: ~50-100 MB
    - **CRITICAL**: This is what you'll provide

3. SAM2, YOLO, and other models are auto-downloaded by ultralytics/sam2

## Performance Impact

-   **OSNet (person)**: +100ms per frame (batch of ~10 people)
-   **TransReID (vehicle)**: +200ms per frame (batch of ~10 vehicles)
-   **Universal (other)**: +150ms per frame (batch of ~10 objects)

All three run in parallel when needed; the pipeline only runs extractors for classes present in the frame.

## Fallback Behavior

If a checkpoint is missing:

-   **OSNet missing**: Person class uses Universal encoder instead (less accurate)
-   **TransReID missing**: Vehicle classes use Universal encoder instead (less accurate)
-   **Universal missing**: Pipeline fails (this is the required fallback)

All three models gracefully degrade if imports fail.

## Extending to New Classes

To add specialized ReID for a new class (e.g., animals):

1. Create `services/reid/animal_reid.py` with the same interface as OSNetReID
2. Update `MultiClassReID.ANIMAL_CLASSES` set in `services/reid/__init__.py`
3. Add initialization in `MultiClassReID.__init__()`
4. Update the dispatcher logic in `get_encoder_for_class()`

## Tracking Behavior

The BoT-SORT tracker uses:

-   **Motion**: Centroid distance (primary signal for non-specialized classes)
-   **Appearance**: Cosine distance of embeddings (primary signal for people/vehicles)
-   **Assignment**: Hungarian algorithm with combined cost

For better tracking of vehicles, consider increasing `lambda_app` in BoTSortTracker (currently 0.7).

## Next Steps

1. Obtain TransReID-Vehicle checkpoint and place in `models/transreid_vehicle.pth`
2. Run pipeline: `START_RUNNER=true uvicorn api.server:app --host 0.0.0.0 --port 8000`
3. Monitor tracking stability via profiler and MJPEG stream
4. Tune `ALLOWED_CLASSES` and `YOLO_CONFIDENCE` as needed
