# Weapon Detection Integration Guide

## Overview

The weapon detection system is fully integrated into the pipeline and can be enabled with a simple configuration flag. When enabled, the system uses a multi-detector architecture that combines:

1. **Primary Detector** (YOLO12n/YOLOv8): Standard object detection (people, vehicles, etc.)
2. **Weapon Detector** (Firearm YOLOv8n): Specialized firearm detection (pistols, rifles, armed persons)

## Quick Start

### Enable Weapon Detection

Set the environment variable before running:

```bash
export ENABLE_WEAPON_DETECTION=true
python scripts/run_multi_camera.py
```

Or in your `.env` file:

```bash
ENABLE_WEAPON_DETECTION=true
WEAPON_CONFIDENCE=0.5
```

### Disable Weapon Detection (Default)

```bash
export ENABLE_WEAPON_DETECTION=false
# or simply omit the variable (defaults to false)
python scripts/run_multi_camera.py
```

## Configuration Options

All weapon detection settings can be configured via environment variables or in your `.env` file:

### Environment Variables

| Variable                  | Default                     | Description                                         |
| ------------------------- | --------------------------- | --------------------------------------------------- |
| `ENABLE_WEAPON_DETECTION` | `false`                     | Enable/disable weapon detection                     |
| `WEAPON_MODEL`            | `models/firearm-yolov8n.pt` | Path to weapon detection model                      |
| `WEAPON_CONFIDENCE`       | `0.5`                       | Confidence threshold for weapon detection (0.0-1.0) |
| `WEAPON_REID_TRACKING`    | `false`                     | Enable ReID tracking for weapons (not recommended)  |

### Example Configuration Files

#### Development Environment (`config/dev.env`)

```bash
# Standard detection
YOLO_MODEL=yolo12n.pt
YOLO_CONFIDENCE=0.4
ALLOWED_CLASSES=person,car,truck

# Weapon detection
ENABLE_WEAPON_DETECTION=true
WEAPON_CONFIDENCE=0.5

# Performance
DEVICE=mps
DETECTION_SKIP=1
```

#### Production Environment (`config/prod.env`)

```bash
# High-confidence standard detection
YOLO_MODEL=yolo12n.pt
YOLO_CONFIDENCE=0.5
ALLOWED_CLASSES=person,car,truck,motorcycle

# Strict weapon detection (fewer false positives)
ENABLE_WEAPON_DETECTION=true
WEAPON_CONFIDENCE=0.6

# Performance
DEVICE=cuda
DETECTION_SKIP=1
PARALLEL=true
```

## How It Works

### Multi-Detector Architecture

When weapon detection is enabled, the system automatically switches from single-detector to multi-detector mode:

```python
# Single detector (default)
detector = YOLODetector("yolo12n.pt")

# Multi-detector (weapon detection enabled)
detector = MultiDetector({
    "primary": {
        "model": "yolo12n.pt",
        "confidence": 0.4,
        "priority": 1
    },
    "weapon": {
        "model": "models/firearm-yolov8n.pt",
        "confidence": 0.5,
        "priority": 2  # Higher priority for alerts
    }
})
```

### Detection Classes

#### Primary Detector (COCO 80 classes)

-   person, car, truck, motorcycle, bicycle
-   bus, train, boat, airplane
-   And 70+ other common objects

#### Weapon Detector (3 classes)

-   `pistol` - Handguns, sidearms
-   `rifle` - Long guns, assault rifles
-   `person-with-firearm` - Person actively holding/carrying weapon

### Priority System

Detections are prioritized for alerts and display:

-   Priority 2: Weapons (highest)
-   Priority 1: Standard objects (lower)

High-priority detections can trigger special alerts in your monitoring system.

## Usage Examples

### 1. Enable for Multi-Camera System

```bash
# Set environment
export ENABLE_WEAPON_DETECTION=true
export WEAPON_CONFIDENCE=0.5

# Run multi-camera system
python scripts/run_multi_camera.py
```

Output:

```
[INFO] Using device: mps
[INFO] Initializing multi-detector with weapon detection
  - Primary: yolo12n.pt (conf=0.4)
  - Weapon: models/firearm-yolov8n.pt (conf=0.5)
[INFO] Detector type: multi
[INFO] Weapon detection: ENABLED
[INFO] Active detectors: primary, weapon
```

### 2. Test on Single Video

```bash
export ENABLE_WEAPON_DETECTION=true
python scripts/run_pipeline.py
```

### 3. Test on Image

```bash
python scripts/test_weapon_detection.py path/to/image.jpg
```

### 4. Adjust Sensitivity

Higher confidence = fewer false positives, might miss some weapons:

```bash
export WEAPON_CONFIDENCE=0.7  # Strict (recommended for production)
```

Lower confidence = catch more weapons, more false positives:

```bash
export WEAPON_CONFIDENCE=0.3  # Sensitive (good for testing)
```

## Automatic Model Download

If the weapon model is not found, it will be automatically downloaded from HuggingFace:

```
[WARNING] Weapon model not found at models/firearm-yolov8n.pt, downloading...
Downloading firearm-yolov8n.pt: 100%|████████| 6.24M/6.24M [00:03<00:00, 2.14MB/s]
[INFO] Weapon model downloaded successfully
```

Manual download:

```bash
python scripts/setup_weapon_detection.py
```

## Performance Considerations

### Model Size

-   Primary (YOLO12n): ~6 MB
-   Weapon (firearm-yolov8n): ~6 MB
-   **Total**: ~12 MB (both models loaded in memory)

### Speed Impact

-   Single detector: ~30 FPS
-   Multi-detector: ~25 FPS (~15% slower)
-   Recommendation: Use `DETECTION_SKIP=2` for real-time on slower devices

### Memory Usage

-   Single detector: ~500 MB VRAM
-   Multi-detector: ~800 MB VRAM

## Integration with Tracking

### ReID Behavior

By default:

-   **People**: Full ReID tracking (cross-camera identification)
-   **Vehicles**: Motion-based tracking only
-   **Weapons**: Motion-based tracking only (no ReID)

This means:

-   Each weapon detection is treated independently per frame
-   Weapons are not given persistent IDs across cameras
-   Focus is on "weapon present" alerts, not "track specific gun #5"

To enable weapon ReID tracking (not recommended):

```bash
export WEAPON_REID_TRACKING=true
```

## Monitoring and Alerts

### Detection Output Format

Each detection includes:

```json
{
    "class_name": "pistol",
    "confidence": 0.87,
    "bbox": [120, 340, 180, 420],
    "priority": 2,
    "detector": "weapon"
}
```

### Alert Triggers

High-priority weapon detections can trigger:

1. WebSocket notifications to connected clients
2. Database logging for incident reports
3. Email/SMS alerts (configure in your alert service)

## Troubleshooting

### Model Not Downloading

```bash
# Manual download
python scripts/setup_weapon_detection.py

# Check model exists
ls -lh models/firearm-yolov8n.pt
```

### Low Detection Rate

Try lowering confidence:

```bash
export WEAPON_CONFIDENCE=0.3
```

### Too Many False Positives

Try raising confidence:

```bash
export WEAPON_CONFIDENCE=0.7
```

### Performance Issues

Reduce detection frequency:

```bash
export DETECTION_SKIP=2  # Detect every other frame
```

## Testing

### Unit Test

```bash
python scripts/test_weapon_detection.py test_image.jpg
```

### Integration Test

```bash
# Enable weapon detection
export ENABLE_WEAPON_DETECTION=true

# Run on test video
python scripts/run_pipeline.py

# Check output for weapon detections
ls output/samples/
```

### Verify Configuration

```bash
python -c "from config.pipeline_config import PipelineConfig; PipelineConfig.print_summary()"
```

## Advanced: Adding More Weapon Models

You can register additional weapon detection models:

1. Add to `services/detector/weapon_models.py`:

```python
WEAPON_MODELS = {
    "firearm-yolov8n": {
        "repo_id": "Subh775/Firearm_Detection_Yolov8n",
        "filename": "firearm-yolov8n.pt",
        "classes": ["pistol", "rifle", "person-with-firearm"]
    },
    "knife-detector": {
        "repo_id": "YourRepo/knife-detection",
        "filename": "knife-yolov8n.pt",
        "classes": ["knife", "machete", "sword"]
    }
}
```

2. Configure in detector factory:

```python
detector_config = {
    "primary": {...},
    "firearm": {...},
    "knife": {
        "model": "models/knife-yolov8n.pt",
        "confidence": 0.5,
        "priority": 2
    }
}
```

## Summary

-   ✅ Zero-config weapon detection: just set `ENABLE_WEAPON_DETECTION=true`
-   ✅ Automatic model download from HuggingFace
-   ✅ Backward compatible: defaults to single detector if disabled
-   ✅ Flexible confidence thresholds per detector
-   ✅ Priority system for critical detections
-   ✅ Works with all existing scripts and services

For questions or issues, check the test scripts in `scripts/test_weapon_detection.py`.
