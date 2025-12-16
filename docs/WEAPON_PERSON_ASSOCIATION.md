# Weapon-Person Association System

## Overview

Automatic system that associates weapon detections with person detections and alerts armed individuals with visual indicators.

## Features

### 1. Spatial Association

-   Uses IoU (Intersection over Union) to associate weapon bounding boxes with person bounding boxes
-   Low IoU threshold (0.05) since weapons are typically small objects
-   Finds the person with highest overlap for each weapon

### 2. Visual Indicators

-   **Green boxes**: Normal person tracks
-   **Red boxes**: Armed person tracks
-   **[ARMED] label**: Added to track label for armed persons

### 3. Metadata Alerts

-   Automatic tagging with `'armed'` tag
-   Critical severity alert: `"Person armed with: [weapon types]"`
-   Attributes: `weapons_detected`, `weapon_detection_count`
-   **One-time alert**: Checks if already tagged to avoid redundant alerts

### 4. API Integration

-   Query armed persons: `GET /api/metadata/search?tag=armed`
-   Get specific track: `GET /api/metadata/track/{track_id}`
-   View all alerts: `GET /api/metadata/search?alert_type=armed_person`

## How It Works

### Detection Flow

```
1. YOLO detects persons + weapons
2. Tracker updates person tracks
3. Associate weapons with persons (IoU-based)
4. Check if person already tagged as 'armed'
5. If NOT tagged → Add alert + 'armed' tag
6. If tagged → Skip (no redundant alert)
7. Renderer checks 'armed' tag → Red box
```

### Code Flow

**Processor** (`services/pipeline/processor.py`):

```python
# After tracking
self._associate_weapons_and_alert(tracks, filtered_boxes, filtered_class_ids, class_names)
```

**Association Logic**:

```python
def _associate_weapons_with_persons(person_boxes, weapon_boxes, weapon_class_names):
    # For each weapon, find person with highest IoU
    # Return dict: {person_idx: [weapon_info, ...]}
```

**Alert Logic**:

```python
# Only add alert if not already tagged
if 'armed' not in track.metadata.get('tags', []):
    track.add_tag('armed')
    track.add_alert('armed_person', message, severity='critical')
```

**Renderer** (`services/output/renderer.py`):

```python
is_armed = 'armed' in track.metadata.get('tags', [])
box_color = (0, 0, 255) if is_armed else self.bbox_color  # Red : Green
```

## Configuration

### Weapon Detection Classes

The system looks for these keywords in class names:

-   `pistol`
-   `rifle`
-   `gun`
-   `knife`
-   `weapon`
-   `firearm`

### IoU Threshold

```python
# In processor._associate_weapons_and_alert()
associations = _associate_weapons_with_persons(
    person_boxes,
    weapon_boxes,
    weapon_names,
    iou_threshold=0.05  # Adjust if needed
)
```

### Enable Weapon Detection

Set in environment or `config/pipeline_config.py`:

```python
ENABLE_WEAPON_DETECTION = True
WEAPON_MODEL = "models/firearm-yolov8n.pt"
WEAPON_CONFIDENCE = 0.5
```

## Usage Examples

### Query Armed Persons (API)

```bash
# Get all armed persons
curl http://localhost:8000/api/metadata/search?tag=armed

# Response:
{
  "total_matches": 1,
  "results": [
    {
      "track_id": 42,
      "metadata": {
        "tags": ["person", "armed"],
        "alerts": [
          {
            "type": "armed_person",
            "message": "Person armed with: pistol",
            "severity": "critical",
            "timestamp": 1234567890.5
          }
        ],
        "attributes": {
          "weapons_detected": ["pistol"],
          "weapon_detection_count": 1
        }
      }
    }
  ]
}
```

### Check in Python

```python
from services.tracker.metadata_manager import get_metadata_manager

manager = get_metadata_manager()

# Search for armed persons
armed_tracks = manager.search_metadata(tag='armed')
print(f"Found {len(armed_tracks)} armed persons")

# Get critical alerts
critical_alerts = manager.search_metadata(alert_type='armed_person')
for track_data in critical_alerts:
    tid = track_data['track_id']
    alerts = track_data['metadata']['alerts']
    print(f"Track {tid}: {len(alerts)} alerts")
```

### Manual Association (Testing)

```python
from services.pipeline.processor import _associate_weapons_with_persons

person_boxes = np.array([[100, 100, 200, 400]])
weapon_boxes = np.array([[150, 200, 180, 250]])
weapon_names = ['pistol']

associations = _associate_weapons_with_persons(
    person_boxes,
    weapon_boxes,
    weapon_names,
    iou_threshold=0.05
)

# associations = {0: [{'box': [...], 'class': 'pistol', 'iou': 0.123}]}
```

## Visual Examples

### Normal Person (Green Box)

```
┌─────────────────┐
│ ID:42 person    │ ← Green box
│   0.95          │
│                 │
│       ●         │ ← Green centroid
│                 │
└─────────────────┘
```

### Armed Person (Red Box)

```
┌─────────────────┐
│ [ARMED]         │ ← Red box + label
│ ID:42 person    │
│   0.95          │
│                 │
│       ●         │ ← Red centroid
│    🔫          │ ← Weapon detected
└─────────────────┘
```

## Performance Considerations

### Timing

-   Weapon association: ~1-2ms per frame
-   No performance impact if no weapons detected
-   IoU calculation is O(N×M) where N=persons, M=weapons

### Memory

-   No additional memory overhead
-   Metadata stored in existing Track objects
-   Alert added once per track (not per frame)

### Optimization

```python
# Skip association if no weapons or persons
if len(weapon_boxes) == 0 or person_class_id is None:
    return  # Fast exit
```

## Troubleshooting

### No Associations Found

**Problem**: Weapons detected but not associated with persons

**Solutions**:

1. Lower IoU threshold:

    ```python
    iou_threshold=0.01  # Very lenient
    ```

2. Check weapon class names:

    ```python
    # Add custom keywords
    weapon_keywords = ['pistol', 'rifle', 'your_weapon_class']
    ```

3. Verify weapon box overlaps person box:
    ```python
    # Print IoU values for debugging
    for person_idx, person_box in enumerate(person_boxes):
        for weapon_idx, weapon_box in enumerate(weapon_boxes):
            iou = _iou(person_box, weapon_box)
            print(f"Person {person_idx} ↔ Weapon {weapon_idx}: IoU={iou:.4f}")
    ```

### Duplicate Alerts

**Problem**: Same person getting multiple armed alerts

**Check**:

```python
# This check should prevent duplicates
if 'armed' in track.metadata.get('tags', []):
    continue  # Skip if already alerted
```

### Red Box Not Showing

**Problem**: Armed tag exists but box still green

**Check renderer code**:

```python
# Ensure this check is working
is_armed = 'armed' in track.metadata.get('tags', [])
print(f"Track {track.track_id} armed status: {is_armed}")
```

### Weapons Not Detected

**Problem**: MultiDetector not finding weapons

**Solutions**:

1. Check weapon detector is loaded:

    ```python
    from services.detector.detector_factory import get_detector_info
    info = get_detector_info(detector)
    print(info)  # Should show weapon_detection_enabled=True
    ```

2. Lower weapon confidence:

    ```python
    WEAPON_CONFIDENCE = 0.3  # More sensitive
    ```

3. Verify model file exists:
    ```bash
    ls -lh models/firearm-yolov8n.pt
    ```

## Testing

Run test suite:

```bash
python scripts/test_weapon_person_association.py
```

Expected output:

```
✅ ALL TESTS PASSED

System features working:
  ✓ Weapon-person association via IoU
  ✓ Automatic 'armed' tagging
  ✓ One-time alert (no duplicates)
  ✓ Red box rendering for armed persons
  ✓ Metadata persistence
```

## Integration with Existing Systems

### Multi-Camera Tracking

-   Armed status persists across camera views via ReID
-   Global track ID maintains armed tag
-   Cross-camera queries: `GET /api/metadata/search?tag=armed`

### PTZ Control

```python
# Auto-focus on armed persons
armed_tracks = [t for t in tracks if 'armed' in t.metadata.get('tags', [])]
if armed_tracks:
    # Priority target
    ptz.track_target(armed_tracks[0].box)
```

### Alerts & Notifications

```python
# Send alert when armed person detected
for track in tracks:
    metadata = track.metadata
    alerts = metadata.get('alerts', [])

    for alert in alerts:
        if alert['type'] == 'armed_person' and alert.get('sent') != True:
            send_notification(f"ALERT: {alert['message']}")
            alert['sent'] = True  # Mark as sent
```

## Future Enhancements

1. **Confidence Scoring**: Weight association by detection confidence
2. **Temporal Tracking**: Track how long person has been armed
3. **Weapon Type Filtering**: Different colors for different weapon types
4. **Proximity Alerts**: Alert if armed person approaches restricted zone
5. **Action Recognition**: Detect if weapon is being aimed/used
6. **Multi-Person**: Handle scenarios with multiple armed persons

## Files Modified

-   `services/pipeline/processor.py`: Association logic
-   `services/output/renderer.py`: Red box rendering
-   `scripts/test_weapon_person_association.py`: Test suite
-   `docs/WEAPON_PERSON_ASSOCIATION.md`: This documentation
