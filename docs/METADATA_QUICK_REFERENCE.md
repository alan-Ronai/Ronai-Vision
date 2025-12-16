# Track Metadata System - Quick Reference

## Overview

Comprehensive system for annotating tracked objects with detailed contextual information. Metadata persists beyond track lifetime and is accessible via REST API.

## Quick Start

### 1. Add Metadata to Tracks (Python)

```python
from services.tracker.metadata_manager import get_metadata_manager

# In your tracking loop
tracks = tracker.update(boxes, class_ids, confidences, features)
manager = get_metadata_manager()

for track in tracks:
    # Add note
    track.add_note("Person entered from north entrance", author="operator_1")

    # Add tags
    track.add_tag("suspicious")

    # Set attributes
    track.set_attribute("color", "blue_shirt")

    # Add alert
    track.add_alert("zone_violation", "Entered restricted area", severity="critical")

    # Set behavior
    track.set_behavior("loitering", confidence=0.85)

    # Add zone
    track.add_zone("zone_a")

    # Sync to manager
    manager.update_track_metadata(track.track_id, track.class_id, track.get_metadata_summary())
```

### 2. Query Metadata (API)

```bash
# Get all metadata
curl http://localhost:8000/api/metadata/all

# Get specific track
curl http://localhost:8000/api/metadata/track/42

# Get all tracks of a class
curl http://localhost:8000/api/metadata/class/0  # class 0 = person

# Search by tag
curl http://localhost:8000/api/metadata/search?tag=suspicious

# Search by zone
curl http://localhost:8000/api/metadata/search?zone=restricted_area

# Search by alert type
curl http://localhost:8000/api/metadata/search?alert_type=weapon_detected

# Get statistics
curl http://localhost:8000/api/metadata/stats
```

### 3. Update Metadata (API)

```bash
# Add note
curl -X POST http://localhost:8000/api/metadata/track/42/note \
  -H "Content-Type: application/json" \
  -d '{"text": "Person carrying backpack", "author": "operator_1"}'

# Add tag
curl -X POST http://localhost:8000/api/metadata/track/42/tag \
  -H "Content-Type: application/json" \
  -d '{"tag": "high_priority"}'

# Add alert
curl -X POST http://localhost:8000/api/metadata/track/42/alert \
  -H "Content-Type: application/json" \
  -d '{
    "alert_type": "weapon_detected",
    "message": "Pistol detected at entrance",
    "severity": "critical"
  }'

# Set attribute
curl -X POST http://localhost:8000/api/metadata/track/42/attribute \
  -H "Content-Type: application/json" \
  -d '{"key": "vehicle_color", "value": "blue"}'
```

## Track Metadata Fields

-   **created_at**: Timestamp when track was created
-   **updated_at**: Last update timestamp
-   **notes**: Array of timestamped notes with author
-   **tags**: Array of string tags
-   **attributes**: Key-value pairs for custom data
-   **alerts**: Array of alerts with type, message, severity, timestamp
-   **zones_visited**: Set of zones/areas visited
-   **behavior**: Detected behavior with type, confidence, timestamp
-   **custom**: Flexible dictionary for any custom metadata

## Common Use Cases

### Weapon Detection

```python
if weapon_detected:
    track.add_alert("weapon_detected", f"{weapon_type} detected", severity="critical")
    track.add_tag("armed")
    track.set_attribute("weapon_type", weapon_type)
```

### Zone Monitoring

```python
current_zone = get_zone_for_box(track.box)
if current_zone:
    track.add_zone(current_zone)
    if current_zone in RESTRICTED_ZONES:
        track.add_alert("zone_violation", f"Entered {current_zone}", severity="warning")
```

### Behavioral Analysis

```python
behavior, confidence = detect_behavior(track)
if confidence > 0.7:
    track.set_behavior(behavior, confidence)
    if behavior == "loitering":
        track.add_alert("suspicious_behavior", "Loitering detected", severity="info")
```

## Management

```bash
# Save metadata to disk
curl -X POST http://localhost:8000/api/metadata/save

# Clean up expired tracks
curl -X POST http://localhost:8000/api/metadata/cleanup

# Clear all metadata (dangerous!)
curl -X DELETE http://localhost:8000/api/metadata/clear
```

## Configuration

```python
from services.tracker.metadata_manager import MetadataManager

# Custom TTL (default 1 hour)
manager = MetadataManager(ttl=1800)  # 30 minutes

# Custom persistence path
manager = MetadataManager(persistence_path="custom/path/metadata.json")
```

## Testing

```bash
# Run test script
python scripts/test_metadata_system.py

# Test formatted output only
python scripts/test_metadata_system.py --format
```

## Best Practices

1. **Use tags for categorical data**: `track.add_tag("suspicious")`
2. **Use attributes for structured data**: `track.set_attribute("color", "blue")`
3. **Use alerts for important events**: `track.add_alert("weapon_detected", "...", severity="critical")`
4. **Add context to notes**: Include who, what, when, where
5. **Sync metadata regularly**: Call `manager.update_track_metadata()` after changes
6. **Clean up periodically**: Run cleanup every hour to free memory
7. **Save before shutdown**: `manager.save_metadata()` before stopping server

## API Endpoints Summary

| Endpoint                             | Method | Description                          |
| ------------------------------------ | ------ | ------------------------------------ |
| `/api/metadata/track/{id}`           | GET    | Get metadata for specific track      |
| `/api/metadata/class/{id}`           | GET    | Get all tracks of a class            |
| `/api/metadata/search`               | GET    | Search by tag, zone, alert, behavior |
| `/api/metadata/all`                  | GET    | Get all metadata                     |
| `/api/metadata/stats`                | GET    | Get statistics                       |
| `/api/metadata/track/{id}/note`      | POST   | Add note                             |
| `/api/metadata/track/{id}/tag`       | POST   | Add tag                              |
| `/api/metadata/track/{id}/tag`       | DELETE | Remove tag                           |
| `/api/metadata/track/{id}/alert`     | POST   | Add alert                            |
| `/api/metadata/track/{id}/attribute` | POST   | Set attribute                        |
| `/api/metadata/track/{id}/behavior`  | POST   | Set behavior                         |
| `/api/metadata/track/{id}/zone`      | POST   | Add zone                             |
| `/api/metadata/cleanup`              | POST   | Clean up expired                     |
| `/api/metadata/save`                 | POST   | Save to disk                         |
| `/api/metadata/clear`                | DELETE | Clear all                            |

## Example Response

```json
{
    "track_id": 42,
    "metadata": {
        "created_at": 1234567890.0,
        "updated_at": 1234567891.0,
        "notes": [
            {
                "text": "Person carrying backpack",
                "author": "operator_1",
                "timestamp": 1234567890.5
            }
        ],
        "tags": ["suspicious", "high_priority"],
        "attributes": {
            "color": "blue_shirt",
            "backpack": true
        },
        "alerts": [
            {
                "type": "suspicious_behavior",
                "message": "Loitering near entrance",
                "severity": "warning",
                "timestamp": 1234567890.8
            }
        ],
        "zones_visited": ["entrance", "lobby"],
        "behavior": {
            "type": "loitering",
            "confidence": 0.85,
            "detected_at": 1234567890.9
        },
        "custom": {}
    },
    "formatted": "=== Track 42 Metadata ===\n..."
}
```

## Full Documentation

See [METADATA_SYSTEM.md](METADATA_SYSTEM.md) for complete documentation.
