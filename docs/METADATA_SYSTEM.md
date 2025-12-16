# Track Metadata System

Comprehensive metadata system for annotating and tracking detailed context about tracked objects.

## Overview

The metadata system provides:

-   **Rich annotations**: Add notes, tags, alerts, and custom attributes to any tracked object
-   **Persistent storage**: Metadata survives beyond track lifetime (configurable TTL)
-   **API access**: Query and manage metadata through REST endpoints
-   **Automatic sync**: Metadata automatically synchronized from Track objects to persistent storage
-   **Search capabilities**: Find tracks by tags, alerts, zones, or behavior

## Use Cases

### Security/Surveillance

-   **Weapon associations**: Note which person is carrying which weapon
-   **Zone violations**: Track which restricted areas a person has entered
-   **Behavioral alerts**: Mark suspicious behavior (loitering, running, etc.)
-   **High-risk tracking**: Tag high-priority targets for enhanced monitoring

### Traffic Monitoring

-   **Vehicle history**: Track which zones/cameras a vehicle has passed through
-   **Violation tracking**: Note traffic violations, speeding, wrong-way driving
-   **License plate notes**: Associate plate numbers with vehicles
-   **Dwell time**: Track how long vehicles spend in certain areas

### Retail Analytics

-   **Customer behavior**: Note shopping patterns, time spent in sections
-   **VIP tracking**: Tag important customers for personalized service
-   **Zone analytics**: Track which areas customers visit most

## Architecture

### Components

1. **Track.metadata** - Metadata dictionary on each Track object
2. **MetadataManager** - Centralized storage and persistence layer
3. **API Routes** - REST endpoints for querying/updating metadata
4. **Automatic Sync** - Tracker updates MetadataManager on each frame

### Data Flow

```
Track Object → MetadataManager → Persistent Storage (JSON)
     ↑              ↓
     └──────────────┘
    (Sync on update)

API Endpoints ← → MetadataManager ← → Track Metadata
```

## Track Metadata Structure

Each track has the following metadata fields:

```python
{
    "created_at": 1234567890.0,      # Unix timestamp when track created
    "updated_at": 1234567890.5,      # Last update timestamp
    "notes": [                        # User notes/annotations
        {
            "text": "Person carrying backpack",
            "author": "operator_1",
            "timestamp": 1234567890.2
        }
    ],
    "tags": ["suspicious", "high_priority"],  # Custom tags
    "attributes": {                   # Key-value attributes
        "color": "blue",
        "vehicle_type": "sedan",
        "license_plate": "ABC-1234"
    },
    "alerts": [                       # Alert history
        {
            "type": "weapon_detected",
            "message": "Pistol detected at 14:23:45",
            "severity": "critical",
            "timestamp": 1234567890.3
        }
    ],
    "zones_visited": ["zone_a", "zone_b"],  # Zones/areas visited
    "behavior": {                     # Detected behavior
        "type": "loitering",
        "confidence": 0.85,
        "detected_at": 1234567890.4
    },
    "custom": {                       # Flexible custom metadata
        "any_key": "any_value"
    }
}
```

## API Usage

### Query Metadata

#### Get metadata for specific track

```bash
GET /api/metadata/track/{track_id}
```

Response:

```json
{
    "track_id": 42,
    "metadata": { ... },
    "formatted": "=== Track 42 Metadata ===\nCreated: 2024-01-15 14:23:45\n..."
}
```

#### Get all tracks of a class

```bash
GET /api/metadata/class/{class_id}
# Example: GET /api/metadata/class/0  (all people)
```

#### Search metadata

```bash
GET /api/metadata/search?tag=suspicious&alert_type=weapon_detected
GET /api/metadata/search?zone=restricted_area
GET /api/metadata/search?behavior=loitering
```

#### Get all metadata

```bash
GET /api/metadata/all
```

#### Get statistics

```bash
GET /api/metadata/stats
```

Response:

```json
{
    "total_tracks": 150,
    "total_notes": 45,
    "total_alerts": 12,
    "tracks_by_class": {
        "0": 120, // people
        "2": 30 // cars
    },
    "recent_tracks": 25,
    "oldest_track_age": 3600.5
}
```

### Update Metadata

#### Add note

```bash
POST /api/metadata/track/{track_id}/note
{
    "text": "Person entered restricted zone",
    "author": "operator_1"
}
```

#### Add/remove tag

```bash
POST /api/metadata/track/{track_id}/tag
{
    "tag": "suspicious"
}

DELETE /api/metadata/track/{track_id}/tag
{
    "tag": "cleared"
}
```

#### Add alert

```bash
POST /api/metadata/track/{track_id}/alert
{
    "alert_type": "weapon_detected",
    "message": "Rifle detected at gate entrance",
    "severity": "critical"  # info, warning, critical
}
```

#### Set attribute

```bash
POST /api/metadata/track/{track_id}/attribute
{
    "key": "vehicle_color",
    "value": "blue"
}
```

#### Set behavior

```bash
POST /api/metadata/track/{track_id}/behavior
{
    "behavior": "loitering",
    "confidence": 0.85
}
```

#### Add zone

```bash
POST /api/metadata/track/{track_id}/zone
{
    "zone_name": "restricted_area_1"
}
```

### Management

#### Clean up expired tracks

```bash
POST /api/metadata/cleanup
```

#### Save to disk

```bash
POST /api/metadata/save
```

#### Clear all (use with caution)

```bash
DELETE /api/metadata/clear
```

## Programmatic Usage (Python)

### In your pipeline code

```python
from services.tracker.metadata_manager import get_metadata_manager

# Get tracks from tracker
tracks = tracker.update(boxes, class_ids, confidences, features)

# Add metadata to specific tracks
for track in tracks:
    if track.class_id == 0:  # person
        # Add note
        track.add_note("Person detected in frame", author="system")

        # Add tag
        track.add_tag("person")

        # Set attribute
        track.set_attribute("color", "blue_shirt")

        # Add alert if suspicious
        if is_suspicious(track):
            track.add_alert(
                "suspicious_behavior",
                "Person loitering for 5+ minutes",
                severity="warning"
            )

        # Set behavior
        if is_running(track):
            track.set_behavior("running", confidence=0.92)

        # Add zone
        if in_zone(track.box, "zone_a"):
            track.add_zone("zone_a")

# Metadata is automatically synced to MetadataManager by the tracker
```

### Weapon detection integration

```python
# When weapon is detected
if weapon_detected:
    track.add_alert(
        "weapon_detected",
        f"Pistol detected with {confidence:.2f} confidence",
        severity="critical"
    )
    track.add_tag("armed")
    track.set_attribute("weapon_type", "pistol")
    track.set_attribute("weapon_confidence", float(confidence))
```

### Cross-camera tracking

```python
# When track is re-identified from another camera
if reid_match:
    track.add_note(
        f"Re-identified from camera {previous_camera}",
        author="reid_system"
    )
    track.set_attribute("previous_camera", previous_camera)
    track.set_attribute("cameras_seen", cameras_seen_count)
```

## Persistence

### Automatic saving

The MetadataManager automatically:

-   Keeps metadata in memory for active tracks
-   Maintains metadata for expired tracks (configurable TTL, default 1 hour)
-   Syncs to disk on shutdown

### Manual persistence

```python
from services.tracker.metadata_manager import get_metadata_manager

manager = get_metadata_manager()

# Save to default location (output/track_metadata.json)
manager.save_metadata()

# Save to custom location
manager.save_metadata("backups/metadata_20240115.json")

# Clean up old metadata (older than TTL)
deleted_count = manager.cleanup_expired()
```

### Persistence format

Metadata is saved as JSON:

```json
{
    "metadata": {
        "42": { ... },
        "43": { ... }
    },
    "last_seen": {
        "42": 1234567890.0,
        "43": 1234567891.0
    },
    "class_index": {
        "0": [42, 43],
        "2": [44]
    },
    "saved_at": 1234567900.0
}
```

## Configuration

### TTL (Time-to-Live)

Default: 1 hour (3600 seconds)

```python
from services.tracker.metadata_manager import MetadataManager

# Custom TTL: 30 minutes
manager = MetadataManager(ttl=1800)

# Or modify global instance
from services.tracker.metadata_manager import get_metadata_manager
manager = get_metadata_manager()
manager.ttl = 1800  # 30 minutes
```

### Persistence path

Default: `output/track_metadata.json`

```python
manager = MetadataManager(persistence_path="custom/path/metadata.json")
```

## Best Practices

### 1. Use meaningful tags

```python
# Good
track.add_tag("suspicious")
track.add_tag("high_priority")
track.add_tag("armed")

# Avoid
track.add_tag("tag1")
track.add_tag("x")
```

### 2. Use attributes for structured data

```python
# Good - searchable and structured
track.set_attribute("vehicle_color", "blue")
track.set_attribute("license_plate", "ABC-1234")

# Avoid - hard to search
track.add_note("Blue vehicle with plate ABC-1234")
```

### 3. Use alerts for important events

```python
# Good - queryable by type and severity
track.add_alert(
    "zone_violation",
    "Entered restricted area 3",
    severity="critical"
)

# Less useful - just a note
track.add_note("Entered restricted area 3")
```

### 4. Add context to notes

```python
# Good - includes who and when (automatic timestamp)
track.add_note(
    "Person carrying large backpack, entered from north entrance",
    author="operator_smith"
)

# Less useful - vague
track.add_note("Something interesting")
```

### 5. Use behavior detection

```python
# Good - structured and searchable
if loitering_detected:
    track.set_behavior("loitering", confidence=0.85)

# Can then search: GET /api/metadata/search?behavior=loitering
```

### 6. Clean up regularly

```python
# Run cleanup periodically (e.g., every hour)
manager.cleanup_expired()
manager.save_metadata()
```

## Integration Examples

### Weapon detection system

```python
# In detection loop
for detection in weapon_detections:
    track_id = associate_weapon_to_track(detection)
    track = get_track_by_id(track_id)

    track.add_alert(
        "weapon_detected",
        f"{detection.weapon_class} detected",
        severity="critical"
    )
    track.add_tag("armed")
    track.set_attribute("weapon_type", detection.weapon_class)
    track.set_attribute("weapon_first_seen", time.time())
```

### Zone monitoring

```python
# In tracking loop
for track in active_tracks:
    current_zone = get_zone_for_box(track.box)

    if current_zone:
        track.add_zone(current_zone)

        # Check for restricted zones
        if current_zone in RESTRICTED_ZONES:
            track.add_alert(
                "zone_violation",
                f"Entered restricted zone: {current_zone}",
                severity="warning"
            )
            track.add_tag("zone_violator")
```

### Behavioral analysis

```python
# In behavior detection module
behavior_type, confidence = detect_behavior(track)

if behavior_type and confidence > 0.7:
    track.set_behavior(behavior_type, confidence)

    if behavior_type in ["loitering", "running", "fighting"]:
        track.add_alert(
            "suspicious_behavior",
            f"{behavior_type} detected",
            severity="warning"
        )
```

## Performance Considerations

### Memory usage

-   Metadata is kept in memory until cleanup
-   Default TTL: 1 hour (configurable)
-   Typical memory per track: ~1-5 KB
-   1000 tracks ≈ 1-5 MB

### Disk usage

-   JSON format (human-readable)
-   Typical file size: ~1-2 MB per 1000 tracks
-   Saved on shutdown or manual save

### API performance

-   Metadata queries are O(1) for track ID lookups
-   O(n) for class queries and searches (in-memory, fast)
-   No database required

### Recommendations

-   Run cleanup every hour: `POST /api/metadata/cleanup`
-   Save to disk periodically: `POST /api/metadata/save`
-   Use class index for efficient class queries
-   Use search endpoints with filters for targeted queries

## Troubleshooting

### Metadata not showing up

```python
# Ensure track is confirmed (not tentative)
tracks = tracker.update(...)
for track in tracks:  # Only confirmed tracks returned
    track.add_note("This will show up")
```

### Metadata lost after restart

```python
# Save before shutdown
from services.tracker.metadata_manager import get_metadata_manager
manager = get_metadata_manager()
manager.save_metadata()
```

### Can't find track in API

```python
# Check if track has been cleaned up
stats = manager.get_stats()
print(f"Total tracks: {stats['total_tracks']}")

# Check last seen time
metadata = manager.get_track_metadata(track_id)
if metadata:
    print(f"Last seen: {metadata['updated_at']}")
```

### Memory growing too large

```python
# Reduce TTL
manager.ttl = 1800  # 30 minutes instead of 1 hour

# Clean up more frequently
manager.cleanup_expired()
```

## Future Enhancements

Potential future additions:

-   Database backend (PostgreSQL, MongoDB)
-   Metadata export to CSV/Excel
-   Real-time metadata webhooks
-   Metadata templates for common scenarios
-   Bulk metadata operations
-   Metadata versioning/history
-   Advanced search (regex, fuzzy matching)
-   Metadata-based alerting rules
