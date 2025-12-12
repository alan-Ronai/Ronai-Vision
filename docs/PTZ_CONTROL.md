# PTZ Camera Control Guide

This guide covers PTZ (Pan-Tilt-Zoom) camera control via ONVIF protocol.

## Setup

### 1. Install Dependencies

```bash
pip install onvif-zeep
```

### 2. Camera Configuration

Add PTZ camera details to `config/camera_settings.json`:

```json
{
    "cameras": {
        "cam1": {
            "type": "rtsp",
            "source": "rtsp://admin:password@192.168.1.100:554/stream",
            "description": "PTZ Camera 1",
            "ptz": {
                "host": "192.168.1.100",
                "port": 80,
                "username": "admin",
                "password": "password"
            }
        }
    }
}
```

### 3. Start API Server

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### Connect to PTZ Camera

```bash
POST /api/ptz/connect
{
    "camera_id": "cam1",
    "host": "192.168.1.100",
    "port": 80,
    "username": "admin",
    "password": "password"
}
```

### Get PTZ Status

```bash
GET /api/ptz/status?camera_id=cam1
```

Response:
```json
{
    "ok": true,
    "position": {
        "pan": 0.0,
        "tilt": 0.0,
        "zoom": 0.5
    },
    "move_status": "IDLE"
}
```

### Continuous Movement

Move with velocity (-1.0 to 1.0):

```bash
POST /api/ptz/continuous_move
{
    "camera_id": "cam1",
    "pan_velocity": 0.5,    # Positive = right, negative = left
    "tilt_velocity": 0.3,   # Positive = up, negative = down
    "zoom_velocity": 0.2,   # Positive = zoom in, negative = zoom out
    "timeout": 3.0          # Duration in seconds (optional)
}
```

### Stop Movement

```bash
POST /api/ptz/stop?camera_id=cam1
```

### Absolute Positioning

Move to specific position (-1.0 to 1.0):

```bash
POST /api/ptz/absolute_move
{
    "camera_id": "cam1",
    "pan": 0.0,      # Center position
    "tilt": 0.0,     # Center position
    "zoom": 0.5,     # Mid zoom
    "pan_speed": 0.8,
    "tilt_speed": 0.8,
    "zoom_speed": 0.8
}
```

### Relative Movement

Move relative to current position:

```bash
POST /api/ptz/relative_move
{
    "camera_id": "cam1",
    "pan_delta": 0.1,    # Move right 10%
    "tilt_delta": -0.05, # Move down 5%
    "zoom_delta": 0.2    # Zoom in 20%
}
```

### Home Position

```bash
POST /api/ptz/goto_home?camera_id=cam1&speed=0.8
```

### Presets

**Get all presets:**
```bash
GET /api/ptz/presets?camera_id=cam1
```

**Save current position as preset:**
```bash
POST /api/ptz/set_preset
{
    "camera_id": "cam1",
    "preset_name": "entrance_view"
}
```

**Go to preset:**
```bash
POST /api/ptz/goto_preset
{
    "camera_id": "cam1",
    "preset_token": "1",
    "speed": 0.8
}
```

### List Connected Cameras

```bash
GET /api/ptz/connected
```

## Testing

Run the interactive test script:

```bash
python scripts/test_ptz.py
```

This will:
1. Connect to your PTZ camera
2. Test all movement types (continuous, absolute, relative)
3. Test presets
4. Run a scan pattern
5. Return to home position

## Common ONVIF Ports

- **Port 80**: Most common (HTTP)
- **Port 8080**: Alternative HTTP
- **Port 8899**: Some cameras use this
- **Port 554**: RTSP port (for streaming, not ONVIF control)

## Troubleshooting

### Connection Failed

1. **Verify camera is reachable:**
   ```bash
   ping <camera_ip>
   ```

2. **Check ONVIF port:**
   - Try ports 80, 8080, or 8899
   - Some cameras have ONVIF disabled by default (check camera web UI)

3. **Verify credentials:**
   - ONVIF username/password may differ from RTSP credentials
   - Some cameras require creating a separate ONVIF user

4. **Check firewall:**
   - Ensure camera allows connections from your server IP

### Movement Not Working

1. **Check PTZ support:**
   ```bash
   GET /api/ptz/status?camera_id=cam1
   ```
   If this returns position data, PTZ is supported.

2. **Check camera limits:**
   - Some cameras have restricted pan/tilt ranges
   - Positions are normalized (-1.0 to 1.0), but actual ranges vary

3. **Check movement speed:**
   - Try slower velocities (0.1-0.3) for testing
   - Some cameras require minimum speed thresholds

### Preset Issues

1. **Check existing presets first:**
   ```bash
   GET /api/ptz/presets?camera_id=cam1
   ```

2. **Some cameras limit preset count** (e.g., 8-256 presets)

3. **Preset tokens vary by camera:**
   - Some use numbers: "1", "2", "3"
   - Others use UUIDs

## Camera-Specific Notes

### Your Current Camera (46.210.89.167)

- **RTSP URL:** `rtsp://admin:ytcom123456@46.210.89.167:554/Streaming/channels/1/`
- **ONVIF Host:** `46.210.89.167`
- **ONVIF Port:** Typically 80 (test with script)
- **Username:** `admin`
- **Password:** `ytcom123456`

**To test:**
```bash
python scripts/test_ptz.py
```

## Python API Usage

```python
from services.ptz.onvif_ptz import ONVIFPTZController

# Connect to camera
ptz = ONVIFPTZController(
    host="192.168.1.100",
    port=80,
    username="admin",
    password="password"
)

# Get status
status = ptz.get_status()
print(f"Pan: {status['position']['pan']}")

# Move camera
ptz.continuous_move(pan_velocity=0.5, tilt_velocity=0.0, zoom_velocity=0.0, timeout=2.0)

# Stop
ptz.stop()

# Go to position
ptz.absolute_move(pan=0.0, tilt=0.0, zoom=0.5)

# Save preset
ptz.set_preset("my_preset")

# Go to preset
ptz.goto_preset("1")
```

## Integration with Detection Pipeline

PTZ control can be integrated with the detection pipeline for:

1. **Auto-tracking:** Follow detected persons/vehicles
2. **Zone scanning:** Patrol predefined areas
3. **Event-triggered:** Move to specific positions on detections
4. **Multi-camera coordination:** Handoff tracking between cameras

Example auto-tracking (future feature):
```python
# In run_multi_camera.py
if track.class_id == "person" and track.confidence > 0.8:
    # Calculate pan/tilt to center person in frame
    pan_target = (track.box[0] + track.box[2]) / 2 / frame_width * 2 - 1
    tilt_target = (track.box[1] + track.box[3]) / 2 / frame_height * 2 - 1

    # Move camera
    ptz.absolute_move(pan=pan_target, tilt=tilt_target)
```

## Resources

- [ONVIF Specification](https://www.onvif.org/specs/stream/ONVIF-Streaming-Spec.pdf)
- [python-onvif-zeep Documentation](https://github.com/FalkTannhaeuser/python-onvif-zeep)
- [ONVIF Device Test Tool](https://www.onvif.org/test-tool/) - For testing camera ONVIF compliance
