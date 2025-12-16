# New Features Implementation Summary

This document describes the major features that have been implemented in the Ronai Vision system.

## Overview

Four major feature sets have been implemented:

1. **Event-Based Video Recording System**
2. **Gemini AI Analysis for Detailed Object Description**
3. **Text-to-Speech (TTS) System with RTP Streaming**
4. **Operational Event Logging System**

---

## 1. Event-Based Video Recording System

### Description
Automatic and manual video recording system with circular buffer for capturing events like armed person detection.

### Features
- **Circular Buffer**: Pre-event footage (default: 5 seconds / 150 frames at 30fps)
- **Automatic Triggers**: Starts recording when armed person detected
- **Track-Based Recording**: Records specific tracked objects
- **Manual Control**: API endpoints for start/stop recording
- **Auto-Stop**: Stops recording when trigger condition ends (e.g., armed person leaves frame)

### Files Created/Modified
- `services/recorder/event_recorder.py` - Core recording service
- `services/recorder/__init__.py` - Module exports
- `api/routes/recorder.py` - API endpoints
- `services/pipeline/worker_manager.py` - Integration with pipeline

### API Endpoints

#### Start Manual Recording
```http
POST /api/recorder/start
Content-Type: application/json

{
  "camera_id": "camera_1",
  "trigger_reason": "manual_inspection",
  "include_buffer": true
}
```

**Response:**
```json
{
  "session_id": "camera_1_1234567890_1",
  "camera_id": "camera_1",
  "output_path": "output/recordings/camera_1_20250101_120000_manual_inspection.mp4",
  "message": "Recording started for camera camera_1"
}
```

#### Stop Recording
```http
POST /api/recorder/stop/{session_id}
```

**Response:**
```json
{
  "session_id": "camera_1_1234567890_1",
  "output_path": "output/recordings/camera_1_20250101_120000.mp4",
  "message": "Recording stopped: ..."
}
```

#### Get Active Sessions
```http
GET /api/recorder/sessions?camera_id=camera_1
```

**Response:**
```json
[
  {
    "session_id": "camera_1_1234567890_1",
    "camera_id": "camera_1",
    "trigger_type": "auto",
    "trigger_reason": "armed_person_detected",
    "track_id": 42,
    "duration": 15.5,
    "frame_count": 465,
    "output_path": "output/recordings/..."
  }
]
```

### Configuration
Configure in code or via environment variables:

```python
from services.recorder import RecorderConfig, get_event_recorder

config = RecorderConfig(
    output_dir="output/recordings",
    buffer_size=150,  # frames
    max_duration=300,  # seconds
    fps=30.0,
    codec="mp4v",  # or 'avc1' for H.264
    resolution=(1920, 1080)
)

recorder = get_event_recorder(config)
```

### Automatic Recording Triggers
The system automatically starts recording when:
1. Armed person detected (weapon + person association)
2. Track tagged as "armed"

Recording stops when:
1. Track no longer present in frame
2. Track no longer tagged as "armed"
3. Max duration reached

---

## 2. Gemini AI Analysis System

### Description
Detailed object analysis using Google Gemini API for cars and persons, with Hebrew prompts and responses.

### Features
- **Car Analysis**: Model, license plate, color
- **Person Analysis**: Clothing, physical features, carried items
- **Hebrew Language**: All prompts and responses in Hebrew
- **Rate Limiting**: Max 2-3 analyses per track ID
- **Automatic Throttling**: 1 second minimum between requests
- **Metadata Integration**: Results stored in track metadata

### Files Created/Modified
- `services/gemini/analyzer.py` - Gemini API integration
- `services/gemini/__init__.py` - Module exports
- `services/tracker/metadata_manager.py` - Analysis tracking
- `services/pipeline/worker_manager.py` - Pipeline integration
- `requirements.txt` - Added google-generativeai

### Setup

#### 1. Install Dependencies
```bash
pip install google-generativeai
```

#### 2. Set API Key
```bash
export GEMINI_API_KEY="your-api-key-here"
```

Or in Python:
```python
from services.gemini import get_gemini_analyzer

analyzer = get_gemini_analyzer(api_key="your-api-key")
```

### Usage

#### Programmatic Usage
```python
from services.gemini import get_gemini_analyzer
import cv2

# Initialize analyzer
analyzer = get_gemini_analyzer()

# Analyze a car
frame = cv2.imread("car_image.jpg")
bbox = [100, 100, 400, 300]  # [x1, y1, x2, y2]
result = analyzer.analyze_car(frame, bbox)

print(result)
# {
#   "דגם": "טויוטה קורולה",
#   "מספר_רישוי": "12-345-67",
#   "צבע": "כסוף",
#   "timestamp": 1234567890.0,
#   "error": None
# }

# Analyze a person
result = analyzer.analyze_person(frame, bbox)
print(result)
# {
#   "לבוש": "חולצה כחולה וג'ינס",
#   "צבע_עור": "בהיר",
#   "צבע_שיער": "חום",
#   "מין_משוער": "זכר",
#   "גיל_משוער": "30-40",
#   "פריטים_בידיים": ["תיק גב"],
#   "תיאור_נוסף": "משקפיים, שעון יד",
#   "timestamp": 1234567890.0,
#   "error": None
# }
```

### Automatic Pipeline Integration
The Gemini analysis is automatically triggered for:
- New car detections (first 2 frames of each track)
- New person detections (first 2 frames of each track)

Results are automatically stored in track metadata and can be queried via the metadata API.

### Hebrew Prompts
The system uses carefully crafted Hebrew prompts for consistent responses:

**Car Prompt:**
```
תאר את הרכב בתמונה בפורמט JSON בדיוק כזה:
{
  "דגם": "יצרן ודגם הרכב",
  "מספר_רישוי": "מספר הרישוי אם נראה, אחרת null",
  "צבע": "צבע הרכב"
}
```

**Person Prompt:**
```
תאר את האדם בתמונה בפירוט רב בפורמט JSON בדיוק כזה:
{
  "לבוש": "תיאור מפורט של הבגדים - סוג, צבע, סגנון",
  "צבע_עור": "גוון העור",
  "צבע_שיער": "צבע השיער",
  "מין_משוער": "זכר/נקבה/לא ברור",
  "גיל_משוער": "טווח גיל משוער",
  "פריטים_בידיים": "רשימת פריטים שהאדם מחזיק, אחרת null",
  "תיאור_נוסף": "כל מידע נוסף רלוונטי"
}
```

### Cost Optimization
- **Rate Limiting**: Maximum 2 analyses per track ID
- **Class Filtering**: Only analyzes cars and persons
- **Automatic Throttling**: 1 second minimum between API calls
- **Error Handling**: Graceful degradation if API fails

---

## 3. Text-to-Speech (TTS) System

### Description
Hebrew text-to-speech system with file export and RTP streaming capabilities.

### Features
- **Hebrew TTS**: Uses pyttsx3 with espeak backend
- **File Export**: Save audio to WAV files
- **RTP Streaming**: Stream synthesized audio to active RTP sessions
- **Configurable**: Sample rate, voice, speed
- **API Endpoints**: REST API for synthesis

### Files Modified
- `api/routes/audio.py` - Added TTS endpoints
- `services/audio/tts.py` - Already existed, enhanced

### API Endpoints

#### Synthesize Text to Speech
```http
POST /api/audio/tts/synthesize
Content-Type: application/json

{
  "text": "שלום, זוהה אדם חמוש במצלמה 1",
  "save_file": true
}
```

**Response:**
```json
{
  "success": true,
  "message": "Speech synthesized successfully",
  "audio_file": "output/tts/tts_1234567890.wav",
  "sample_rate": 16000,
  "duration": 2.5
}
```

#### Synthesize and Stream to RTP
```http
POST /api/audio/tts/speak-and-stream?text=שלום&session_id=session_123
```

**Response:**
```json
{
  "success": true,
  "message": "Speech synthesized and ready to stream to session session_123",
  "audio_size_bytes": 32000,
  "sample_rate": 8000,
  "note": "Full RTP streaming implementation requires additional packet handling"
}
```

#### Get TTS Status
```http
GET /api/audio/tts/status
```

**Response:**
```json
{
  "initialized": true,
  "engine": "pyttsx3",
  "sample_rate": 16000
}
```

### Programmatic Usage
```python
from services.audio.tts import HebrewTTS

# Initialize TTS
tts = HebrewTTS(sample_rate=16000, engine="pyttsx3")

# Synthesize to numpy array
audio = tts.synthesize("שלום עולם")
print(f"Generated {len(audio)} samples at {tts.sample_rate}Hz")

# Synthesize to file
success = tts.synthesize_to_file("שלום עולם", "output/hello.wav")

# Generate RTP payload
payload, sample_rate = tts.text_to_rtp_payload("שלום עולם", codec="g711_ulaw")
print(f"RTP payload: {len(payload)} bytes at {sample_rate}Hz")
```

### Installation
For Hebrew support, install espeak:

**Linux:**
```bash
sudo apt-get install espeak espeak-data
```

**macOS:**
```bash
brew install espeak
```

**Python dependencies** (already in requirements.txt):
```bash
pip install pyttsx3 librosa soundfile
```

---

## 4. Operational Event Logging System

### Description
High-level operational event logging for important system events, separate from technical/debug logs.

### Features
- **Event Types**: Detection, Alert, Analysis, Recording, System
- **Severity Levels**: Info, Warning, Critical
- **Hebrew Support**: UTF-8 encoding for Hebrew messages
- **JSON Lines Format**: Structured logging for easy parsing
- **Console Output**: Color-coded console display
- **Automatic Integration**: Auto-logs important pipeline events

### Files Created/Modified
- `services/logging/operational_logger.py` - Logger implementation
- `services/logging/__init__.py` - Module exports
- `services/pipeline/worker_manager.py` - Auto-logging integration

### Event Types

| Event Type | Description | Examples |
|------------|-------------|----------|
| `detection` | New object detected | New person/car in frame |
| `alert` | Critical alert | Armed person detected |
| `analysis` | AI analysis completed | Gemini analysis result |
| `recording` | Recording started/stopped | Video recording events |
| `system` | System events | Startup, shutdown, errors |

### Log Format
Logs are stored in JSONL (JSON Lines) format at `output/logs/operational.jsonl`:

```json
{"timestamp": 1234567890.0, "event_type": "detection", "severity": "info", "camera_id": "camera_1", "track_id": 42, "message": "אובייקט חדש זוהה במצלמה camera_1: person (ID: 42)", "details": {}}
{"timestamp": 1234567891.0, "event_type": "alert", "severity": "critical", "camera_id": "camera_1", "track_id": 42, "message": "התראה במצלמה camera_1: אדם חמוש זוהה עם pistol (ID: 42)", "details": {"alert_type": "armed_person", "weapons": ["pistol"]}}
{"timestamp": 1234567892.0, "event_type": "recording", "severity": "info", "camera_id": "camera_1", "track_id": 42, "message": "הקלטה החלה במצלמה camera_1, ID: 42 (סיבה: armed_person_detected)", "details": {"action": "started", "trigger_reason": "armed_person_detected"}}
```

### Programmatic Usage
```python
from services.logging import get_operational_logger, EventSeverity

# Get logger instance
logger = get_operational_logger()

# Log detection
logger.log_detection(
    camera_id="camera_1",
    track_id=42,
    class_name="person",
    details={"gemini_analysis": {...}}
)

# Log alert
logger.log_alert(
    camera_id="camera_1",
    track_id=42,
    alert_type="armed_person",
    alert_message="אדם חמוש זוהה עם רובה",
    details={"weapons": ["rifle"]}
)

# Log analysis
logger.log_analysis(
    camera_id="camera_1",
    track_id=42,
    class_name="person",
    analysis_result={
        "לבוש": "חולצה שחורה",
        "צבע_שיער": "חום",
        ...
    }
)

# Log recording
logger.log_recording(
    camera_id="camera_1",
    action="started",
    trigger_reason="armed_person_detected",
    track_id=42
)

# Log system event
logger.log_system_event(
    message="מערכת הופעלה בהצלחה",
    severity=EventSeverity.INFO
)
```

### Automatic Logging
The system automatically logs:
- **New Detections**: When a new car or person is first detected
- **Armed Person Alerts**: When weapon + person association occurs
- **Gemini Analysis**: When AI analysis completes successfully
- **Recording Events**: When recording starts/stops

### Console Output
Events are displayed in color-coded format:

```
[INFO] [2025-01-15 12:00:00] [DETECTION] [camera_1] [Track:42] אובייקט חדש זוהה במצלמה camera_1: person (ID: 42)
[CRITICAL] [2025-01-15 12:00:01] [ALERT] [camera_1] [Track:42] התראה במצלמה camera_1: אדם חמוש זוהה עם pistol (ID: 42)
[INFO] [2025-01-15 12:00:02] [RECORDING] [camera_1] [Track:42] הקלטה החלה במצלמה camera_1, ID: 42 (סיבה: armed_person_detected)
```

Colors:
- **INFO**: Default (white)
- **WARNING**: Yellow
- **CRITICAL**: Red

---

## Integration Summary

### Pipeline Integration Flow

```
Frame Input
    ↓
Detection (YOLO)
    ↓
Tracking (BoTSort)
    ↓
┌─────────────────────────────────────────────┐
│ NEW INTEGRATIONS                            │
├─────────────────────────────────────────────┤
│ 1. Add frame to recorder buffer             │
│ 2. Log new detections (operational logger)  │
│ 3. Perform Gemini analysis (if eligible)    │
│    - Check rate limit (max 2 per ID)        │
│    - Analyze car/person                     │
│    - Store in metadata                      │
│    - Log analysis result                    │
│ 4. Check for armed person                   │
│    - Start recording if armed               │
│    - Log alert                              │
│    - Log recording started                  │
│ 5. Check recording stop conditions          │
│    - Stop if armed person left              │
│    - Log recording stopped                  │
└─────────────────────────────────────────────┘
    ↓
Render & Publish Output
```

### Data Flow

```
Track Metadata
    ├── gemini_car_analysis (if car)
    │   ├── דגם
    │   ├── מספר_רישוי
    │   ├── צבע
    │   └── timestamp
    ├── gemini_person_analysis (if person)
    │   ├── לבוש
    │   ├── צבע_עור
    │   ├── צבע_שיער
    │   ├── מין_משוער
    │   ├── גיל_משוער
    │   ├── פריטים_בידיים
    │   └── תיאור_נוסף
    ├── tags (armed, etc.)
    ├── alerts (armed_person, etc.)
    └── attributes (weapons_detected, etc.)
```

---

## Configuration

### Environment Variables

```bash
# Gemini API
export GEMINI_API_KEY="your-gemini-api-key"

# Recording Configuration
export RECORDER_OUTPUT_DIR="output/recordings"
export RECORDER_BUFFER_SIZE="150"
export RECORDER_MAX_DURATION="300"

# Operational Logging
export OP_LOG_DIR="output/logs"
export OP_LOG_FILE="operational.jsonl"

# TTS Configuration
export TTS_SAMPLE_RATE="16000"
export TTS_ENGINE="pyttsx3"
```

### Python Configuration

```python
# Recorder
from services.recorder import RecorderConfig, get_event_recorder
config = RecorderConfig(
    output_dir="output/recordings",
    buffer_size=150,
    max_duration=300,
    fps=30.0,
    codec="mp4v",
    resolution=(1920, 1080)
)
recorder = get_event_recorder(config)

# Gemini
from services.gemini import get_gemini_analyzer
analyzer = get_gemini_analyzer(
    api_key="your-key",
    model_name="gemini-1.5-flash"
)

# Operational Logger
from services.logging import get_operational_logger
logger = get_operational_logger(
    log_dir="output/logs",
    log_file="operational.jsonl",
    console_output=True
)

# TTS
from services.audio.tts import HebrewTTS
tts = HebrewTTS(sample_rate=16000, engine="pyttsx3")
```

---

## Testing

### Test Video Recording
```bash
# Start recording manually
curl -X POST http://localhost:8000/api/recorder/start \
  -H "Content-Type: application/json" \
  -d '{"camera_id": "camera_1", "include_buffer": true}'

# Wait a few seconds...

# Stop recording
curl -X POST http://localhost:8000/api/recorder/stop/SESSION_ID

# Check recordings
ls output/recordings/
```

### Test Gemini Analysis
```python
from services.gemini import get_gemini_analyzer
import cv2

analyzer = get_gemini_analyzer(api_key="your-key")
frame = cv2.imread("test_car.jpg")
result = analyzer.analyze_car(frame)
print(result)
```

### Test TTS
```bash
# Synthesize Hebrew text
curl -X POST http://localhost:8000/api/audio/tts/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "שלום עולם", "save_file": true}'

# Check output
ls output/tts/
```

### Test Operational Logging
```python
from services.logging import get_operational_logger

logger = get_operational_logger()
logger.log_detection("camera_1", 42, "person")
logger.log_alert("camera_1", 42, "armed_person", "אדם חמוש זוהה")

# Check logs
cat output/logs/operational.jsonl
```

---

## Performance Considerations

### Video Recording
- **Buffer Size**: Larger buffer = more memory, more pre-event footage
- **Resolution**: Higher resolution = larger files, more disk I/O
- **Codec**: H.264 (avc1) recommended for better compression

### Gemini Analysis
- **Rate Limiting**: Max 2 analyses per track to control API costs
- **API Throttling**: 1 second minimum between requests
- **Asynchronous**: Consider thread pool for non-blocking analysis

### TTS
- **Lazy Loading**: Engine initialized on first use
- **Caching**: Consider caching frequently used phrases
- **Sample Rate**: 16kHz recommended for balance of quality/performance

### Operational Logging
- **JSONL Format**: Efficient append-only writes
- **Log Rotation**: Consider implementing log rotation for long-running systems
- **Filtering**: Can disable console output for production

---

## Troubleshooting

### Video Recording Issues
```
Problem: Recording not starting
- Check camera is active and producing frames
- Check output directory is writable
- Check disk space available

Problem: Video file corrupted
- Ensure recording was properly stopped
- Check for system crashes during recording
- Verify OpenCV VideoWriter codec support
```

### Gemini API Issues
```
Problem: "API key not found"
- Set GEMINI_API_KEY environment variable
- Or pass api_key parameter to get_gemini_analyzer()

Problem: "Rate limit exceeded"
- Gemini has rate limits per minute/day
- System already limits to 2 analyses per track
- Consider increasing delay between requests

Problem: "Invalid JSON response"
- Gemini occasionally returns non-JSON
- System logs error and continues
- Check analysis_result["error"] field
```

### TTS Issues
```
Problem: "espeak not found"
- Install espeak: sudo apt-get install espeak (Linux)
- Or: brew install espeak (macOS)

Problem: "No Hebrew voice found"
- Install Hebrew espeak data
- Or use default voice (still works with Hebrew text)

Problem: "Audio file not created"
- Check output/tts directory exists and is writable
- Check soundfile library installed: pip install soundfile
```

### Operational Logging Issues
```
Problem: "Log file not created"
- Check output/logs directory is writable
- Check UTF-8 encoding support for Hebrew text

Problem: "Console output garbled"
- Ensure terminal supports UTF-8
- Set: export LANG=en_US.UTF-8

Problem: "Logs too large"
- Implement log rotation
- Or periodically archive/compress old logs
```

---

## Future Enhancements

### Video Recording
- [ ] H.264 hardware encoding support
- [ ] Multi-track recording (multiple objects in one video)
- [ ] Cloud storage integration (S3, Azure Blob)
- [ ] Video trimming/editing API

### Gemini Analysis
- [ ] Async analysis with thread pool
- [ ] Response caching for similar images
- [ ] Multiple angles analysis aggregation
- [ ] Custom prompt templates via config

### TTS
- [ ] Neural TTS models (better quality)
- [ ] Real-time RTP packet streaming
- [ ] Voice cloning for custom voices
- [ ] SSML support for prosody control

### Operational Logging
- [ ] Real-time log streaming API (WebSocket)
- [ ] Log aggregation and search
- [ ] Alerting and notifications
- [ ] Dashboard/UI for log visualization

---

## API Quick Reference

### Video Recording
```
POST   /api/recorder/start          - Start recording
POST   /api/recorder/stop/{id}      - Stop recording
GET    /api/recorder/sessions       - List active sessions
GET    /api/recorder/sessions/{id}  - Get session info
DELETE /api/recorder/cleanup        - Cleanup old sessions
```

### Text-to-Speech
```
POST   /api/audio/tts/synthesize         - Synthesize text to audio
POST   /api/audio/tts/speak-and-stream   - Synthesize and stream to RTP
GET    /api/audio/tts/status             - Get TTS engine status
```

### Metadata (includes Gemini analysis)
```
GET    /api/metadata/tracks/{id}         - Get track metadata (includes Gemini results)
GET    /api/metadata/search              - Search metadata by tags/alerts
GET    /api/metadata/stats               - Get metadata statistics
```

---

## License

Same as parent project (Ronai Vision).

## Support

For questions or issues, please refer to the main project documentation or contact the development team.
