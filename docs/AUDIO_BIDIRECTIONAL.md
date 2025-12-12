# Bidirectional Hebrew Audio System

Complete audio processing system for field device communication with Hebrew transcription, command processing, and text-to-speech responses.

## Overview

The system provides:

-   **Audio Reception**: Receives RTP audio streams from field communication devices
-   **Hebrew Transcription**: Transcribes Hebrew speech using Whisper Large v3 Hebrew model (offline)
-   **Command Processing**: Detects Hebrew keywords and executes corresponding actions
-   **Text-to-Speech**: Converts Hebrew text to speech (offline with pyttsx3)
-   **Audio Transmission**: Sends audio responses back to field devices via RTP

## Architecture

```
Field Device → RTP Audio → Transcription → Command Detection → Action
                                                              ↓
Field Device ← RTP Audio ← TTS Synthesis ← Response Text ←────┘
```

### Components

1. **RTP Server** (`services/audio/rtp_server.py`): Bidirectional RTP/RTSP server
2. **Transcriber** (`services/audio/transcriber.py`): Hebrew Whisper transcription
3. **Command Processor** (`services/audio/command_processor.py`): Keyword detection and routing
4. **TTS** (`services/audio/tts.py`): Hebrew text-to-speech synthesis
5. **Pipeline** (`services/audio/audio_pipeline.py`): Orchestrates entire flow
6. **API** (`api/routes/audio.py`): HTTP endpoints for control

## Quick Start

### 1. Install Dependencies

```bash
# Install espeak for TTS (required)
# macOS:
brew install espeak

# Linux:
sudo apt-get install espeak espeak-data

# Install Python packages
pip install transformers pyttsx3 librosa soundfile
```

### 2. Start the System

```bash
# Start FastAPI server
uvicorn api.server:app --host 0.0.0.0 --port 8000

# In another terminal or via API:
curl -X POST http://localhost:8000/api/audio/start
curl -X POST http://localhost:8000/api/audio/transcription/start
```

### 3. Connect Field Device

Configure your field device to send RTP audio to:

-   **RTSP URL**: `rtsp://SERVER_IP:8554/audio`
-   **RTP Port**: `5004` (UDP)
-   **Codec**: G.711 μ-law (preferred) or G.711 A-law

## API Endpoints

### Audio Server Control

#### Start Audio Server

```bash
POST /api/audio/start
```

#### Stop Audio Server

```bash
POST /api/audio/stop
```

#### Get Server Status

```bash
GET /api/audio/status
```

Response:

```json
{
    "running": true,
    "rtsp_host": "0.0.0.0",
    "rtsp_port": 8554,
    "rtp_base_port": 5004,
    "active_sessions": 1,
    "active_receivers": 1,
    "active_senders": 1
}
```

### Transcription Control

#### Start Transcription Pipeline

```bash
POST /api/audio/transcription/start
```

#### Stop Transcription Pipeline

```bash
POST /api/audio/transcription/stop
```

#### Get Pipeline Status

```bash
GET /api/audio/transcription/status
```

Response:

```json
{
    "running": true,
    "chunks_processed": 150,
    "transcriptions": 45,
    "commands_detected": 12,
    "responses_sent": 12,
    "queue_size": 2,
    "transcriber_ready": true,
    "tts_ready": true
}
```

### Command History

#### Get Recent Commands

```bash
GET /api/audio/commands/history?limit=10
```

Response:

```json
[
    {
        "command_id": "track_person",
        "keyword": "עקוב",
        "text": "עקוב אחרי אדם",
        "confidence": 1.0,
        "timestamp": "2025-12-11T10:30:45.123456",
        "parameters": {
            "session_id": "abc123"
        }
    }
]
```

#### Clear Command History

```bash
DELETE /api/audio/commands/history
```

### Send Audio Responses

#### Send Text as Speech

```bash
POST /api/audio/speak
Content-Type: application/json

{
  "session_id": "abc123",
  "text": "המערכת פועלת כרגיל"
}
```

#### Send Pre-recorded Audio File

```bash
POST /api/audio/send-audio/{session_id}
Content-Type: multipart/form-data

audio_file: <WAV file>
```

#### List Active Sessions

```bash
GET /api/audio/sessions
```

Response:

```json
["session_abc123", "session_def456"]
```

## Supported Hebrew Commands

The system recognizes the following Hebrew commands:

| Command ID      | Keywords                         | Action             |
| --------------- | -------------------------------- | ------------------ |
| `track_person`  | עקוב, עקוב אחרי, תעקוב, מעקב     | Track person       |
| `track_car`     | עקוב אחרי רכב, תעקוב אחרי מכונית | Track car          |
| `zoom_in`       | זום, זום פנימה, תקרב, הקרב       | Zoom in camera     |
| `zoom_out`      | זום החוצה, תרחק, הרחק            | Zoom out camera    |
| `pan_left`      | שמאלה, פנה שמאלה, תזוז שמאלה     | Pan camera left    |
| `pan_right`     | ימינה, פנה ימינה, תזוז ימינה     | Pan camera right   |
| `status_report` | סטטוס, דוח, מה הסטטוס, דווח      | Get status report  |
| `list_tracks`   | רשימת מעקבים, כמה עוקבים         | List active tracks |
| `stop`          | עצור, תפסיק, סטופ, הפסק          | Stop operation     |
| `start`         | התחל, תתחיל, סטארט               | Start operation    |

### Default Hebrew Responses

| Command       | Response               |
| ------------- | ---------------------- |
| Track person  | מתחיל מעקב אחרי אדם    |
| Track car     | מתחיל מעקב אחרי רכב    |
| Zoom in       | מקרב תמונה             |
| Zoom out      | מרחיק תמונה            |
| Pan left      | זז שמאלה               |
| Pan right     | זז ימינה               |
| Status report | המערכת פועלת כרגיל     |
| List tracks   | אין מעקבים פעילים כרגע |
| Stop          | עוצר                   |
| Start         | מתחיל                  |

## Adding Custom Commands

### 1. Register Command in Code

```python
from services.audio import AudioPipeline, CommandProcessor

# Get pipeline
pipeline = get_audio_pipeline()

# Register custom command
def handle_my_command(match):
    print(f"Custom command: {match.text}")
    # TODO: Your implementation here

pipeline.command_processor.register_command(
    command_id="my_custom_command",
    keywords=["מילת מפתח", "עוד מילה"],  # Hebrew keywords
    handler=handle_my_command
)
```

### 2. Add Custom Response

Modify `audio_pipeline.py` method `_generate_response()` to add your response:

```python
responses = {
    # ... existing responses ...
    "my_custom_command": "תגובה בעברית",  # Hebrew response
}
```

## Performance Considerations

### Transcription Speed

-   **Whisper Large v3**: ~1-2 seconds per 30 seconds of audio on CPU
-   **CPU Usage**: High during transcription (1 core at 100%)
-   **Memory**: ~2-3GB for model in RAM

**Optimization Tips**:

-   Use GPU (CUDA/MPS) for 5-10x faster transcription
-   Consider Whisper Medium for faster inference with slightly lower accuracy
-   Batch audio chunks (3-5 seconds) for better accuracy vs. real-time latency

### TTS Speed

-   **pyttsx3**: Near real-time (~1x speed)
-   **Quality**: Depends on espeak voice quality
-   **Offline**: No internet required

### Network

-   **Bandwidth**: ~64 kbps for G.711 audio (8kHz, 8-bit)
-   **Latency**: ~100-200ms for audio + ~1-2s for transcription
-   **Packet Loss**: Jitter buffer handles up to 10% loss

## Troubleshooting

### Whisper Model Not Found

```
Error: Unable to load whisper-large-v3-hebrew
```

**Solution**: Verify model exists in `models/whisper-large-v3-hebrew/` directory with all required files (config.json, model files, tokenizer).

### TTS Not Working

```
Error: Failed to load TTS engine
```

**Solution**: Install espeak:

-   macOS: `brew install espeak`
-   Linux: `sudo apt-get install espeak`

### No Hebrew Voice

```
Warning: No Hebrew voice found, using default
```

**Solution**: Install Hebrew voice pack for espeak or configure system TTS with Hebrew support.

### Audio Not Sending

```
Error: Failed to send RTP packet
```

**Solution**:

-   Check session exists: `GET /api/audio/sessions`
-   Verify field device is listening on correct port
-   Check firewall allows UDP port 5004

## Session Management

Each field device connection creates a unique session:

-   **Session ID**: Generated on RTSP SETUP
-   **Lifecycle**: Created on connection, destroyed on TEARDOWN or timeout
-   **Isolation**: Each session has independent audio buffers and state

### Session Timeout

Default: 60 seconds of inactivity

Configure in `config/audio_settings.json`:

```json
{
    "rtsp": {
        "session_timeout": 60
    }
}
```

## Configuration Files

### Audio Settings (`config/audio_settings.json`)

```json
{
    "rtsp": {
        "host": "0.0.0.0",
        "port": 8554,
        "session_timeout": 60
    },
    "rtp": {
        "base_port": 5004,
        "jitter_buffer_ms": 100
    },
    "codecs": {
        "enabled": ["g711", "opus", "amr"]
    },
    "storage": {
        "base_path": "audio_storage/recordings"
    }
}
```

### Environment Variables

```bash
# Audio server
AUDIO_SERVER_ENABLED=true
AUDIO_CONFIG=config/audio_settings.json
AUDIO_STORAGE_PATH=audio_storage/recordings

# Device selection for Whisper
DEVICE=auto  # or 'cpu', 'cuda', 'mps'
```

## Future Enhancements

-   [ ] Support for multiple languages (currently Hebrew only)
-   [ ] Fuzzy keyword matching with confidence scores
-   [ ] Streaming transcription (per audio chunk vs. full buffer)
-   [ ] Custom wake word detection
-   [ ] Audio preprocessing (noise reduction, echo cancellation)
-   [ ] Integration with PTZ camera control
-   [ ] Integration with video tracking system
-   [ ] WebSocket API for real-time transcription events
-   [ ] Recording and playback of command sessions
-   [ ] Command syntax parser (structured commands with parameters)
