# Multi-Protocol Audio System Documentation

## Overview

This document describes the comprehensive bidirectional audio communication system with Hebrew transcription, command processing, and multi-protocol support that has been implemented in the Ronai-Vision project.

## System Architecture

### Components

The audio system consists of several interconnected modules:

1. **Audio Receivers** - Multiple protocol support for receiving audio from field devices
2. **Transcription Service** - Hebrew speech-to-text using Whisper Large v3
3. **Command Processor** - Hebrew keyword detection and action execution
4. **Text-to-Speech** - Hebrew voice synthesis for responses
5. **RTP Sender** - Audio transmission back to field devices
6. **Unified Manager** - Orchestrates all protocols and audio flow

## Features Implemented

### 1. Hebrew Audio Transcription

-   **Model**: Whisper Large v3 Hebrew (offline)
-   **Location**: `models/whisper-large-v3-hebrew/`
-   **Implementation**: `services/audio/transcriber.py`
-   **Capabilities**:
    -   Real-time transcription from audio streams
    -   File-based transcription
    -   Automatic resampling to 16kHz
    -   Beam search decoding for accuracy
    -   Language forced to Hebrew

### 2. Hebrew Command Processing

-   **Implementation**: `services/audio/command_processor.py`
-   **Default Commands** (10 built-in):
    1. **עקוב** (track) - Start tracking an object
    2. **הפסק עקיבה** (stop tracking) - Stop tracking
    3. **זום פנימה** (zoom in) - Zoom in camera
    4. **זום החוצה** (zoom out) - Zoom out camera
    5. **שמאלה** (left) - Pan camera left
    6. **ימינה** (right) - Pan camera right
    7. **מעלה** (up) - Tilt camera up
    8. **מטה** (down) - Tilt camera down
    9. **סטטוס** (status) - Get system status
    10. **עצור** (stop) - Stop all operations

### 3. Hebrew Text-to-Speech

-   **Implementation**: `services/audio/tts.py`
-   **Engine**: pyttsx3 with espeak backend
-   **Features**:
    -   Offline Hebrew voice synthesis
    -   Automatic voice detection
    -   RTP payload generation (G.711 μ-law)
    -   Configurable rate and volume

### 4. Multi-Protocol Audio Support

#### A. RTSP (Real-Time Streaming Protocol)

-   **Standard RTSP/RTP audio streaming**
-   Uses existing RTP server implementation
-   Supports DESCRIBE, SETUP, PLAY, TEARDOWN

#### B. Raw RTP/UDP Receiver

-   **Implementation**: `services/audio/raw_rtp_receiver.py`
-   **Features**:
    -   Direct UDP/RTP reception without RTSP signaling
    -   Auto-detection of codec from RTP payload type
    -   Jitter buffer for packet reordering
    -   Support for G.711 μ-law/A-law, Opus, AMR
-   **Use Case**: Simple field devices that send raw RTP

#### C. SIP + RTP (VoIP)

-   **Implementation**: `services/audio/sip_server.py`
-   **Features**:
    -   Basic SIP server (INVITE, ACK, BYE)
    -   SDP negotiation for RTP parameters
    -   Automatic RTP port allocation
    -   Session management
-   **Use Case**: VoIP phones, SIP-enabled field devices

#### D. FFmpeg Integration

-   **Implementation**: `services/audio/stream_integrations.py`
-   **Features**:
    -   Flexible audio input via FFmpeg
    -   Support for RTSP, HTTP, files, devices
    -   Multiple audio format support
    -   PCM output for processing
-   **Use Case**: Complex audio sources, format conversion

#### E. GStreamer Integration

-   **Implementation**: `services/audio/stream_integrations.py`
-   **Features**:
    -   Pipeline-based audio processing
    -   RTSP, UDP, file sources
    -   Advanced audio transformations
    -   Real-time streaming
-   **Use Case**: Advanced audio pipelines, low-latency streaming

#### F. Unified Audio Manager

-   **Implementation**: `services/audio/unified_receiver.py`
-   **Features**:
    -   Single interface for all protocols
    -   Dynamic protocol enable/disable
    -   Status monitoring
    -   Automatic resource cleanup

### 5. Bidirectional Communication

-   **Implementation**: `services/audio/audio_pipeline.py` + `services/audio/rtp_sender.py`
-   **Flow**:
    1. Receive audio from field device
    2. Transcribe to Hebrew text
    3. Process commands and detect keywords
    4. Generate Hebrew response
    5. Convert to speech (TTS)
    6. Send back via RTP

### 6. API Endpoints

-   **File**: `api/routes/audio.py`
-   **Endpoints** (20+ added):

#### Protocol Management

```
POST   /audio/protocols/rtsp/enable
POST   /audio/protocols/rtsp/disable
POST   /audio/protocols/sip/enable
POST   /audio/protocols/sip/disable
POST   /audio/protocols/raw-rtp/enable
POST   /audio/protocols/raw-rtp/disable
POST   /audio/protocols/ffmpeg/enable
POST   /audio/protocols/ffmpeg/disable
POST   /audio/protocols/gstreamer/enable
POST   /audio/protocols/gstreamer/disable
GET    /audio/protocols/status
```

#### Receiver Management

```
POST   /audio/receivers/raw-rtp/add
DELETE /audio/receivers/raw-rtp/remove/{receiver_id}
POST   /audio/receivers/ffmpeg/add
DELETE /audio/receivers/ffmpeg/remove/{receiver_id}
POST   /audio/receivers/gstreamer/add
DELETE /audio/receivers/gstreamer/remove/{receiver_id}
GET    /audio/receivers/list
```

#### Transcription & TTS

```
POST   /audio/transcription/start
POST   /audio/transcription/stop
POST   /audio/transcription/file
POST   /audio/speak
```

#### Command System

```
GET    /audio/commands/list
POST   /audio/commands/register
GET    /audio/commands/history
POST   /audio/commands/execute
```

## Installation

### Dependencies

1. **Python Packages** (already in requirements.txt):

```bash
pip install transformers pyttsx3 librosa soundfile
```

2. **System Dependencies**:

```bash
# macOS
brew install espeak ffmpeg gstreamer

# Ubuntu/Debian
apt-get install espeak ffmpeg gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good
```

3. **Models**:

-   Whisper Large v3 Hebrew model should be in `models/whisper-large-v3-hebrew/`
-   Model files are already present in the project

## Usage

### Starting the System

1. **Start the API Server**:

```bash
cd /Users/alankantor/Downloads/Ronai/Ronai-Vision
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

2. **Initialize the Audio Pipeline** (in your code):

```python
from services.audio.audio_pipeline import AudioPipeline

# Create pipeline
pipeline = AudioPipeline(
    transcriber_enabled=True,
    command_processor_enabled=True,
    tts_enabled=True
)

# Start processing
pipeline.start()
```

### Enabling Protocols

#### Via API:

**Enable Raw RTP Receiver**:

```bash
curl -X POST http://localhost:8000/audio/protocols/raw-rtp/enable \
  -H "Content-Type: application/json" \
  -d '{"port": 5004, "codec": "pcmu"}'
```

**Enable SIP Server**:

```bash
curl -X POST http://localhost:8000/audio/protocols/sip/enable \
  -H "Content-Type: application/json" \
  -d '{"port": 5060}'
```

**Enable FFmpeg Receiver**:

```bash
curl -X POST http://localhost:8000/audio/protocols/ffmpeg/enable \
  -H "Content-Type: application/json" \
  -d '{
    "source": "rtsp://camera.local/audio",
    "format": "s16le",
    "sample_rate": 16000
  }'
```

**Enable GStreamer Receiver**:

```bash
curl -X POST http://localhost:8000/audio/protocols/gstreamer/enable \
  -H "Content-Type: application/json" \
  -d '{
    "pipeline": "udpsrc port=5004 ! application/x-rtp ! rtppcmudepay ! audio/x-raw,rate=8000 ! audioconvert ! audio/x-raw,rate=16000,format=S16LE"
  }'
```

#### Via Python Code:

```python
from services.audio.unified_receiver import UnifiedAudioReceiver

# Create unified receiver
receiver = UnifiedAudioReceiver()

# Enable protocols
receiver.enable_raw_rtp(port=5004, codec="pcmu")
receiver.enable_sip(port=5060)
receiver.enable_ffmpeg(source="rtsp://camera.local/audio")
receiver.enable_gstreamer(pipeline="your-gstreamer-pipeline")

# Check status
status = receiver.get_status()
print(status)
```

### Using the Audio Pipeline

#### Basic Transcription:

```python
from services.audio.transcriber import HebrewTranscriber

transcriber = HebrewTranscriber()

# Transcribe file
text = transcriber.transcribe("path/to/audio.wav")
print(f"Transcribed: {text}")

# Transcribe audio buffer
audio_data = np.array([...])  # 16kHz audio
text = transcriber.transcribe(audio_data, sample_rate=16000)
```

#### Command Processing:

```python
from services.audio.command_processor import CommandProcessor

processor = CommandProcessor()

# Register custom command
def custom_action(args):
    print(f"Custom action with args: {args}")
    return "Action completed"

processor.register_command(
    keywords=["מיוחד", "פעולה"],  # Hebrew: special, action
    action=custom_action,
    description="Custom action"
)

# Process transcribed text
result = processor.process("עקוב אחרי המכונית")  # Track the car
if result:
    print(f"Command executed: {result}")
```

#### Text-to-Speech:

```python
from services.audio.tts import HebrewTTS

tts = HebrewTTS()

# Generate audio file
tts.synthesize("שלום, אני מערכת רונאי", output_file="greeting.wav")

# Generate RTP payload
rtp_payload = tts.text_to_rtp_payload("הפעולה הושלמה בהצלחה")
# Send via RTP sender
```

#### Bidirectional Communication:

```python
from services.audio.audio_pipeline import AudioPipeline

# Create pipeline
pipeline = AudioPipeline(
    transcriber_enabled=True,
    command_processor_enabled=True,
    tts_enabled=True,
    rtp_sender_config={
        'dest_ip': '192.168.1.100',
        'dest_port': 5004
    }
)

# Set up callback for transcriptions
def on_transcription(text):
    print(f"Heard: {text}")

pipeline.on_transcription = on_transcription

# Start
pipeline.start()

# Pipeline will now:
# 1. Receive audio
# 2. Transcribe to Hebrew text
# 3. Process commands
# 4. Generate responses
# 5. Send audio back via RTP
```

### Field Device Configuration

#### Raw RTP Device:

```
Device Settings:
- Codec: G.711 μ-law (PCMU)
- Sample Rate: 8000 Hz
- Destination IP: <server-ip>
- Destination Port: 5004
- RTP payload type: 0 (PCMU)
```

#### SIP Device:

```
SIP Settings:
- SIP Server: <server-ip>:5060
- Codec: G.711 μ-law
- Transport: UDP
- Registration: Not required
```

#### RTSP Source:

```
RTSP URL: rtsp://<device-ip>/audio
Enable via FFmpeg or GStreamer receiver
```

## Configuration

### Pipeline Configuration

Edit `config/pipeline_config.py`:

```python
# Audio settings (if needed)
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
```

### Audio Settings

Edit `config/audio_settings.json`:

```json
{
    "transcription": {
        "model": "models/whisper-large-v3-hebrew",
        "device": "cpu",
        "language": "he"
    },
    "tts": {
        "engine": "pyttsx3",
        "voice": "hebrew",
        "rate": 150
    },
    "rtp": {
        "codec": "pcmu",
        "sample_rate": 8000,
        "payload_type": 0
    }
}
```

## Testing

### Test Audio System:

```bash
python scripts/test_audio_system.py
```

### Test Individual Components:

```python
# Test transcriber
python -c "from services.audio.transcriber import HebrewTranscriber; t = HebrewTranscriber(); print(t.transcribe('test.wav'))"

# Test TTS
python -c "from services.audio.tts import HebrewTTS; t = HebrewTTS(); t.synthesize('שלום', 'output.wav')"

# Test command processor
python -c "from services.audio.command_processor import CommandProcessor; p = CommandProcessor(); print(p.process('עקוב'))"
```

### Test with cURL:

**Check Protocol Status**:

```bash
curl http://localhost:8000/audio/protocols/status
```

**Send TTS Request**:

```bash
curl -X POST http://localhost:8000/audio/speak \
  -H "Content-Type: application/json" \
  -d '{"text": "שלום עולם", "dest_ip": "192.168.1.100", "dest_port": 5004}'
```

**Upload Audio for Transcription**:

```bash
curl -X POST http://localhost:8000/audio/transcription/file \
  -F "file=@audio.wav"
```

## Performance Optimization

### Model Selection:

-   **Whisper Large v3**: Highest accuracy (~500 tokens/sec on CPU)
-   **Whisper Medium**: Faster (~800 tokens/sec), slightly lower accuracy
-   **Whisper Small**: Fastest (~1200 tokens/sec), good for real-time

To change model, modify `services/audio/transcriber.py`:

```python
MODEL_PATH = "models/whisper-medium-hebrew"  # or whisper-small-hebrew
```

### Audio Buffer Size:

Adjust in `services/audio/audio_pipeline.py`:

```python
AUDIO_CHUNK_SIZE = 16000  # 1 second at 16kHz (smaller = lower latency)
```

### Threading:

The pipeline uses separate threads for:

-   Audio reception
-   Transcription
-   Command processing
-   TTS generation
-   RTP transmission

## Troubleshooting

### Common Issues:

1. **"No Hebrew voice found"**:

    - Install espeak: `brew install espeak` (macOS) or `apt-get install espeak` (Linux)
    - Verify: `espeak --voices | grep hebrew`

2. **"Model not found"**:

    - Ensure Whisper model is in `models/whisper-large-v3-hebrew/`
    - Check all required files are present (config.json, pytorch_model.bin, etc.)

3. **"Port already in use"**:

    - Check if port is occupied: `lsof -i :5004`
    - Kill process: `kill -9 <PID>`
    - Or use different port

4. **Poor transcription quality**:

    - Check audio quality (16kHz, mono, clear speech)
    - Verify Hebrew language setting
    - Consider using Whisper Large v3 for better accuracy

5. **RTP audio not received**:
    - Check firewall settings
    - Verify destination IP/port
    - Use Wireshark to inspect RTP packets
    - Ensure codec compatibility (G.711 μ-law recommended)

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Field Devices                             │
│  (Cameras, Radios, Phones, Sensors)                         │
└──────────┬──────────┬──────────┬──────────┬─────────────────┘
           │          │          │          │
      ┌────▼────┐┌───▼────┐┌───▼────┐┌────▼─────┐
      │  RTSP   ││Raw RTP ││  SIP   ││ FFmpeg/  │
      │         ││  UDP   ││        ││GStreamer │
      └────┬────┘└───┬────┘└───┬────┘└────┬─────┘
           │         │         │          │
           └─────────┴─────────┴──────────┘
                     │
          ┌──────────▼───────────┐
          │ Unified Audio Manager│
          │ (unified_receiver.py)│
          └──────────┬───────────┘
                     │
          ┌──────────▼───────────┐
          │   Audio Pipeline     │
          │ (audio_pipeline.py)  │
          └──────────┬───────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
   ┌────▼─────┐ ┌───▼──────┐ ┌──▼─────┐
   │Transcriber│ │Command  │ │  TTS   │
   │(Whisper) │ │Processor│ │(pyttsx3)│
   │          │ │         │ │        │
   └────┬─────┘ └───┬─────┘ └──┬─────┘
        │           │          │
        └───────────┴──────────┘
                    │
          ┌─────────▼──────────┐
          │   RTP Sender       │
          │ (rtp_sender.py)    │
          └─────────┬──────────┘
                    │
          ┌─────────▼──────────┐
          │   Field Devices    │
          │   (Audio Output)   │
          └────────────────────┘
```

## File Structure

```
services/audio/
├── __init__.py
├── transcriber.py              # Hebrew Whisper transcription
├── command_processor.py        # Hebrew command detection
├── tts.py                      # Hebrew text-to-speech
├── rtp_sender.py              # RTP audio transmission
├── audio_pipeline.py          # Main orchestrator
├── raw_rtp_receiver.py        # Raw UDP/RTP receiver
├── sip_server.py              # SIP + RTP VoIP server
├── stream_integrations.py     # FFmpeg/GStreamer integration
├── unified_receiver.py        # Multi-protocol manager
└── rtp_server.py              # RTSP/RTP server (existing)

api/routes/
├── audio.py                   # Audio API endpoints (20+ endpoints)

config/
├── audio_settings.json        # Audio configuration

models/
└── whisper-large-v3-hebrew/   # Hebrew transcription model
```

## API Reference

See complete API documentation in the code comments or access Swagger UI:

```
http://localhost:8000/docs
```

## Future Enhancements

Potential improvements:

1. WebRTC support for browser-based communication
2. Multi-language support (Arabic, English)
3. Speaker diarization (identify multiple speakers)
4. Emotion detection from voice
5. Audio encryption for secure communication
6. Recording and playback of conversations
7. Integration with cloud speech services (AWS Transcribe, Google STT)

## License

Part of Ronai-Vision project.

## Support

For issues or questions, refer to the main project documentation or contact the development team.

---

**Last Updated**: December 11, 2025
**Version**: 1.0.0
