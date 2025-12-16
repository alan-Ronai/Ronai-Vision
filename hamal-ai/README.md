# HAMAL-AI Security Command Center
# חמ"ל AI - מרכז שליטה ובקרה

AI-powered security command center for real-time monitoring, threat detection, and emergency response.

## Features

- **Real-time Camera Monitoring**: MJPEG/HLS streaming from RTSP cameras
- **AI Detection**: YOLO-based object detection with Mac MPS acceleration
- **Emergency Mode**: Full-screen alerts with visual/audio notifications
- **Radio Transcription**: Live Hebrew transcription via Whisper
- **Voice Commands**: Automatic simulation triggers ("רחפן", "צפרדע", etc.)
- **Soldier App**: Mobile video upload from field personnel
- **Gemini Integration**: Advanced scene analysis and vehicle verification

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (React)                        │
│                    http://localhost:5173                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Socket.IO + REST
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Backend (Express.js)                       │
│                    http://localhost:3000                        │
│                                                                 │
│  ├── Events API                                                 │
│  ├── Cameras API                                                │
│  ├── Uploads API                                                │
│  ├── Radio API                                                  │
│  └── Stream API                                                 │
└─────────────────────────────────────────────────────────────────┘
         │                              │
         │ HTTP                         │ MongoDB
         ▼                              ▼
┌─────────────────────┐     ┌─────────────────────┐
│   AI Service        │     │     MongoDB         │
│   (FastAPI)         │     │                     │
│   :8000             │     │   - Events          │
│                     │     │   - Cameras         │
│   - YOLO Detection  │     └─────────────────────┘
│   - Gemini Analysis │
│   - TTS Service     │
└─────────────────────┘
         │
         │ RTP Audio
         ▼
┌─────────────────────┐
│   Radio Service     │
│   (Python)          │
│   :5004 UDP         │
│                     │
│   - RTP Receiver    │
│   - Whisper STT     │
│   - Command Detect  │
└─────────────────────┘
```

## Quick Start

### Prerequisites

- Node.js 20+
- Python 3.11+
- MongoDB (local or Atlas URI)
- FFmpeg (for HLS conversion)

### 1. Backend Setup

```bash
cd hamal-ai/backend
cp .env.example .env
# Edit .env with your MongoDB URI

npm install
npm run dev
```

### 2. Frontend Setup

```bash
cd hamal-ai/frontend
npm install
npm run dev
```

Open http://localhost:5173

### 3. AI Service Setup

```bash
cd hamal-ai/ai-service
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt

# Copy environment file
cp .env.example .env
# Edit .env with your GEMINI_API_KEY

python main.py
```

### 4. Radio Service (Optional)

```bash
cd hamal-ai/radio-service
# Uses parent project's venv and RTP receiver

# Start the transcriber
python transcriber.py
```

### 5. Mobile App (Optional)

```bash
cd hamal-ai/mobile
npm install

# Update server URL in app
# Edit app/index.jsx and app/camera.jsx - change SERVER_URL

npx expo start
```

## Environment Variables

### Backend (.env)
```env
PORT=3000
MONGODB_URI=mongodb://localhost:27017/hamal
AI_SERVICE_URL=http://localhost:8000
FRONTEND_URL=http://localhost:5173
```

### AI Service (.env)
```env
GEMINI_API_KEY=your_api_key
BACKEND_URL=http://localhost:3000
YOLO_MODEL=yolo12n.pt
DEVICE=auto
PORT=8000
```

## API Endpoints

### Events
- `GET /api/events` - List events (with filtering)
- `GET /api/events/recent` - Recent events
- `GET /api/events/critical` - Unacknowledged critical events
- `POST /api/events` - Create event
- `PATCH /api/events/:id/acknowledge` - Acknowledge event

### Cameras
- `GET /api/cameras` - List cameras
- `POST /api/cameras` - Add camera
- `PATCH /api/cameras/:id/status` - Update status
- `POST /api/cameras/seed` - Seed demo cameras

### Radio
- `POST /api/radio/transcription` - Receive transcription
- `GET /api/radio/transcriptions` - Get recent transcriptions
- `POST /api/radio/command` - Trigger simulation

### Stream
- `POST /api/stream/start-hls` - Start RTSP→HLS
- `POST /api/stream/stop-hls` - Stop HLS
- `GET /api/stream/status` - Stream status

### Uploads
- `POST /api/uploads/soldier-video` - Upload soldier video
- `POST /api/uploads/snapshot` - Upload detection snapshot

## Socket.IO Events

### Client → Server
- `identify` - Identify client type
- `camera:select` - Select main camera
- `emergency:acknowledge` - Acknowledge emergency
- `emergency:end` - End emergency mode
- `simulation:trigger` - Trigger simulation

### Server → Client
- `event:new` - New event created
- `emergency:start` - Emergency started
- `emergency:end` - Emergency ended
- `radio:transcription` - New radio transcription
- `camera:status` - Camera status changed

## Demo Mode

The frontend includes demo controls (🎮 button in bottom-left):

1. **Create Events**: info/warning/critical
2. **Trigger Simulations**:
   - 🚁 Drone dispatch
   - 📞 Phone call
   - 📢 PA announcement
   - 📻 Code broadcast
   - ✅ End event

## Simulation Types

| Type | Hebrew | Trigger Keywords |
|------|--------|-----------------|
| `drone_dispatch` | רחפן הוקפץ | רחפן, חוזי |
| `phone_call` | חיוג למפקד | התקשרו, חייגו |
| `pa_announcement` | כריזה | כריזה, כרזו |
| `code_broadcast` | שידור קוד | צפרדע |
| `threat_neutralized` | חדל | חדל, סיום, נוטרל |

## Integration with Existing System

This project integrates with your existing Ronai-Vision codebase:

- Uses existing `SimpleRTPReceiver` for radio audio
- Can share YOLO models from `models/` directory
- Compatible with existing camera configurations

To use existing streams:
```javascript
// In frontend, update API_URL to point to existing API
const API_URL = 'http://localhost:8000'; // Existing FastAPI
```

## Development

### Project Structure
```
hamal-ai/
├── backend/          # Express.js server
├── frontend/         # React + Vite + Tailwind
├── ai-service/       # Python FastAPI
├── radio-service/    # Radio transcription
├── mobile/           # Expo React Native
└── data/             # Local storage
    ├── hls/          # HLS segments
    ├── clips/        # Video clips
    └── recordings/   # Audio recordings
```

### Running All Services

```bash
# Terminal 1 - Backend
cd backend && npm run dev

# Terminal 2 - Frontend
cd frontend && npm run dev

# Terminal 3 - AI Service
cd ai-service && python main.py

# Terminal 4 - Radio (optional)
cd radio-service && python transcriber.py
```

## License

MIT

## Contact

For questions or support, contact the development team.
