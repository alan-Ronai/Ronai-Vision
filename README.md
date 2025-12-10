# Ronai-Vision

Real-time RTSP/video ingestion pipeline with object detection, segmentation (SAM2), tracking, and re-identification (ReID).
Built for CPU-first local development on M3 MacBook with GPU-ready containerization.

## Pipeline Overview

The Ronai-Vision pipeline consists of the following stages (executed sequentially or per-camera in parallel):

1. **Camera Input** — RTSP/file reader simulated via `LocalCameraSimulator` (dev) or RTSP (prod)
2. **Object Detection** — **YOLO12n** (Ultralytics) detects objects (default: people, cars, etc.) → boxes + confidence scores
3. **Segmentation** — **SAM2** (Segment Anything Model 2) creates precise masks for detected objects
4. **ReID Extraction** — **OSNet** (via `torchreid`) extracts 256-dim feature vectors from detected objects
5. **Per-Camera Tracking** — **BoT-SORT** tracks objects frame-to-frame within each camera
6. **Cross-Camera ReID** — **FAISS** aggregates embeddings and links tracks across cameras (global IDs)
7. **Rendering** — Overlay bounding boxes + track IDs on frames (optional: mask overlay)
8. **Output** — Publish frames to MJPEG/WebSocket stream or save locally

### Key Points

-   **YOLO (Detection):** All objects are detected first; detections drive the rest of the pipeline.
-   **SAM2 (Segmentation):** Optional but recommended for precise masks. Can be filtered by class (e.g., only segment "person" boxes).
-   **OSNet (ReID):** Enables cross-camera tracking. Runs only on detected + segmented objects.
-   **Parallel Mode:** When `PARALLEL=true`, each camera runs independently; per-camera trackers are isolated, but embeddings are aggregated to the same FAISS store for global ID consistency.

## Project Structure

```
.
├── api/
│   ├── routes/          # FastAPI route handlers (health, ptz, websocket)
│   ├── server.py        # FastAPI app creation
│   └── websocket.py     # WebSocket connection manager
├── services/
│   ├── camera/          # RTSP/file readers and simulators
│   ├── detector/        # YOLO wrapper
│   ├── segmenter/       # SAM2 wrapper (not yet implemented)
│   ├── tracker/         # Tracking logic (not yet implemented)
│   ├── reid/            # Re-ID feature extraction (not yet implemented)
│   └── ptz/             # PTZ interface and simulator
├── entrypoints/
│   └── dev_run.py       # Local dev runner
├── config/              # Configuration files (camera, env)
└── assets/              # Sample video files for testing
```

## Quick Start

### 1. Setup Python Environment

```bash
# Create and activate virtual environment (M1/M3 MacBook recommended: conda + conda-forge)
# For conda users (recommended on macOS):
conda create -n ronai python=3.12 -c conda-forge
conda activate ronai
pip install -r requirements.txt

# Or standard venv:
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If you plan to run ReID (OSNet) locally, install `torchreid` and ensure `torch` is available:

```bash
# macOS: prefer conda-forge to avoid OpenMP conflicts
conda install pytorch::pytorch torchvision -c pytorch -c conda-forge
pip install torchreid==0.2.5
```

### 2. Run Multi-Camera Pipeline (Standalone)

```bash
# Single-threaded mode (default):
python scripts/run_multi_camera.py

# Parallel mode (per-camera workers + aggregator queue):
PARALLEL=true python scripts/run_multi_camera.py

# Save rendered frames and metadata (off by default):
SAVE_FRAMES=true python scripts/run_multi_camera.py
```

Output: Reads simulated cameras, runs detector→segmenter→reid→tracker, publishes frames to broadcaster.
Check `output/multi/` for saved frames/metadata if `SAVE_FRAMES=true`.

### 3. Run FastAPI Server (with Optional Embedded Runner)

```bash
# Default: server only, no embedded runner
uvicorn api.server:app --host 0.0.0.0 --port 8000

# Embedded runner: start runner as background thread in the server
START_RUNNER=true uvicorn api.server:app --host 0.0.0.0 --port 8000

# Embedded runner + parallel per-camera workers:
START_RUNNER=true PARALLEL=true uvicorn api.server:app --host 0.0.0.0 --port 8000
```

**Note:** Do not use `--reload` with `START_RUNNER=true` to avoid spawning the runner twice.

**Port already in use?** If you see `Address already in use: ('127.0.0.1', 8000)`:

```bash
# Kill the process on port 8000 (macOS/Linux):
lsof -i :8000 | grep LISTEN | awk '{print $2}' | xargs kill -9

# Or use the helper script:
bash scripts/kill_port.sh 8000

# Then restart the server
START_RUNNER=true uvicorn api.server:app
```

### 4. Streaming Endpoints

Once the server is running (with or without embedded runner), access:

#### MJPEG Stream (OpenCV/VLC compatible)

```bash
# Watch cam1 as MJPEG in browser or VLC:
http://localhost:8000/api/stream/mjpeg/cam1
# or:
vlc http://localhost:8000/api/stream/mjpeg/cam1
```

#### WebSocket Stream (Real-time base64 frames)

```bash
# Python example:
import asyncio
import websockets
import json
import base64
import cv2
import numpy as np

async def watch():
    async with websockets.connect("ws://localhost:8000/api/stream/ws/cam1") as ws:
        while True:
            msg = json.loads(await ws.recv())
            if msg.get("type") == "frame":
                jpeg = base64.b64decode(msg["data"])
                img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                cv2.imshow("cam1", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

asyncio.run(watch())
```

#### Stream Status

```bash
curl http://localhost:8000/api/stream/status
# {"last_published": {"cam1": 1234567890.123, "cam2": 1234567890.456}}
```

### 5. Performance Metrics

```bash
# Query runtime per-stage timing averages (when runner is active):
curl http://localhost:8000/api/status/perf
# {"ok": true, "metrics": {"detect_s": 0.123, "segment_s": 0.456, "reid_s": 0.089, ...}}
```

Logs: Runner prints per-stage timing averages every 10 frames (console output).

### 6. Track Queries

```bash
# Query global tracks by ID:
curl http://localhost:8000/api/tracks/1
# {"global_id": 1, "camera_id": "cam1", "track_id": 5, "last_seen": ..., ...}

# List all tracks:
curl http://localhost:8000/api/tracks/
# [{"global_id": 1, ...}, {"global_id": 2, ...}]
```

### 7. Health Check

```bash
curl http://localhost:8000/api/health
# {"status": "ok"}
```

## Configuration

### Camera Configuration

Cameras are configured in `config/camera_settings.json`:

```json
{
    "cameras": {
        "cam1": {
            "type": "simulator",
            "source": "assets/sample_video.mp4",
            "description": "Test video file (development)"
        },
        "cam2": {
            "type": "rtsp",
            "source": "rtsp://username:password@192.168.1.100:554/stream",
            "description": "Real RTSP camera"
        }
    }
}
```

- **type**: Either `"simulator"` (video file) or `"rtsp"` (live camera)
- **source**: File path for simulator, RTSP URL for real cameras

### Environment Variables

Create or edit `config/dev.env`:

```dotenv
# Device: 'cpu' or 'cuda'
DEVICE=cpu

# Camera configuration file path
CAMERA_CONFIG=config/camera_settings.json

# Stream server URL (for publishing frames from runner to server)
STREAM_SERVER_URL=http://127.0.0.1:8000
STREAM_PUBLISH_TOKEN=AlanRules

# Save rendered frames and metadata locally (off by default)
SAVE_FRAMES=false

# Enable per-camera parallel workers (off by default)
PARALLEL=false

# Only process specific classes (comma-separated, e.g., "person" or "person,car")
ALLOWED_CLASSES=person

# Embedded runner (when running server)
START_RUNNER=false
```

Load the env file before running:

```bash
# Bash:
export $(cat config/dev.env | xargs)
uvicorn api.server:app

# Or inline:
$(cat config/dev.env | xargs) uvicorn api.server:app
```

## SAM2 Segmentation & Text Prompting

SAM2 (Segment Anything Model 2) does **not** accept free-text prompts directly. Instead, use spatial prompts:

### Recommended: Box Prompting (via YOLO Detector)

```python
from services.detector import YOLODetector
from services.segmenter.sam2_segmenter import SAM2Segmenter

detector = YOLODetector(model_name="yolo12n.pt", device="cpu")
segmenter = SAM2Segmenter(model_type="small", device="cpu")

frame = cv2.imread("frame.jpg")

# Run detector to find people
det = detector.predict(frame, confidence=0.25)
boxes = det.boxes  # Bounding boxes for detected objects

# Run SAM2 on all boxes
seg = segmenter.segment(frame, boxes=boxes)
masks = seg.masks
```

### Class-Filtered Box Prompting (Text-Like Behavior)

```python
# Run SAM2 only on boxes of specific classes (e.g., "person")
seg = segmenter.segment_from_detections(
    frame,
    boxes=det.boxes,
    class_ids=det.class_ids,
    class_names=det.class_names,
    allowed_class_names=["person"]
)
masks = seg.masks
```

This is the **recommended workflow** for extracting person masks: use your detector to propose boxes, then filter by class name and pass to SAM2.

## Output Rendering

The renderer produces clean output by default:

-   **Boxes + Track IDs** (default) — `FrameRenderer(show_masks=False)`
-   **Boxes + Masks** — `FrameRenderer(show_masks=True)` (optional, for debugging)

To enable mask overlays in the stream output, modify `scripts/run_multi_camera.py`:

```python
renderer = FrameRenderer(show_masks=True)  # Enable mask blending
```

## Shutdown & Graceful Cleanup

### Embedded Runner (via FastAPI)

```bash
# Start server with runner:
START_RUNNER=true uvicorn api.server:app

# Stop: press Ctrl-C
# The server will:
# 1. Set the runner's stop_event
# 2. Join the runner thread (5s timeout)
# 3. Call cm.stop_all() in finally block
# Exit cleanly.
```

### Standalone Runner

```bash
python scripts/run_multi_camera.py
# Press Ctrl-C
# Cooperative stop checks + finally block cleanup
```

## Development Notes

-   **Per-Camera Workers (PARALLEL=true):** Independent threads for each camera reduce bottlenecks. Each worker runs detect→segment→reid→tracker and pushes embeddings to a central aggregator queue. Aggregator updates ReID store and returns global IDs.
-   **Timing Metrics:** Logged every 10 frames. Query via `GET /api/status/perf` for running averages.
-   **Model Warmup:** First inference may be slow; runner performs a warmup pass on startup to amortize.
-   **Frame Rate:** Default sleep(0.05) between iterations = ~20 fps. Adjust `time.sleep()` in `run_loop()` for higher throughput.
-   **FAISS Index Persistence:** ReID store persists latest embedding per track ID; call `store.decay_stale()` to remove inactive tracks (future: add HTTP endpoint).

## Roadmap

1. ✅ RTSP/file reader scaffold
2. ✅ PTZ simulator and interface
3. ✅ FastAPI server with health/ptz endpoints
4. ✅ WebSocket scaffold (echo mode)
5. ✅ YOLO detector wrapper
6. ✅ SAM2 segmenter wrapper (CPU mode)
7. ✅ Tracker implementation (BoT-SORT per-camera)
8. ✅ ReID feature extraction (OSNet)
9. ✅ Multi-camera runner with aggregator
10. ✅ Frame overlay renderer (bboxes, IDs, optional masks)
11. ✅ MJPEG + WebSocket + publish streaming endpoints
12. ✅ Per-stage timing metrics and perf endpoint
13. ✅ Optional per-camera parallelization
14. ⏳ Example detectors (weapons, car-color)
15. ⏳ Checkpoint persistence (save/restore ReID store)
16. ⏳ Admin endpoint for index rebuild/decay
17. ⏳ Async event handlers and alerts
18. ⏳ Comprehensive integration tests
19. ⏳ Runbook + conda/conda-forge troubleshooting guide
20. ⏳ WebRTC production deployment (media server)

## Contributing

See `PROJECT_BRIEF.md` for acceptance criteria and design notes.
