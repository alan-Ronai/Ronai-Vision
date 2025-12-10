✅ FULL PROJECT BRIEF — Real-Time Video Segmentation System Using SAM2

Detailed explanation of the problem and the desired solution

(Readable, shareable with engineers or an AI agent)

Problem — what's broken / missing today

You want a production-quality real-time RTSP video segmentation & tracking system that:

ingests one or more RTSP camera streams (including PTZ cameras like Hikvision DS-MH6171),

runs modern computer-vision models (SAM2 for high-quality segmentation + detector + tracker + ReID as needed),

produces live masks and/or overlayed video and structured metadata (bounding boxes, IDs, timestamps),

drives PTZ cameras automatically based on Multi-Object Tracking (MOT) decisions (center group, follow runner, handoff between cameras),

runs locally for development (MacBook M3, CPU-only) and can be containerized and deployed later to GPU hosts (AWS G4/G5, Runpod, on-prem PC, Jetson).

Right now you have the design, constraints, and algorithms in mind, but you need a complete, runnable, shareable project that:

is modular so parts can be swapped (SAM2 vs FastSAM, YOLO versions, trackers),

can be developed and debugged on a Mac (slow CPU testing) and then packaged into a GPU-ready Docker image with no code changes,

includes a basic Web API / WebSocket dashboard and PTZ simulation so you can validate logic without physical PTZ cameras.

Desired solution — what success looks like

A single codebase you can develop and test locally which:

Ingests RTSP or local video using OpenCV/FFmpeg with per-camera threads and frame queues.

Processes frames with a pluggable inference pipeline: Detector → (optional) Segmenter (SAM2) → Tracker → ReID → Event logic.

Produces outputs:

live overlayed video (OpenCV window) for development,

structured JSON per-frame with masks (binary) or polygons,

option to stream processed video via RTMP/WebRTC in production.

Controls PTZ via an abstract PTZ interface with a simulation implementation and a Hikvision ISAPI implementation (HTTP/Digest auth) for production.

Is containerized with a Dockerfile and environment-based GPU/CPU toggle.

Has instrumented logs, config files, and small test data so an AI agent or human dev can iterate quickly.

Full folder structure (complete, ready to hand to an AI agent)
ai-vision-system/
│
├── .env.dev # development env variables (DEVICE=cpu etc)
├── .env.prod # production env variables (DEVICE=cuda etc)
├── README.md
├── docker/
│ ├── Dockerfile # main Dockerfile (GPU ready)
│ └── docker-compose.yml # optional dev compose with simulation cameras
│
├── configs/
│ ├── camera_settings.json # camera list + PTZ config + HFOVs
│ ├── models.json # model choices and paths
│ └── logging.yml
│
├── scripts/
│ ├── build_image.sh
│ ├── run_local.sh
│ ├── run_gpu_test.sh
│ └── download_sample_videos.sh
│
├── models/ # downloaded models (gitignored)
│ ├── yolo12n.pt
│ ├── yolov8n.pt
│ ├── sam2_small.pt
│ └── osnet_x1_0_imagenet.pth # optional: OSNet checkpoint for ReID (load via torchreid)
│
├── src/
│ ├── main.py # orchestrator: loads config and starts services
│ ├── entrypoints/
│ │ ├── dev_run.py # run everything in dev mode (sim cameras)
│ │ └── prod_run.py # run in prod config
│ │
│ ├── services/
│ │ ├── camera/
│ │ │ ├── rtsp_reader.py # RTSP/FFmpeg reader (threaded)
│ │ │ └── simulator.py # local video file -> simulated camera stream
│ │ │
│ │ ├── detector/
│ │ │ ├── base_detector.py # abstract detector interface
│ │ │ └── yolov8_detector.py # YOLO wrapper (Ultralytics)
│ │ │
│ │ ├── segmenter/
│ │ │ ├── base_segmenter.py # abstract segmenter interface
│ │ │ └── sam2_segmenter.py # SAM2 wrapper (slow in CPU)
│ │ │
│ │ ├── tracker/
│ │ │ ├── base_tracker.py
│ │ │ └── botsort_tracker.py
│ │ │
│ │ ├── reid/
│ │ │ ├── reid_store.py # FAISS / vector store wrapper
│ │ │ └── osnet_reid.py
│ │ │
│ │ ├── ptz/
│ │ │ ├── ptz_interface.py # abstract PTZ controller
│ │ │ ├── ptz_simulator.py # simulate pan/tilt/zoom locally
│ │ │ └── hikvision_isapi.py # production implementation
│ │ │
│ │ └── output/
│ │ ├── renderer.py # drawing masks & overlays
│ │ ├── rtmp_output.py # ffmpeg RTMP publisher
│ │ └── webrtc_output.py # aiortc pipeline (optional)
│ │
│ ├── api/
│ │ ├── server.py # FastAPI app bootstrap
│ │ ├── routes.py # endpoints (/start, /stop, /ptz, /prompt)
│ │ └── websocket.py # websockets for events & bbox stream
│ │
│ ├── utils/
│ │ ├── math_utils.py # px->deg conversions, smoothing/PID
│ │ ├── video_utils.py # to/from numpy <-> ffmpeg frames
│ │ └── logging_utils.py
│ │
│ └── tests/
│ ├── test_simulation.py
│ └── test_detector_cpu.py
│
└── docs/
├── architecture.md
├── api_spec.md
└── deploy_guide.md

Notes:

models/ is .gitignore but listed so AI/human agent knows where to place large model files.

configs/camera_settings.json contains HFOV and PTZ ranges per camera (needed for math).

entrypoints/dev_run.py boots the simulator + API + UI. prod_run.py expects RTSP URLs and optional GPU.

File-level descriptions (short explanations of important files)

main.py — top-level orchestrator (parses env, loads config, starts camera services, detector, tracker, API).

rtsp_reader.py — robust reader with reconnect logic and a small queue per camera. Uses OpenCV or FFmpeg backend.

simulator.py — reads a video file and provides identical API to rtsp_reader (get_frame(), start(), stop()).

yolov8_detector.py — wraps Ultralytics YOLO model with a .predict(frame) returning boxes, scores, classes.

sam2_segmenter.py — wrapper that accepts a frame & optional prompt (point/box) and returns masks. In dev it runs in CPU (one frame at time).

botsort_tracker.py — maintains track IDs per camera; exposes update(detections) -> tracks.

ptz_interface.py — abstract functions move_relative(pan_speed, tilt_speed), move_to(pan_deg, tilt_deg), go_preset(n). Production hikvision_isapi.py does real HTTP calls.

renderer.py — overlays masks, polygons, and tracks for local UI and for frames fed to RTMP/WebRTC outputs.

server.py — FastAPI endpoints for control and health-checks. websocket.py streams per-frame metadata to UI.

math_utils.py — functions for pixel→degrees conversions (needs HFOV), deadzone logic, smoothing (EMA or PID).

Step-by-step development + test plan (executable checklist)

Follow these steps locally on your MacBook M3. Each step includes commands and expected outcomes.

Step 0 — repo bootstrapping

Create repo and folder structure above (or run the AI agent to scaffold).

Add .env.dev:

DEVICE=cpu
CAMERA_CONFIG=configs/camera_settings.json
LOG_LEVEL=DEBUG

Create requirements.txt:

fastapi
uvicorn[standard]
opencv-python-headless
ultralytics
numpy
pillow
faiss-cpu
pydantic
requests
aiohttp
aiortc # optional, for WebRTC output
python-multipart

Create virtualenv, install:

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Outcome: Local dev env ready.

Step 1 — camera simulation & RTSP reader

Tasks:

Implement src/services/camera/simulator.py.

Implement src/services/camera/rtsp_reader.py with thread + queue. Use cv2.VideoCapture for file or RTSP.

Test:

python3 -m src.entrypoints.dev_run

# dev_run should start 1 simulated camera and print "frame read" messages

Expected result: frames are read repeatedly from sample video; no exceptions.

Step 2 — detector basic (CPU)

Tasks:

Implement src/services/detector/yolov8_detector.py with CPU device default.

Add src/tests/test_detector_cpu.py that loads an image and prints bounding boxes.

Test:

python3 src/services/detector/yolov8_detector.py # or run test

Expected: boxes printed (may be slow). This verifies model and wrapper.

Step 3 — simple pipeline: read → detect → render

Tasks:

Implement pipeline in entrypoints/dev_run.py:

start simulator

read frame

call detector.detect(frame)

call renderer overlay and cv2.imshow()

Test:

python3 -m src.entrypoints.dev_run

Expected: a window shows video with bounding boxes, updated in (slow) real time.

Step 4 — tracker (MOT) integration

Tasks:

Implement botsort_tracker.py skeleton or use ByteTrack/BoT-SORT Python wrappers.

Maintain per-camera tracks and display IDs on overlay.

Test:

Run pipeline: detections are assigned persistent track IDs across frames.

Expected: people/objects have persistent IDs for the short test video.

Step 5 — PTZ simulation + auto-framing logic

Tasks:

Implement ptz/ptz_simulator.py.

Implement math_utils.compute_pan_deg() and utils.deadzone().

Add logic: compute super-box for people, calculate desired pan/tilt/zoom, apply smoothing (EMA), call ptz_simulator.move_to().

Test:

Run pipeline; print PTZ moves each step; optionally show cropped/zoomed simulated camera output reflecting PTZ.

Expected: PTZ simulator prints movement commands; the simulated cropped view recenters around super-box.

Step 6 — SAM2 segmenter (development, CPU, low FPS)

Tasks:

Add sam2_segmenter.py wrapper. Use a small model or a mock if you don’t have weights.

Add API to request interactive segmentation (point/box) and automatic mode.

Test:

Select a frame, run segmenter on a single frame, overlay mask.

Expected: mask overlays are visible (slow, likely <1 FPS).

Step 7 — API + WebSocket for control and live metadata

Tasks:

Implement api/server.py and api/websocket.py.

Add endpoints:

POST /ptz/move — move camera

POST /prompt — send interactive prompt to SAM (x,y or box)

GET /health

Test:

Run server and call endpoints with curl or Postman.

Connect a WebSocket client to receive per-frame JSON: frame_id, detections, tracks.

Expected: endpoints accept commands, WebSocket pushes JSON each frame.

Step 8 — Dockerize (CPU-first)

Tasks:

Create docker/Dockerfile (simple CPU image).

Add build_image.sh and run_local.sh.

Example docker/Dockerfile (CPU):

FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN apt-get update && apt-get install -y ffmpeg && pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "src.entrypoints.dev_run"]

Test:

cd docker
./build_image.sh
./run_local.sh

Expected: container starts and runs the simulation pipeline.

Step 9 — GPU-ready Dockerfile (for later)

Add a GPU-optimized Dockerfile (commented) that uses nvidia/cuda base, installs PyTorch + CUDA, and uses --gpus when running.

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Install apt deps, python, pip, ffmpeg...

# pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118

# pip install -r requirements.txt

Do not run this locally on Mac; use on cloud or GPU host later.

Step 10 — small integration tests + sample runs on cheap GPU

When ready to test real-time performance:

Push Docker image to registry (ECR/GitHub/Runpod).

Spin up a cheap Runpod/T4 instance.

Run the GPU container and provide real RTSP URLs.

Validate FPS, latency, PTZ commands.

Expected: real-time masks and PTZ control with acceptable latency.

Implementation math & heuristics (copy-ready)

Pixel → degrees conversion (put in utils/math_utils.py):

def px_per_deg(frame_width, hfov_deg):
return frame_width / hfov_deg

def pan_deg_for_pixel_offset(offset_px, frame_width, hfov_deg):
return offset_px / px_per_deg(frame_width, hfov_deg)

Normalized offset → speed:

def normalized_offset(offset_px, frame_width):
return offset_px / (frame_width / 2.0)

def speed_from_norm(offset_norm, max_speed=100, deadzone=0.05):
if abs(offset_norm) < deadzone:
return 0
return max(-max_speed, min(max_speed, int(offset_norm \* max_speed)))

Super-box for group framing:

def super_box(boxes):
x_min = min(b[0] for b in boxes)
y_min = min(b[1] for b in boxes)
x_max = max(b[2] for b in boxes)
y_max = max(b[3] for b in boxes)
return x_min, y_min, x_max, y_max

EMAsmoothing:

smoothed = alpha _ new_value + (1-alpha) _ prev_value

Deployment checklist (when you decide to push to production)

Choose deployment target: AWS EC2 G5/G4 or on-prem RTX 40xx / Jetson Orin.

Provision instance with GPU and Docker + NVIDIA Container Toolkit (if NVIDIA).

Push Docker image to registry and pull on target.

Set .env.prod with DEVICE=cuda, camera credentials, and PTZ credentials.

Run container with --gpus all (NVIDIA) or appropriate GPU runtime.

Monitor CPU/GPU usage, FPS, latency. Add autoscaling / health checks if needed.

Extras you should include in the repo (recommended)

docs/deploy_guide.md with exact AWS instance types & commands.

docs/ptz_protocols.md with example ISAPI calls for Hikvision and how to map speed/angles.

benchmarks/ with sample FPS numbers for dev and prod runs.

LICENSE and CONTRIBUTING.md.

1. Project Title

Real-Time RTSP Video Segmentation & Mask Streaming System (Using SAM2)

2. Project Summary

Build a system that ingests a live RTSP video stream, processes each frame through SAM2 (Segment Anything Model 2) for real-time segmentation, and outputs either:

segmented frames (with overlays),

raw masks per frame, or

object-level tracking data.

The system must run locally on a developer’s machine (MacBook M3) for development and later be containerized for deployment to:

AWS GPU instance (EC2 G5/G6 or ECS Fargate GPU),

or On-Prem GPU server

or any container-running environment.

3. Primary Objectives

Ingest an RTSP stream in real time.

Process frames with SAM2 (auto-segmentation or object tracking mode).

Return segmentation per frame: overlayed video OR binary masks OR object tracking.

Stream output via:

WebRTC (real-time, low latency),

RTMP/HLS (if needed),

Or local display (OpenCV).

Allow interactive prompting: e.g., click object → SAM2 tracks it across frames.

Containerize entire system for later deployment.

4. Functional Requirements
   4.1 Input

Must support RTSP URLs such as:

rtsp://user:pass@ip:port/path

Should handle network interruptions gracefully.

Should allow switching between:

local webcam,

local video file,

RTSP.

4.2 Processing Pipeline

Each incoming frame must be passed through:

Decoding layer

ffmpeg / OpenCV to convert RTSP → numpy frame.

SAM2 inference layer
Support the following modes:

Automatic segmentation
(Sam2AutomaticMaskGenerator)

Interactive segmentation
(initial prompt: point or box)

Memory tracking
SAM2 must maintain memory across frames.

Post-processing layer

Blending masks into the frame

Coloring masks

Generating metadata:

mask_id, confidence, polygon, bbox

4.3 Output

Support any of the following output modes:

Mode 1 — Local display

Show live segmentation in an OpenCV window.

Mode 2 — Output segmented video stream

Options:

WebRTC

RTMP → e.g., YouTube / media server

HLS (IOS-friendly)

Mode 3 — Output machine-readable mask data

Example JSON per frame:

{
"frame": 204,
"objects": [
{"id": 1, "mask": "<binary>", "bbox": [x,y,w,h], "label": null},
{"id": 2, "mask": "<binary>"}
]
}

5. Non-Functional Requirements
   5.1 Performance

Target >= 5 FPS on M3 MacBook for development.

Optimize for >= 20–30 FPS on cloud GPU deployment.

5.2 Architecture

Modular components:

rtsp_reader/

sam_engine/

mask_renderer/

stream_output/

api_server/

Easy to swap models.

5.3 Deployment

Generate a Dockerfile that:

uses base image:
pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

installs SAM2 + dependencies

exposes API endpoints

can run on AWS EC2 with GPU:

G5 (Tesla A10G)

G6 (L40S)

6. Tech Stack
   Backend / AI

Python 3.11

PyTorch 2.x

SAM2 official library

OpenCV

FFmpeg (for better decoding)

FastAPI

WebRTC: aiortc (optional)

Container

Docker

Nvidia CUDA base images (for deployment)

Dev Environment (Local)

macOS M3, 8GB RAM

Conda or uv environment

Use CPU for dev, GPU simulated mode (slow but functional)

7. Detailed Tasks (For AI Agent)
   Phase 1 — Environment Setup

Set up Python env that works on macOS ARM.

Install SAM2 and verify inference on a static image.

Implement a simple script to run automatic segmentation on a folder of frames.

Phase 2 — RTSP Input Layer

Create rtsp_reader.py

Use OpenCV or FFmpeg decoding

Ensure:

stable continuous reading

thread-based frame buffering

reconnection logic

Phase 3 — SAM2 Engine

Create sam_engine.py

Implement:

automatic segmentation

interactive segmentation (point/box)

memory tracking across frames

Allow selection of different SAM2 models:

sam2_hiera_small

sam2_hiera_base

Phase 4 — Mask Rendering

Create mask_renderer.py

Blend binary masks onto frames

Assign random colors

Optional: draw bounding boxes

Phase 5 — Output Options

Implement three independent output modules:

1. Local Viewer
   cv2.imshow(…)

2. RTMP / HLS Output

use ffmpeg subprocess to encode frames back into a video stream.

3. WebRTC Output

use aiortc:

create a video track class

push processed frames into the track

Phase 6 — REST API

Expose control API via FastAPI:

/start

/stop

/prompt (add point/box)

/mode (change segmentation mode)

/fps

/health

Phase 7 — Dockerization

Write Dockerfile

Make it GPU-ready

Test with:

docker run --gpus all segmentation-app

Phase 8 — Optional Add-Ons

Object name classification

Motion tracking

Saving masks to disk

Analytics dashboard

8. Deliverables

Git repository with:

/src

/docker

/config

README.md (full setup instructions)

Dockerfile (GPU ready)

Python package or modular code structure

Demo script for:

python run_local_viewer.py --rtsp rtsp://...

Benchmark report:

FPS on M3 CPU mode

FPS on AWS GPU

9. Acceptance Criteria

SAM2 runs locally on Mac M3 in CPU mode.

RTSP stream is processed without crashes.

Masks generated per frame correctly.

Optional: real-time display works.

Docker image builds successfully.

System runs on GPU instance without code changes.
