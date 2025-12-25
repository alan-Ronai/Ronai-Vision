"""
HAMAL-AI Detection Service - Unified

FastAPI service for AI-powered detection using:
- YOLO for object detection
- ReID (DeepSort) for persistent tracking
- Gemini for ALL analysis (vehicle, person, threat assessment)
- Google Cloud TTS for Hebrew announcements

Supports Mac MPS acceleration.
"""

import os

# Disable gRPC fork handlers BEFORE importing any gRPC-related modules (like google.generativeai)
# This prevents the "Other threads are currently calling into gRPC, skipping fork() handlers" warning
# when using subprocess (FFmpeg) alongside gRPC (Gemini API)
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel
import threading
import time
import torch
from ultralytics import YOLO
import numpy as np
import cv2
import asyncio
import httpx
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Filter to suppress noisy polling endpoint logs
class PollingEndpointFilter(logging.Filter):
    """Filter out frequent polling endpoint logs to reduce noise."""
    SUPPRESSED_PATHS = [
        "/api/detections/",  # WebRTC overlay polling
        "/detection/stats",  # Stats polling
        "/health",           # Health checks
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        for path in self.SUPPRESSED_PATHS:
            if path in message and "HTTP/1.1\" 200" in message:
                return False  # Suppress successful polling requests
        return True

# Apply filter to uvicorn access logger
logging.getLogger("uvicorn.access").addFilter(PollingEndpointFilter())

# Import services
from services.reid import ReIDTracker
from services.gemini import GeminiAnalyzer
from services.tts_service import TTSService
from services.streaming import get_rtsp_manager, RTSPConfig, get_ffmpeg_manager, FFmpegConfig
from services.streaming import get_gstreamer_manager, GStreamerConfig, is_gstreamer_available
from services.detection_loop import init_detection_loop, get_detection_loop, LoopConfig
from services.detection import get_frame_buffer_manager, get_stable_tracker
from services.radio import (
    init_radio_service, get_radio_service, stop_radio_service,
    init_transcribers, get_gemini_transcriber, get_whisper_transcriber,
    get_whisper_semaphore, get_transcriber_stats, record_transcription_stats,
    record_whisper_queue_wait
)
from services.radio.radio_transmit import router as radio_transmit_router
from services.recording import init_recording_manager, get_recording_manager

# RTSP Backend selection
RTSP_BACKEND = os.getenv("RTSP_BACKEND", "ffmpeg").lower()
if RTSP_BACKEND == "gstreamer" and not is_gstreamer_available():
    logger.warning("GStreamer requested but not available, falling back to FFmpeg")
    RTSP_BACKEND = "ffmpeg"
logger.info(f"RTSP Backend: {RTSP_BACKEND}")

# FastAPI app
app = FastAPI(
    title="HAMAL-AI Detection Service - Unified",
    description="AI-powered security detection with ReID tracking and Gemini analysis",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(radio_transmit_router, prefix="/api")

# Device detection
def get_device():
    device_env = os.getenv("DEVICE", "auto")
    if device_env != "auto":
        return device_env

    if torch.backends.mps.is_available():
        logger.info("🍎 Using Apple Metal (MPS)")
        return "mps"
    elif torch.cuda.is_available():
        logger.info("🎮 Using CUDA")
        return "cuda"
    else:
        logger.info("💻 Using CPU")
        return "cpu"

DEVICE = get_device()
# Export DEVICE to environment so ReID models can use it
os.environ["DEVICE"] = DEVICE

# Limit PyTorch threads when running on CPU to prevent 400% CPU usage
if DEVICE == "cpu":
    torch.set_num_threads(2)  # Limit to 2 threads for CPU inference
    logger.info("Limited PyTorch to 2 CPU threads")

# Model paths
MODEL_PATH = os.getenv("YOLO_MODEL", "yolo12m.pt")  # Upgraded from yolo12n for better accuracy
CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", "0.35"))

logger.info("=" * 60)
logger.info("YOLO Detection Configuration")
logger.info("=" * 60)
logger.info(f"Model: {MODEL_PATH}")
logger.info(f"Base Confidence: {CONFIDENCE:.2f}")
logger.info(f"Device: {DEVICE}")
logger.info("=" * 60)

# FPS Configuration
TARGET_FPS = int(os.getenv("TARGET_FPS", "15"))
DETECTION_FPS = int(os.getenv("DETECTION_FPS", str(TARGET_FPS)))
STREAM_FPS = int(os.getenv("STREAM_FPS", str(TARGET_FPS)))

def get_target_fps() -> int:
    """Get the target FPS for RTSP readers and cameras."""
    return TARGET_FPS

def validate_fps(camera_fps: int, target_fps: int) -> int:
    """Ensure target FPS doesn't exceed camera FPS."""
    if target_fps > camera_fps:
        logger.warning(
            f"Target FPS ({target_fps}) > Camera FPS ({camera_fps}). "
            f"Lowering to camera FPS."
        )
        return camera_fps
    return target_fps

logger.info("=" * 60)
logger.info("FPS Configuration")
logger.info("=" * 60)
logger.info(f"Target FPS (RTSP/Camera): {TARGET_FPS}")
logger.info(f"Detection FPS:            {DETECTION_FPS}")
logger.info(f"Stream FPS:               {STREAM_FPS}")
logger.info("=" * 60)

# Models directory - all models should be in hamal-ai/ai-service/models/
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)
logger.info(f"Models directory: {MODELS_DIR}")


def load_yolo_model_optimized(model_name: str, device: str = "cpu") -> Optional[YOLO]:
    """Load a YOLO model, preferring optimized formats when available.

    Priority order:
    - CUDA: TensorRT (.engine) > ONNX (.onnx) > PyTorch (.pt)
    - CPU/MPS: ONNX (.onnx) > PyTorch (.pt)

    TensorRT provides 2-5x faster inference on NVIDIA GPUs.
    ONNX provides 20-40% faster inference on CPU.

    To create optimized models:
        yolo export model=yolov8m.pt format=engine device=0  # TensorRT (CUDA only)
        yolo export model=yolov8m.pt format=onnx             # ONNX (works everywhere)

    Args:
        model_name: Model filename (e.g., "yolov8m.pt")
        device: Target device ("cpu", "cuda", "mps")

    Returns:
        Loaded YOLO model or None if not found
    """
    # Strip "models/" prefix if present
    if model_name.startswith("models/"):
        model_name = model_name[7:]

    # Get base name without extension
    base_name = model_name.rsplit('.', 1)[0] if '.' in model_name else model_name

    # Build search paths based on device - prioritize optimized formats
    search_paths = []

    if device == "cuda":
        # CUDA: TensorRT > ONNX > PyTorch
        search_paths.extend([
            os.path.join(MODELS_DIR, f"{base_name}.engine"),
            f"{base_name}.engine",
            os.path.join(MODELS_DIR, f"{base_name}.onnx"),
            f"{base_name}.onnx",
        ])
    else:
        # CPU/MPS: ONNX > PyTorch (TensorRT doesn't work on CPU)
        search_paths.extend([
            os.path.join(MODELS_DIR, f"{base_name}.onnx"),
            f"{base_name}.onnx",
        ])

    # Always add PyTorch paths as fallback
    search_paths.extend([
        os.path.join(MODELS_DIR, model_name),
        model_name,
        f"models/{model_name}",
    ])

    # Try each path
    for model_path in search_paths:
        if os.path.exists(model_path):
            try:
                is_tensorrt = model_path.endswith('.engine')
                is_onnx = model_path.endswith('.onnx')

                format_name = "TensorRT" if is_tensorrt else ("ONNX" if is_onnx else "PyTorch")
                logger.info(f"Loading YOLO model from: {model_path} ({format_name})")

                model = YOLO(model_path)

                # Move to device (TensorRT/ONNX models handle device internally)
                if not is_tensorrt and not is_onnx:
                    model.to(device)

                if is_tensorrt:
                    logger.info(f"✅ YOLO {model_name} loaded with TensorRT (2-5x faster on GPU)")
                elif is_onnx:
                    logger.info(f"✅ YOLO {model_name} loaded with ONNX Runtime (20-40% faster on CPU)")
                else:
                    logger.info(f"✅ YOLO {model_name} loaded on {device}")
                    # Suggest optimization based on device
                    if device == "cuda":
                        logger.info(f"💡 TIP: Export to TensorRT for 2-5x speedup: "
                                  f"yolo export model={model_path} format=engine device=0")
                    else:
                        logger.info(f"💡 TIP: Export to ONNX for 20-40% speedup on CPU: "
                                  f"yolo export model={model_path} format=onnx")

                return model
            except Exception as e:
                logger.warning(f"Could not load model from {model_path}: {e}")
                continue

    return None


# Load main YOLO model (prefers ONNX on CPU, TensorRT on CUDA)
yolo = load_yolo_model_optimized(MODEL_PATH, DEVICE)

if yolo is None:
    logger.info("⚠️ YOLO model not found, downloading default")
    yolo = YOLO("yolo11n.pt")
    yolo.to(DEVICE)
    logger.info(f"✅ YOLO loaded on {DEVICE}")

# Optional: Load weapon detection model with TensorRT preference
weapon_model_name = os.getenv("WEAPON_MODEL", "firearm-yolov8n.pt")
weapon_detector = load_yolo_model_optimized(weapon_model_name, DEVICE)

# Initialize services
tracker = ReIDTracker(max_age=30, n_init=3)
gemini = GeminiAnalyzer()
tts = TTSService()

# Backend URL for sending events
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:3000")

# Vehicle classes for YOLO
VEHICLE_CLASSES = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']
PERSON_CLASS = 'person'

# RTSP Stream Manager - stores active streams
class RTSPStreamManager:
    def __init__(self):
        self.streams: Dict[str, Dict] = {}  # cameraId -> {cap, thread, frame, running, url}
        self.lock = threading.Lock()

    def get_camera_config(self, camera_id: str) -> Optional[Dict]:
        """Fetch camera config from backend"""
        import httpx
        try:
            response = httpx.get(f"{BACKEND_URL}/api/cameras/{camera_id}", timeout=5.0)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Failed to get camera config: {e}")
        return None

    def start_stream(self, camera_id: str, rtsp_url: str, username: str = None, password: str = None) -> bool:
        """Start capturing from RTSP stream.

        NOTE: This is the legacy OpenCV-based reader. The FFmpeg-based rtsp_manager
        (from services.streaming) is more robust and is used for the main detection loop.
        This manager is kept for compatibility with SSE endpoints.
        """
        with self.lock:
            if camera_id in self.streams and self.streams[camera_id].get("running"):
                logger.info(f"Stream already running for {camera_id}")
                return True  # Already running

        # Check if this is a local file path (not a network URL)
        is_local_file = not rtsp_url.startswith(('rtsp://', 'http://', 'https://', 'rtmp://', 'udp://'))
        if is_local_file:
            # Local file path - resolve relative to Ronai-Vision root
            # Path: .../Ronai-Vision/hamal-ai/ai-service/main.py
            # We need to go up 2 levels to reach Ronai-Vision
            script_dir = Path(__file__).parent  # .../hamal-ai/ai-service
            project_root = script_dir.parent.parent  # Ronai-Vision root
            clean_path = rtsp_url.lstrip('/')
            video_path = project_root / clean_path

            if not video_path.exists() and Path(rtsp_url).is_absolute():
                video_path = Path(rtsp_url)

            rtsp_url = str(video_path.absolute())

            if not video_path.exists():
                logger.error(f"Video file not found: {rtsp_url}")
                return False

            logger.info(f"✓ Resolved local video path: {rtsp_url}")

        # Build full RTSP URL with credentials
        full_url = rtsp_url
        if username and password and "@" not in rtsp_url:
            if "://" in rtsp_url:
                protocol, rest = rtsp_url.split("://", 1)
                full_url = f"{protocol}://{username}:{password}@{rest}"

        logger.info(f"Starting RTSP stream for {camera_id}")

        # Set environment for FFmpeg with TCP transport and error handling
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp|"
            "timeout;5000000|"
            "fflags;+genpts+discardcorrupt|"
            "err_detect;ignore_err"
        )

        # OpenCV VideoCapture with RTSP over TCP
        cap = cv2.VideoCapture(full_url, cv2.CAP_FFMPEG)

        # Configure capture for stability
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Slightly larger buffer for stability
        cap.set(cv2.CAP_PROP_FPS, 15)

        if not cap.isOpened():
            logger.error(f"Failed to open RTSP stream for {camera_id}")
            return False

        # Read first frame with retries
        ret, test_frame = None, None
        for attempt in range(5):
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                break
            time.sleep(0.5)

        if not ret or test_frame is None:
            logger.error(f"Failed to read first frame for {camera_id} after 5 attempts")
            cap.release()
            return False

        logger.info(f"First frame received: {test_frame.shape}")

        with self.lock:
            stream_data = {
                "cap": cap,
                "url": full_url,
                "frame": test_frame,
                "running": True,
                "last_update": time.time(),
                "error": None,
                "reconnect_count": 0,
                "is_local_file": is_local_file  # Track if this is a local video file
            }
            self.streams[camera_id] = stream_data

        thread = threading.Thread(target=self._capture_loop, args=(camera_id,), daemon=True)
        with self.lock:
            self.streams[camera_id]["thread"] = thread
        thread.start()

        logger.info(f"✅ RTSP stream started for {camera_id}")
        return True

    def _capture_loop(self, camera_id: str):
        """Background thread to continuously capture frames.

        Uses adaptive error handling:
        - Tolerates occasional frame drops (common with H.264 streams)
        - Only reconnects after sustained failures
        - Uses exponential backoff for reconnects
        - For local video files: loops back to start on EOF instead of reconnecting
        """
        consecutive_errors = 0
        max_consecutive_errors = 50  # Increased tolerance (~5 seconds at 10fps)
        reconnect_attempts = 0
        max_reconnect_attempts = 10
        base_reconnect_delay = 2.0

        while True:
            # Check if we should stop
            with self.lock:
                if camera_id not in self.streams:
                    break
                stream = self.streams[camera_id]
                if not stream.get("running"):
                    break
                cap = stream["cap"]
                stream_url = stream.get("url", "")
                is_local_file = stream.get("is_local_file", False)

            # Read frame
            ret, frame = cap.read()

            if ret and frame is not None:
                consecutive_errors = 0
                reconnect_attempts = 0  # Reset on successful frame
                with self.lock:
                    if camera_id in self.streams:
                        self.streams[camera_id]["frame"] = frame
                        self.streams[camera_id]["last_update"] = time.time()
                        self.streams[camera_id]["error"] = None
            else:
                # For local video files, loop back to start on EOF
                if is_local_file:
                    # Check if this is EOF (video ended)
                    current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

                    if total_frames > 0 and current_pos >= total_frames - 1:
                        # Video ended - loop back to start
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        logger.debug(f"Camera {camera_id}: Video looped back to start")
                        consecutive_errors = 0
                        continue

                consecutive_errors += 1

                # Only log periodically to avoid spam
                if consecutive_errors == 1:
                    logger.debug(f"Frame read failed for {camera_id}")
                elif consecutive_errors % 25 == 0:
                    logger.warning(f"Frame read failures for {camera_id}: {consecutive_errors}/{max_consecutive_errors}")

                if consecutive_errors >= max_consecutive_errors:
                    if reconnect_attempts >= max_reconnect_attempts:
                        logger.error(f"Max reconnect attempts reached for {camera_id}, giving up")
                        with self.lock:
                            if camera_id in self.streams:
                                self.streams[camera_id]["error"] = "Max reconnects exceeded"
                                self.streams[camera_id]["running"] = False
                        break

                    # Exponential backoff
                    delay = min(base_reconnect_delay * (2 ** reconnect_attempts), 30)
                    reconnect_attempts += 1

                    logger.warning(f"Reconnecting {camera_id} (attempt {reconnect_attempts}/{max_reconnect_attempts}, delay {delay:.1f}s)")
                    cap.release()
                    time.sleep(delay)

                    # Reconnect with fresh FFmpeg options
                    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                        "rtsp_transport;tcp|"
                        "timeout;5000000|"
                        "fflags;+genpts+discardcorrupt|"
                        "err_detect;ignore_err"
                    )
                    new_cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
                    new_cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
                    new_cap.set(cv2.CAP_PROP_FPS, 15)

                    if new_cap.isOpened():
                        # Try to read a frame to verify connection
                        test_ret, test_frame = new_cap.read()
                        if test_ret and test_frame is not None:
                            with self.lock:
                                if camera_id in self.streams:
                                    self.streams[camera_id]["cap"] = new_cap
                                    self.streams[camera_id]["frame"] = test_frame
                                    self.streams[camera_id]["reconnect_count"] = \
                                        self.streams[camera_id].get("reconnect_count", 0) + 1
                            cap = new_cap
                            consecutive_errors = 0
                            logger.info(f"✅ Reconnected to {camera_id}")
                        else:
                            new_cap.release()
                            logger.warning(f"Reconnected but no frames for {camera_id}")
                    else:
                        logger.warning(f"Reconnect failed for {camera_id}")
                        with self.lock:
                            if camera_id in self.streams:
                                self.streams[camera_id]["error"] = "Connection lost"

            # Control frame rate - ~10 FPS capture
            time.sleep(0.1)

    def get_frame(self, camera_id: str) -> Optional[np.ndarray]:
        """Get latest frame for camera"""
        with self.lock:
            if camera_id in self.streams:
                frame = self.streams[camera_id].get("frame")
                if frame is not None:
                    return frame.copy()  # Return a copy to avoid threading issues
        return None

    def stop_stream(self, camera_id: str):
        """Stop capturing from stream"""
        with self.lock:
            if camera_id in self.streams:
                self.streams[camera_id]["running"] = False
                cap = self.streams[camera_id].get("cap")
                if cap:
                    cap.release()
                del self.streams[camera_id]
                logger.info(f"Stopped stream for {camera_id}")

    def get_active_streams(self) -> List[str]:
        """Get list of active stream IDs"""
        with self.lock:
            return list(self.streams.keys())

    def get_stream_info(self, camera_id: str) -> Optional[Dict]:
        """Get info about a stream"""
        with self.lock:
            if camera_id in self.streams:
                stream = self.streams[camera_id]
                return {
                    "camera_id": camera_id,
                    "running": stream.get("running"),
                    "last_update": stream.get("last_update"),
                    "error": stream.get("error"),
                    "reconnect_count": stream.get("reconnect_count", 0),
                    "has_frame": stream.get("frame") is not None
                }
        return None

# Global stream manager
stream_manager = RTSPStreamManager()


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "device": DEVICE,
        "model": MODEL_PATH,
        "gemini_configured": gemini.is_configured(),
        "tts_engine": tts.get_engine_info() if tts.is_configured() else None,
        "weapon_detector": weapon_detector is not None,
        "tracker_stats": tracker.get_stats(),
        "active_streams": stream_manager.get_active_streams()
    }


# ============== STREAMING ENDPOINTS ==============

# Global FFmpeg manager for stable streaming
ffmpeg_manager = get_ffmpeg_manager()


async def generate_mjpeg_frames(camera_id: str, fps: int = 15):
    """Async generator that yields MJPEG frames at specified FPS.

    Uses the FFmpeg-based RTSP reader for stable streaming.
    """
    logger.info(f"Starting MJPEG stream generator for {camera_id} at {fps} FPS")
    frame_count = 0
    no_frame_count = 0
    frame_interval = 1.0 / fps
    last_frame_time = 0

    # Use the FFmpeg-based RTSP reader manager (same as detection loop)
    rtsp_manager = get_rtsp_manager()

    while True:
        now = time.time()

        # Rate limiting
        elapsed = now - last_frame_time
        if elapsed < frame_interval:
            await asyncio.sleep(frame_interval - elapsed)
            continue

        last_frame_time = time.time()

        # Get frame from FFmpeg RTSP reader
        frame = rtsp_manager.get_frame(camera_id)
        if frame is not None:
            no_frame_count = 0  # Reset counter
            # Encode frame as JPEG (lower quality for reduced latency)
            ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
            if ret:
                frame_count += 1
                if frame_count % 100 == 0:
                    logger.debug(f"Streamed {frame_count} frames for {camera_id}")
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        else:
            no_frame_count += 1
            if no_frame_count == 1 or no_frame_count % 50 == 0:
                active_cams = rtsp_manager.get_active_cameras()
                logger.warning(
                    f"MJPEG: No frame for {camera_id} (count={no_frame_count}, "
                    f"active_cams={active_cams})"
                )
            await asyncio.sleep(0.05)  # Slightly longer sleep when no frames


async def generate_ffmpeg_mjpeg_frames(camera_id: str, rtsp_url: str, fps: int = 15):
    """MJPEG frames using FFmpeg for stable RTSP streaming."""
    logger.info(f"Starting FFmpeg MJPEG stream for {camera_id} at {fps} FPS")

    # Get or create FFmpeg stream
    stream = ffmpeg_manager.get_stream(camera_id)
    if not stream:
        config = FFmpegConfig(fps=fps, width=1280, height=720)
        stream = ffmpeg_manager.add_camera(camera_id, rtsp_url, config)
        # Wait for first frame
        await asyncio.sleep(1.0)

    frame_count = 0
    frame_interval = 1.0 / fps
    last_frame_time = 0

    while True:
        now = time.time()

        # Rate limiting
        elapsed = now - last_frame_time
        if elapsed < frame_interval:
            await asyncio.sleep(frame_interval - elapsed)
            continue

        last_frame_time = time.time()

        frame, frame_time = stream.get_frame()
        if frame is not None:
            # Encode frame as JPEG (lower quality for reduced latency)
            ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
            if ret:
                frame_count += 1
                if frame_count % 100 == 0:
                    logger.debug(f"FFmpeg stream: {frame_count} frames for {camera_id}")
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        else:
            await asyncio.sleep(0.01)  # Reduced from 0.1 for lower latency


async def generate_mjpeg_frames_buffered(camera_id: str, rtsp_url: str, fps: int = 15,
                                         username: str = None, password: str = None):
    """MJPEG frames using FrameBuffer for smoother streaming."""
    logger.info(f"Starting buffered MJPEG stream for {camera_id} at {fps} FPS")

    # Get or create frame buffer
    buffer_manager = get_frame_buffer_manager()
    buffer = buffer_manager.get_or_create(
        camera_id=camera_id,
        rtsp_url=rtsp_url,
        username=username,
        password=password,
        target_fps=fps
    )

    frame_count = 0
    frame_interval = 1.0 / fps
    last_frame_time = 0

    while True:
        now = time.time()

        # Rate limiting
        elapsed = now - last_frame_time
        if elapsed < frame_interval:
            await asyncio.sleep(frame_interval - elapsed)
            continue

        last_frame_time = time.time()

        frame = buffer.get_frame()
        if frame is not None:
            # Encode frame as JPEG (lower quality for reduced latency)
            ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
            if ret:
                frame_count += 1
                if frame_count % 100 == 0:
                    logger.debug(f"Buffered stream: {frame_count} frames for {camera_id}")
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        else:
            await asyncio.sleep(0.01)  # Reduced from 0.1 for lower latency


@app.get("/api/stream/mjpeg/{camera_id}")
async def stream_mjpeg(camera_id: str, fps: int = 15):
    """
    MJPEG streaming endpoint - streams live video from RTSP camera.

    Uses the same FFmpeg-based RTSP reader as the detection loop for stability.
    The stream is auto-started when cameras are loaded on startup.

    Args:
        camera_id: Camera identifier
        fps: Target frames per second (default 15)
    """
    logger.info(f"MJPEG stream requested for camera: {camera_id} at {fps} FPS")
    fps = max(1, min(fps, 30))  # Clamp to 1-30 FPS

    # Use the FFmpeg-based RTSP reader manager
    rtsp_manager = get_rtsp_manager()
    active_cams = rtsp_manager.get_active_cameras()
    logger.info(f"MJPEG: Active cameras: {active_cams}, requested: {camera_id}")

    # Check if camera is already being read by detection loop
    if camera_id not in rtsp_manager.get_active_cameras():
        # Camera not loaded - try to start it
        import httpx
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{BACKEND_URL}/api/cameras/{camera_id}", timeout=5.0)
                if response.status_code != 200:
                    raise HTTPException(404, f"Camera {camera_id} not found")

                camera_config = response.json()
                rtsp_url = camera_config.get("rtspUrl")
                if not rtsp_url:
                    raise HTTPException(400, f"Camera {camera_id} has no RTSP URL")

                # Start the FFmpeg reader
                config = RTSPConfig(width=1280, height=720, fps=TARGET_FPS, tcp_transport=True)
                rtsp_manager.add_camera(camera_id, rtsp_url, config)

                # Wait for connection
                await asyncio.sleep(2.0)

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to start camera {camera_id}: {e}")
            raise HTTPException(503, f"Failed to connect to camera {camera_id}")

    # Verify we have frames
    frame = rtsp_manager.get_frame(camera_id)
    if frame is None:
        logger.warning(f"No frame available yet for {camera_id}, waiting...")
        await asyncio.sleep(1.0)
        frame = rtsp_manager.get_frame(camera_id)

    # Log reader stats for debugging
    reader = rtsp_manager.get_reader(camera_id)
    if reader:
        stats = reader.get_stats()
        logger.info(f"MJPEG: Reader stats for {camera_id}: {stats}")
    else:
        logger.warning(f"MJPEG: No reader found for {camera_id}")

    logger.info(f"Starting MJPEG response for {camera_id}, has_frame={frame is not None}")
    return StreamingResponse(
        generate_mjpeg_frames(camera_id, fps),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/stream/mjpeg-buffered/{camera_id}")
async def stream_mjpeg_buffered(camera_id: str, fps: int = 15):
    """
    Buffered MJPEG streaming - smoother playback using FrameBuffer.

    Args:
        camera_id: Camera identifier
        fps: Target frames per second (default 15)
    """
    logger.info(f"Buffered MJPEG stream requested for camera: {camera_id} at {fps} FPS")
    fps = max(1, min(fps, 30))

    # Fetch camera config from backend
    camera_config = stream_manager.get_camera_config(camera_id)
    if not camera_config:
        raise HTTPException(404, f"Camera {camera_id} not found")

    rtsp_url = camera_config.get("rtspUrl")
    if not rtsp_url:
        raise HTTPException(400, f"Camera {camera_id} has no RTSP URL configured")

    return StreamingResponse(
        generate_mjpeg_frames_buffered(
            camera_id=camera_id,
            rtsp_url=rtsp_url,
            fps=fps,
            username=camera_config.get("username"),
            password=camera_config.get("password")
        ),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/stream/ffmpeg/{camera_id}")
async def stream_ffmpeg_mjpeg(camera_id: str, fps: int = 15):
    """
    FFmpeg-based MJPEG streaming - most stable option for RTSP.

    Uses FFmpeg subprocess with TCP transport for reliable streaming.
    Handles H.264 packet loss much better than OpenCV.

    Args:
        camera_id: Camera identifier
        fps: Target frames per second (default 15)
    """
    logger.info(f"FFmpeg MJPEG stream requested for camera: {camera_id} at {fps} FPS")
    fps = max(1, min(fps, 30))

    # Fetch camera config from backend
    camera_config = stream_manager.get_camera_config(camera_id)
    if not camera_config:
        raise HTTPException(404, f"Camera {camera_id} not found")

    rtsp_url = camera_config.get("rtspUrl")
    if not rtsp_url:
        raise HTTPException(400, f"Camera {camera_id} has no RTSP URL configured")

    # Build full URL with credentials if provided
    username = camera_config.get("username")
    password = camera_config.get("password")
    if username and password and "@" not in rtsp_url:
        if "://" in rtsp_url:
            protocol, rest = rtsp_url.split("://", 1)
            rtsp_url = f"{protocol}://{username}:{password}@{rest}"

    return StreamingResponse(
        generate_ffmpeg_mjpeg_frames(camera_id, rtsp_url, fps),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.post("/api/stream/start/{camera_id}")
async def start_stream(
    camera_id: str,
    rtsp_url: str = Query(None, description="RTSP URL (optional, fetches from DB if not provided)"),
    username: str = Query(None),
    password: str = Query(None)
):
    """Manually start an RTSP stream"""
    if not rtsp_url:
        camera_config = stream_manager.get_camera_config(camera_id)
        if not camera_config:
            raise HTTPException(404, f"Camera {camera_id} not found")
        rtsp_url = camera_config.get("rtspUrl")
        username = username or camera_config.get("username")
        password = password or camera_config.get("password")

    if not rtsp_url:
        raise HTTPException(400, "No RTSP URL provided or configured")

    success = stream_manager.start_stream(camera_id, rtsp_url, username, password)
    if success:
        return {"message": f"Stream started for {camera_id}", "status": "ok"}
    else:
        raise HTTPException(503, f"Failed to start stream for {camera_id}")


@app.post("/api/stream/stop/{camera_id}")
async def stop_stream(camera_id: str):
    """Stop an RTSP stream"""
    stream_manager.stop_stream(camera_id)
    return {"message": f"Stream stopped for {camera_id}"}


@app.get("/api/stream/status")
async def stream_status():
    """Get status of all active streams"""
    return {
        "active_streams": stream_manager.get_active_streams(),
        "count": len(stream_manager.get_active_streams())
    }


@app.get("/api/stream/snapshot/{camera_id}")
async def get_snapshot(camera_id: str):
    """Get a single JPEG snapshot from a stream"""
    # First try FFmpeg-based rtsp_manager (handles webcams and main detection streams)
    from services.streaming.rtsp_reader import get_rtsp_manager
    rtsp_mgr = get_rtsp_manager()
    frame = rtsp_mgr.get_frame(camera_id)

    # Fallback to legacy stream_manager
    if frame is None:
        frame = stream_manager.get_frame(camera_id)

    if frame is None:
        # Try to start stream first (only for non-webcam cameras)
        camera_config = stream_manager.get_camera_config(camera_id)
        if camera_config and camera_config.get("rtspUrl"):
            # Skip webcams - they should be started via /detection/start
            if camera_config.get("type") == "webcam":
                raise HTTPException(503, f"Webcam {camera_id} not started. Use detection/start first.")

            stream_manager.start_stream(
                camera_id,
                camera_config["rtspUrl"],
                camera_config.get("username"),
                camera_config.get("password")
            )
            await asyncio.sleep(1)  # Wait for first frame
            frame = stream_manager.get_frame(camera_id)

    if frame is None:
        raise HTTPException(503, f"No frame available for {camera_id}")

    ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
    if not ret:
        raise HTTPException(500, "Failed to encode frame")

    return StreamingResponse(
        iter([jpeg.tobytes()]),
        media_type="image/jpeg"
    )


@app.get("/api/stream/sse/{camera_id}")
async def stream_sse(camera_id: str, fps: Optional[int] = None):
    """
    Server-Sent Events streaming - stable single connection.
    Server controls frame rate. Uses detection loop's stream_fps config if not specified.
    """
    import base64

    # Use configured stream_fps if not explicitly provided
    detection_loop = get_detection_loop()
    if fps is None:
        fps = detection_loop.config.stream_fps if detection_loop else 15

    logger.info(f"SSE stream requested for camera: {camera_id}, fps: {fps}")

    # Clamp FPS to reasonable range
    fps = max(1, min(fps, 30))
    frame_interval = 1.0 / fps

    # Ensure stream is started
    if camera_id not in stream_manager.get_active_streams():
        camera_config = stream_manager.get_camera_config(camera_id)
        if not camera_config:
            raise HTTPException(404, f"Camera {camera_id} not found")

        rtsp_url = camera_config.get("rtspUrl")
        if not rtsp_url:
            raise HTTPException(400, f"Camera {camera_id} has no RTSP URL")

        success = stream_manager.start_stream(
            camera_id,
            rtsp_url,
            camera_config.get("username"),
            camera_config.get("password")
        )
        if not success:
            raise HTTPException(503, f"Failed to connect to camera {camera_id}")

        # Wait for first frame
        await asyncio.sleep(1.0)

    async def generate_sse():
        last_frame_time = 0
        error_count = 0
        max_errors = 10

        while True:
            try:
                current_time = time.time()

                # Rate limit
                if current_time - last_frame_time < frame_interval:
                    await asyncio.sleep(frame_interval - (current_time - last_frame_time))
                    continue

                frame = stream_manager.get_frame(camera_id)

                if frame is not None:
                    # Encode as JPEG (lower quality for reduced latency)
                    ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
                    if ret:
                        # Convert to base64
                        frame_b64 = base64.b64encode(jpeg.tobytes()).decode('utf-8')
                        # Send as SSE event
                        yield f"data: {{\"frame\": \"{frame_b64}\"}}\n\n"
                        last_frame_time = time.time()
                        error_count = 0
                else:
                    error_count += 1
                    if error_count >= max_errors:
                        yield f"data: {{\"error\": \"No frames available\"}}\n\n"
                        break
                    await asyncio.sleep(0.1)  # Reduced from 0.5 for lower latency

            except Exception as e:
                logger.error(f"SSE error for {camera_id}: {e}")
                yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"
                break

    return StreamingResponse(
        generate_sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/detect")
async def detect_frame(
    file: UploadFile = File(...),
    camera_id: str = Query("unknown", description="Camera identifier"),
    analyze_new: bool = Query(True, description="Analyze new objects with Gemini")
):
    """
    Main detection pipeline:
    1. YOLO detects objects
    2. Optional: Weapon detection
    3. ReID tracks them (persistent IDs)
    4. Gemini analyzes NEW objects only
    5. Alert if armed persons detected
    """
    # Read frame
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(400, "Invalid image")

    # Run YOLO detection
    results = yolo(frame, verbose=False, conf=CONFIDENCE)[0]

    # Separate detections by class
    vehicle_dets = []
    person_dets = []

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = results.names[cls]
        bbox = box.xyxy[0].tolist()

        if label in VEHICLE_CLASSES:
            vehicle_dets.append((bbox, conf, label))
        elif label == PERSON_CLASS:
            person_dets.append((bbox, conf, PERSON_CLASS))

    # Weapon detection (if model available)
    weapons = []
    if weapon_detector:
        weapon_results = weapon_detector(frame, verbose=False, conf=0.5)[0]
        for box in weapon_results.boxes:
            weapons.append({
                "type": weapon_results.names[int(box.cls[0])],
                "bbox": box.xyxy[0].tolist(),
                "confidence": float(box.conf[0])
            })

    # Track vehicles with ReID
    tracked_vehicles = tracker.update_vehicles(vehicle_dets, frame)

    # Track persons with ReID
    tracked_persons = tracker.update_persons(person_dets, frame)

    # Analyze NEW objects with Gemini (only if not analyzed before)
    new_analyses = []

    if analyze_new and gemini.is_configured():
        # Analyze new vehicles
        for vehicle in tracked_vehicles:
            track_id = vehicle["track_id"]
            if not tracker.has_been_analyzed(track_id):
                analysis = await gemini.analyze_vehicle(frame, vehicle["bbox"])
                tracker.save_metadata(track_id, {
                    "type": "vehicle",
                    "analysis": analysis,
                    "camera_id": camera_id
                })
                tracker.add_appearance(track_id, camera_id, vehicle["bbox"])

                new_analyses.append({
                    "track_id": track_id,
                    "type": "vehicle",
                    "analysis": analysis
                })

                # Send event to backend
                await send_event({
                    "type": "detection",
                    "severity": "info",
                    "title": f"רכב זוהה - {analysis.get('color', '')} {analysis.get('manufacturer', '')}",
                    "source": camera_id,
                    "cameraId": camera_id,
                    "details": {
                        "vehicle": analysis,
                        "track_id": track_id
                    }
                })

        # Analyze new persons
        for person in tracked_persons:
            track_id = person["track_id"]
            if not tracker.has_been_analyzed(track_id):
                analysis = await gemini.analyze_person(frame, person["bbox"])
                tracker.save_metadata(track_id, {
                    "type": "person",
                    "analysis": analysis,
                    "camera_id": camera_id
                })
                tracker.add_appearance(track_id, camera_id, person["bbox"])

                new_analyses.append({
                    "track_id": track_id,
                    "type": "person",
                    "analysis": analysis
                })

                # Check if armed - trigger emergency (only if legacy mode enabled)
                # NOTE: Modern detection uses rule engine which controls when/if to trigger
                if analysis.get("armed") and os.environ.get("LEGACY_ALERTS_ENABLED", "false").lower() == "true":
                    await trigger_emergency(
                        camera_id,
                        analysis,
                        tracked_persons,
                        tracked_vehicles
                    )

    # Check if weapons detected near people (even if person already analyzed)
    if weapons and tracked_persons:
        for person in tracked_persons:
            meta = tracker.get_metadata(person["track_id"])
            if meta:
                prev_analysis = meta.get("analysis", {})
                if not prev_analysis.get("armed"):
                    # Re-analyze - might be armed now
                    analysis = await gemini.analyze_person(frame, person["bbox"])
                    if analysis.get("armed"):
                        tracker.save_metadata(person["track_id"], {"analysis": analysis})
                        # Only trigger if legacy mode enabled
                        if os.environ.get("LEGACY_ALERTS_ENABLED", "false").lower() == "true":
                            await trigger_emergency(
                                camera_id,
                                analysis,
                                tracked_persons,
                                tracked_vehicles
                            )

    return {
        "camera_id": camera_id,
        "frame_size": {"width": frame.shape[1], "height": frame.shape[0]},
        "tracked_vehicles": len(tracked_vehicles),
        "tracked_persons": len(tracked_persons),
        "weapons_detected": len(weapons),
        "new_analyses": len(new_analyses),
        "vehicles": [
            {
                "track_id": v["track_id"],
                "bbox": v["bbox"],
                "metadata": tracker.get_metadata(v["track_id"])
            }
            for v in tracked_vehicles
        ],
        "persons": [
            {
                "track_id": p["track_id"],
                "bbox": p["bbox"],
                "metadata": tracker.get_metadata(p["track_id"])
            }
            for p in tracked_persons
        ],
        "weapons": weapons
    }


@app.post("/analyze-body-cam")
async def analyze_body_camera(
    file: UploadFile = File(...),
    camera_id: str = Query("bodycam", description="Body camera identifier")
):
    """
    Analyze body camera frame to check if threat is neutralized.
    Gemini looks for person lying on ground with weapon secured.
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(400, "Invalid image")

    analysis = await gemini.analyze_threat_neutralized(frame)

    if analysis.get("threatNeutralized"):
        # End emergency - send event
        await send_event({
            "type": "simulation",
            "severity": "info",
            "title": "חדל - סוף אירוע",
            "source": camera_id,
            "details": {
                "simulation": "threat_neutralized",
                "analysis": analysis
            }
        })

        # Generate TTS announcement
        if tts.is_configured():
            try:
                audio_path = await tts.generate_event_end_announcement()
                analysis["audio_path"] = audio_path
            except Exception as e:
                logger.error(f"TTS error: {e}")

        # Mark armed persons as resolved
        for track_id in tracker.get_armed_persons():
            meta = tracker.get_metadata(track_id)
            if meta:
                meta["status"] = "resolved"
                tracker.save_metadata(track_id, meta)

    return analysis


@app.post("/analyze-scene")
async def analyze_scene(
    file: UploadFile = File(...),
    prompt: str = Query(None, description="Custom analysis prompt")
):
    """General scene analysis with Gemini"""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(400, "Invalid image")

    result = await gemini.analyze_scene(frame, prompt)
    return result


@app.post("/verify-vehicle")
async def verify_vehicle(files: List[UploadFile] = File(...)):
    """
    Verify if two frames show the same vehicle.
    Useful for cross-camera tracking verification.
    """
    if len(files) < 2:
        raise HTTPException(400, "At least 2 images required")

    frames = []
    for f in files:
        contents = await f.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is not None:
            frames.append(frame)

    if len(frames) < 2:
        raise HTTPException(400, "Could not decode images")

    # For now, analyze first frame fully
    # TODO: Implement cross-frame verification
    result = await gemini.analyze_vehicle(frames[0], [0, 0, frames[0].shape[1], frames[0].shape[0]])
    return result


@app.post("/tts")
async def text_to_speech(
    text: str = Query(..., description="Text to convert to speech")
):
    """Generate Hebrew TTS audio"""
    if not tts.is_configured():
        raise HTTPException(503, "TTS service not configured")

    try:
        audio_path = await tts.generate(text)
        return {"audio_path": audio_path, "text": text}
    except Exception as e:
        raise HTTPException(500, f"TTS error: {str(e)}")


class TTSGenerateRequest(BaseModel):
    """Request body for TTS generation"""
    text: str
    language: str = "he"
    transmit_radio: bool = True  # Whether to transmit via radio


@app.post("/tts/generate")
async def tts_generate(request: TTSGenerateRequest):
    """
    Generate Hebrew TTS audio and optionally transmit via radio.

    This endpoint generates TTS audio using Gemini and can automatically
    transmit it via the radio module.
    """
    if not tts.is_configured():
        raise HTTPException(503, "TTS service not configured")

    try:
        # Generate the audio file
        audio_path = await tts.generate(request.text)

        # Read and convert to base64
        import base64
        from pathlib import Path

        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise HTTPException(500, "Generated audio file not found")

        # Read the audio file
        with open(audio_file, 'rb') as f:
            audio_data = f.read()

        # Get sample rate from WAV header if possible
        sample_rate = 24000  # Default for Gemini TTS
        if audio_file.suffix.lower() == '.wav' and len(audio_data) > 44:
            import struct
            # WAV sample rate is at bytes 24-28
            try:
                sample_rate = struct.unpack('<I', audio_data[24:28])[0]
            except:
                pass

        audio_base64 = base64.b64encode(audio_data).decode('utf-8')

        # Transmit via radio if requested and audio is valid
        radio_transmitted = False
        if request.transmit_radio and len(audio_data) > 100:
            try:
                from services.radio.radio_transmit import transmit_audio_internal
                radio_result = await transmit_audio_internal(audio_data, "wav", "high")
                radio_transmitted = radio_result.get("success", False)
                if radio_transmitted:
                    logger.info(f"TTS transmitted via radio: {request.text[:50]}...")
                else:
                    logger.warning(f"Radio transmission failed: {radio_result.get('error', 'unknown')}")
            except ImportError:
                logger.warning("Radio transmit module not available")
            except Exception as radio_err:
                logger.error(f"Radio transmission error: {radio_err}")

        return {
            "audio_base64": audio_base64,
            "sample_rate": sample_rate,
            "format": "wav",
            "text": request.text,
            "file_path": str(audio_path),
            "radio_transmitted": radio_transmitted
        }

    except Exception as e:
        logger.error(f"TTS generate error: {e}", exc_info=True)
        raise HTTPException(500, f"TTS error: {str(e)}")


@app.get("/tts/file/{filename}")
async def get_tts_file(filename: str):
    """Serve generated TTS audio file"""
    audio_path = Path(tts.output_dir) / filename
    if not audio_path.exists():
        raise HTTPException(404, "Audio file not found")
    return FileResponse(audio_path, media_type="audio/mpeg")


@app.get("/tracker/stats")
async def get_tracker_stats():
    """Get ReID tracker statistics"""
    return tracker.get_stats()


@app.get("/tracker/stats/{camera_id}")
async def get_tracker_stats_for_camera(camera_id: str):
    """Get tracker statistics for a specific camera.

    Returns counts of visible (actively detected) vs total tracked objects
    for the specified camera.
    """
    from services.detection import get_bot_sort_tracker

    bot_sort = get_bot_sort_tracker()
    if not bot_sort:
        return {"error": "BoT-SORT tracker not initialized"}

    stats = bot_sort.get_stats(camera_id=camera_id)

    return {
        "camera_id": camera_id,
        "reid_tracker": tracker.get_stats(),
        "bot_sort": stats,
        # Convenience fields for frontend
        "persons": {
            "visible": stats.get("camera_stats", {}).get("visible_persons", 0),
            "total": stats.get("camera_stats", {}).get("persons", 0),
        },
        "vehicles": {
            "visible": stats.get("camera_stats", {}).get("visible_vehicles", 0),
            "total": stats.get("camera_stats", {}).get("vehicles", 0),
        },
    }


@app.post("/tracker/refresh-analysis/{track_id}")
async def refresh_analysis(
    track_id: str,
    camera_id: str = Query(None, description="Camera ID to get current frame from")
):
    """
    Refresh Gemini analysis for a tracked object.

    Gets the current frame from the camera and re-analyzes the object
    if it's still visible in the scene.

    Args:
        track_id: Track ID of the object to refresh (e.g., 't_5', 'v_3', or '5')
        camera_id: Camera to use for getting current frame

    Returns:
        Updated analysis or error if object not visible
    """
    from services.backend_sync import parse_gid
    from services.detection import get_bot_sort_tracker

    # Parse track ID to get numeric GID
    gid = parse_gid(track_id)
    if gid is None:
        raise HTTPException(400, f"Invalid track_id: {track_id}")

    # Get detection loop
    detection_loop = get_detection_loop()
    if not detection_loop:
        raise HTTPException(503, "Detection loop not running")

    # Get both trackers - BoT-SORT (main) and StableTracker (backup)
    bot_sort = get_bot_sort_tracker()
    stable_tracker = get_stable_tracker()

    # Determine object type and get object
    obj = None
    obj_type = None
    bbox = None

    logger.info(f"Refresh analysis request for track_id: {track_id}, gid: {gid}")

    # STEP 1: Try to find using BoT-SORT's get_track method (direct lookup)
    if bot_sort:
        obj = bot_sort.get_track(track_id)
        if obj:
            obj_type = getattr(obj, 'object_type', 'person' if track_id.startswith('t_') or track_id.startswith('p_') else 'vehicle')
            bbox = obj.bbox
            if not camera_id:
                camera_id = obj.last_seen_camera
            logger.info(f"Found track {track_id} in BoT-SORT: type={obj_type}, camera={camera_id}")

    # STEP 2: Try StableTracker's get_track method
    if not obj:
        obj = stable_tracker.get_track(track_id)
        if obj:
            obj_type = obj.object_type
            bbox = obj.bbox
            if not camera_id:
                camera_id = getattr(obj, 'camera_id', None)
            logger.info(f"Found track {track_id} in StableTracker: type={obj_type}")

    # STEP 3: Try by string track_id matching in vehicle tracks (handles v_12 format)
    if not obj and track_id.startswith('v_'):
        obj_type = 'vehicle'
        # Check BoT-SORT vehicles
        if bot_sort:
            for track in bot_sort.get_active_tracks('vehicle'):
                if track.track_id == track_id or str(track.track_id) == track_id:
                    obj = track
                    bbox = track.bbox
                    if not camera_id:
                        camera_id = track.last_seen_camera
                    logger.info(f"Found vehicle track {track_id} by iteration")
                    break
        # Check StableTracker vehicles
        if not obj:
            for vehicle in stable_tracker.get_all_vehicles():
                if str(vehicle.track_id) == track_id or vehicle.track_id == track_id:
                    obj = vehicle
                    bbox = vehicle.bbox
                    if not camera_id:
                        camera_id = getattr(vehicle, 'camera_id', None)
                    logger.info(f"Found vehicle track {track_id} in StableTracker by iteration")
                    break

    # STEP 4: Try by string track_id matching in person tracks
    if not obj and (track_id.startswith('t_') or track_id.startswith('p_')):
        obj_type = 'person'
        # Check BoT-SORT persons
        if bot_sort:
            for track in bot_sort.get_active_tracks('person'):
                if track.track_id == track_id or str(track.track_id) == track_id:
                    obj = track
                    bbox = track.bbox
                    if not camera_id:
                        camera_id = track.last_seen_camera
                    logger.info(f"Found person track {track_id} by iteration")
                    break
        # Check StableTracker persons
        if not obj:
            for person in stable_tracker.get_all_persons():
                if str(person.track_id) == track_id or person.track_id == track_id:
                    obj = person
                    bbox = person.bbox
                    if not camera_id:
                        camera_id = getattr(person, 'camera_id', None)
                    logger.info(f"Found person track {track_id} in StableTracker by iteration")
                    break

    # STEP 5: Try numeric GID matching (handles both old and new ID formats)
    # Old format: t_5, v_10
    # New format: p_0_5, v_1_10 (prefix_session_gid)
    if not obj:
        # Try persons
        if bot_sort:
            for track in bot_sort.get_active_tracks('person'):
                # Check old format (t_gid, p_gid)
                if track.track_id == f"t_{gid}" or track.track_id == f"p_{gid}":
                    obj = track
                    obj_type = 'person'
                    bbox = track.bbox
                    if not camera_id:
                        camera_id = track.last_seen_camera
                    break
                # Check new format (p_session_gid) - extract gid from end
                if isinstance(track.track_id, str) and track.track_id.startswith('p_'):
                    parts = track.track_id.split('_')
                    if len(parts) >= 3 and parts[-1] == str(gid):
                        obj = track
                        obj_type = 'person'
                        bbox = track.bbox
                        if not camera_id:
                            camera_id = track.last_seen_camera
                        break
        if not obj and bot_sort:
            for track in bot_sort.get_active_tracks('vehicle'):
                # Check old format (v_gid)
                if track.track_id == f"v_{gid}":
                    obj = track
                    obj_type = 'vehicle'
                    bbox = track.bbox
                    if not camera_id:
                        camera_id = track.last_seen_camera
                    break
                # Check new format (v_session_gid) - extract gid from end
                if isinstance(track.track_id, str) and track.track_id.startswith('v_'):
                    parts = track.track_id.split('_')
                    if len(parts) >= 3 and parts[-1] == str(gid):
                        obj = track
                        obj_type = 'vehicle'
                        bbox = track.bbox
                        if not camera_id:
                            camera_id = track.last_seen_camera
                        break

    if not obj:
        logger.warning(f"Track {track_id} not found in any tracker")
        raise HTTPException(404, "האובייקט כבר לא נמצא בסצנה. ניתן לרענן ניתוח רק עבור אובייקטים פעילים.")

    if not camera_id:
        raise HTTPException(400, "Camera ID not provided and object has no camera")

    if not bbox:
        raise HTTPException(400, "Object has no bounding box")

    # Convert bbox to [x1, y1, x2, y2] format if it's in (x, y, w, h) format
    if len(bbox) == 4:
        x, y, w, h = bbox
        if w < 100 and h < 100:  # Likely already in (x, y, w, h) format
            bbox = [x, y, x + w, y + h]
        # else assume it's already [x1, y1, x2, y2]

    # Get current frame from RTSP manager
    rtsp_manager = get_rtsp_manager()
    frame = rtsp_manager.get_frame(camera_id)

    if frame is None:
        raise HTTPException(503, f"No frame available from camera {camera_id}")

    # Run Gemini analysis
    try:
        # Get existing analysis to compare cutout quality
        existing_cutout = None
        existing_cutout_quality = 0
        if hasattr(obj, 'metadata') and obj.metadata:
            existing_analysis = obj.metadata.get('analysis', {})
            existing_cutout = existing_analysis.get('cutout_image')
            if existing_cutout:
                # Calculate quality score for existing cutout (sharpness via Laplacian variance)
                try:
                    import base64
                    existing_bytes = base64.b64decode(existing_cutout)
                    existing_arr = np.frombuffer(existing_bytes, dtype=np.uint8)
                    existing_img = cv2.imdecode(existing_arr, cv2.IMREAD_COLOR)
                    if existing_img is not None:
                        gray = cv2.cvtColor(existing_img, cv2.COLOR_BGR2GRAY)
                        existing_cutout_quality = cv2.Laplacian(gray, cv2.CV_64F).var()
                        logger.debug(f"Existing cutout quality: {existing_cutout_quality:.2f}")
                except Exception as e:
                    logger.debug(f"Could not calculate existing cutout quality: {e}")

        if obj_type == 'vehicle':
            analysis = await gemini.analyze_vehicle(frame, bbox)
        else:
            analysis = await gemini.analyze_person(frame, bbox)

        # Compare cutout quality - keep the better one
        new_cutout = analysis.get('cutout_image')
        if new_cutout and existing_cutout:
            try:
                import base64
                new_bytes = base64.b64decode(new_cutout)
                new_arr = np.frombuffer(new_bytes, dtype=np.uint8)
                new_img = cv2.imdecode(new_arr, cv2.IMREAD_COLOR)
                if new_img is not None:
                    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
                    new_cutout_quality = cv2.Laplacian(gray, cv2.CV_64F).var()
                    logger.info(f"Cutout quality comparison - Old: {existing_cutout_quality:.2f}, New: {new_cutout_quality:.2f}")

                    # Keep old cutout if it's sharper (higher Laplacian variance = sharper)
                    if existing_cutout_quality > new_cutout_quality:
                        logger.info(f"Keeping original cutout (better quality: {existing_cutout_quality:.2f} > {new_cutout_quality:.2f})")
                        analysis['cutout_image'] = existing_cutout
                    else:
                        logger.info(f"Using new cutout (better quality: {new_cutout_quality:.2f} > {existing_cutout_quality:.2f})")
            except Exception as e:
                logger.debug(f"Could not compare cutout quality: {e}")

        # Update tracker metadata
        if hasattr(obj, 'metadata'):
            obj.metadata['analysis'] = analysis

        # Sync to backend
        from services import backend_sync
        await backend_sync.update_analysis(gid=gid, analysis=analysis)

        logger.info(f"Refreshed analysis for {track_id}: {analysis}")

        return {
            "track_id": track_id,
            "gid": gid,
            "type": obj_type,
            "analysis": analysis,
            "camera_id": camera_id
        }

    except Exception as e:
        logger.error(f"Failed to refresh analysis for {track_id}: {e}")
        raise HTTPException(500, f"Analysis failed: {str(e)}")


@app.get("/tracker/objects")
async def get_tracked_objects(
    obj_type: Optional[str] = Query(None, description="Filter by type: vehicle/person")
):
    """Get all tracked objects with their metadata"""
    return tracker.get_all_tracked(obj_type)


@app.get("/tracker/armed")
async def get_armed_persons():
    """Get all persons marked as armed"""
    armed_ids = tracker.get_armed_persons()
    return {
        "count": len(armed_ids),
        "persons": [
            {
                "track_id": tid,
                "metadata": tracker.get_metadata(tid)
            }
            for tid in armed_ids
        ]
    }


@app.post("/tracker/reset")
async def reset_tracker():
    """Reset all trackers and clear stored data"""
    tracker.reset()
    return {"message": "Tracker reset successfully"}


@app.post("/tracker/clear-armed")
async def clear_armed_status():
    """Clear armed status from all tracked persons.

    Called when an alert/scenario is handled to reset the armed indicators
    in the GID Store panel.
    """
    cleared = tracker.clear_armed_status()

    # Also reset scenario hooks and GID limits in rule engine
    try:
        from services.scenario import get_scenario_hooks
        hooks = get_scenario_hooks()
        if hooks:
            hooks.reset_reported()
    except Exception as e:
        logger.warning(f"Failed to reset scenario hooks: {e}")

    try:
        from services.rules import get_rule_engine
        rule_engine = get_rule_engine()
        if rule_engine:
            rule_engine.reset_gid_limits()
    except Exception as e:
        logger.warning(f"Failed to reset GID limits: {e}")

    return {
        "message": f"Cleared armed status for {cleared} persons",
        "cleared_count": cleared
    }


@app.delete("/tracker/cleanup")
async def cleanup_old_tracks(
    max_age_seconds: int = Query(3600, description="Max age in seconds")
):
    """Remove old tracks that haven't been seen"""
    removed = tracker.cleanup_old_tracks(max_age_seconds)
    return {"removed": removed}


@app.delete("/tracker/clear-all")
async def clear_all_tracking():
    """
    Clear ALL tracking data: GID store, ReID gallery, and backend tracked objects.

    This is a full reset - use with caution.
    Handles in-flight operations gracefully by incrementing session ID.
    Returns stats on what was cleared.
    """
    from services.detection import get_bot_sort_tracker

    stats = {
        "tracker_persons": 0,
        "tracker_vehicles": 0,
        "gallery_persons": 0,
        "gallery_vehicles": 0,
        "backend_cleared": False,
        "detection_loop_cleared": False
    }

    logger.info("🗑️ Starting clear-all operation...")

    # 1. Clear BoT-SORT tracker (in-memory GID store)
    # This increments session ID to invalidate in-flight operations
    try:
        tracker = get_bot_sort_tracker()
        if tracker:
            tracker_stats = tracker.clear_all()
            stats["tracker_persons"] = tracker_stats.get("persons", 0)
            stats["tracker_vehicles"] = tracker_stats.get("vehicles", 0)
    except Exception as e:
        logger.warning(f"Error clearing tracker: {e}")

    # 2. Clear detection loop state (analysis buffer, cooldowns, ReID cache, etc.)
    try:
        loop = get_detection_loop()
        if loop:
            loop.clear_state()  # Full state reset including analysis buffer and analyzed GIDs
            stats["detection_loop_cleared"] = True
            logger.info("Cleared detection loop state (analysis buffer, cooldowns, ReID cache)")
    except Exception as e:
        logger.warning(f"Error clearing detection loop: {e}")

    # 3. Clear ReID gallery
    try:
        from services.reid import get_reid_gallery
        gallery = get_reid_gallery()
        if gallery:
            gallery_stats = gallery.get_stats()
            stats["gallery_persons"] = gallery_stats.get("persons", 0)
            stats["gallery_vehicles"] = gallery_stats.get("vehicles", 0)
            gallery.clear()
    except Exception as e:
        logger.warning(f"Error clearing ReID gallery: {e}")

    # 4. Clear backend tracked objects
    try:
        backend_url = os.environ.get("BACKEND_URL", "http://localhost:3000")
        async with httpx.AsyncClient() as client:
            response = await client.delete(f"{backend_url}/api/tracked/clear-all", timeout=5.0)
            if response.status_code == 200:
                stats["backend_cleared"] = True
                logger.info("Cleared backend tracked objects")
            else:
                logger.warning(f"Backend clear failed: HTTP {response.status_code}")
    except Exception as e:
        logger.warning(f"Error clearing backend: {e}")

    # 5. Reset rule engine GID limits
    try:
        from services.rules import get_rule_engine
        rule_engine = get_rule_engine()
        if rule_engine:
            rule_engine.reset_gid_limits()
            logger.info("Reset rule engine GID limits")
    except Exception as e:
        logger.warning(f"Error resetting rule engine: {e}")

    # Brief pause to let in-flight operations complete/fail gracefully
    await asyncio.sleep(0.1)

    logger.info(f"✅ Clear-all complete: {stats}")

    return {
        "status": "cleared",
        "stats": stats,
        "message": "All tracking data cleared"
    }


async def trigger_emergency(
    camera_id: str,
    person_analysis: Dict[str, Any],
    all_persons: List[Dict],
    all_vehicles: List[Dict]
):
    """Trigger full emergency alert"""
    logger.warning(f"🚨 EMERGENCY TRIGGERED from camera {camera_id}")

    # Gather all info for announcement
    vehicle_info = None
    if all_vehicles:
        v_meta = tracker.get_metadata(all_vehicles[0]["track_id"])
        if v_meta:
            vehicle_info = v_meta.get("analysis")

    # Collect all person analyses
    person_analyses = []
    for person in all_persons:
        meta = tracker.get_metadata(person["track_id"])
        if meta and meta.get("analysis"):
            person_analyses.append(meta["analysis"])

    details = {
        "location": f"מצלמה {camera_id}",
        "person_count": len(all_persons),
        "armed": True,
        "weapon_type": person_analysis.get("weaponType"),
        "vehicle": vehicle_info,
        "persons": person_analyses
    }

    # Send critical event to backend
    await send_event({
        "type": "alert",
        "severity": "critical",
        "title": "חדירה ודאית - אנשים חמושים",
        "source": camera_id,
        "cameraId": camera_id,
        "details": details
    })

    # Generate and store TTS announcement
    if tts.is_configured():
        try:
            audio_path = await tts.generate_emergency_announcement(details)
            logger.info(f"Emergency announcement generated: {audio_path}")
            # TODO: Push audio via socket to subscribers
        except Exception as e:
            logger.error(f"Failed to generate announcement: {e}")


async def send_event(event: Dict[str, Any]):
    """Send event to backend"""
    import httpx

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BACKEND_URL}/api/events",
                json=event,
                timeout=5.0
            )
            if response.status_code in (200, 201):
                logger.info(f"Event sent: {event.get('title', 'Unknown')}")
            else:
                logger.warning(f"Event send failed: {response.status_code}")
    except Exception as e:
        logger.error(f"Failed to send event: {e}")


# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info(f"""
╔════════════════════════════════════════════════════════╗
║                                                        ║
║   🤖 HAMAL-AI Detection Service v2.0 (Unified)         ║
║   שירות זיהוי AI משולב                                  ║
║                                                        ║
║   Device: {DEVICE.ljust(10)}                                ║
║   Model: {MODEL_PATH.ljust(12)}                            ║
║   Gemini: {'✅' if gemini.is_configured() else '❌'}                                        ║
║   TTS: {'✅ Gemini (' + tts.voice + ')' if tts.is_configured() else '❌'}                    ║
║   Weapon Det: {'✅' if weapon_detector else '❌'}                                   ║
║   Backend: {BACKEND_URL}                 ║
║                                                        ║
╚════════════════════════════════════════════════════════╝
    """)

    # Initialize detection loop with env configuration
    detection_fps = int(os.getenv("DETECTION_FPS", "15"))
    stream_fps = int(os.getenv("STREAM_FPS", "15"))
    use_reid_recovery = os.getenv("USE_REID_RECOVERY", "false").lower() == "true"

    # Confidence thresholds
    yolo_confidence = float(os.getenv("YOLO_CONFIDENCE", "0.35"))
    weapon_confidence = float(os.getenv("WEAPON_CONFIDENCE", "0.40"))
    recovery_confidence = float(os.getenv("RECOVERY_CONFIDENCE", "0.20"))

    loop_config = LoopConfig(
        backend_url=BACKEND_URL,
        detection_fps=detection_fps,
        stream_fps=stream_fps,
        draw_bboxes=True,
        send_events=True,
        use_reid_recovery=use_reid_recovery,
        weapon_detector=weapon_detector,
        yolo_confidence=yolo_confidence,
        weapon_confidence=weapon_confidence,
        recovery_confidence=recovery_confidence
    )

    logger.info(f"Detection loop config: detection_fps={detection_fps}, stream_fps={stream_fps}, reid_recovery={use_reid_recovery}")
    logger.info(f"Confidence thresholds: yolo={yolo_confidence}, weapon={weapon_confidence}, recovery={recovery_confidence}")

    # Clear backend tracked store on restart to prevent stale data
    backend_url = os.environ.get("BACKEND_URL", "http://localhost:3000")
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.delete(f"{backend_url}/api/tracked/clear-all", timeout=5.0)
            if response.status_code == 200:
                logger.info("✅ Backend tracked store cleared on startup")
            else:
                logger.warning(f"⚠️ Failed to clear backend tracked store: HTTP {response.status_code}")
    except Exception as e:
        logger.warning(f"⚠️ Failed to clear backend tracked store: {e}")

    # CRITICAL: Initialize ReID Gallery BEFORE detection loop starts
    # This loads persistent features from local file so matching works from the first frame
    from services.reid import initialize_reid_gallery
    try:
        await initialize_reid_gallery()
        logger.info("✅ ReID Gallery initialized for persistent re-identification")
    except Exception as e:
        logger.warning(f"⚠️ ReID Gallery initialization failed (non-fatal): {e}")

    detection_loop = init_detection_loop(yolo, tracker, gemini, loop_config)

    # Start detection loop (gallery already initialized with persistent data)
    await detection_loop.start()

    # Set up RTSP manager to feed frames to detection loop
    rtsp_manager = get_rtsp_manager()
    rtsp_manager.add_frame_callback(detection_loop.on_frame)

    # Auto-load cameras from backend
    asyncio.create_task(auto_load_cameras())

    logger.info("✅ Detection loop initialized and ready")

    # Initialize recording manager for video recording with pre-buffer
    recordings_dir = os.getenv("RECORDINGS_DIR", "/tmp/hamal_recordings")
    await init_recording_manager(recordings_dir=recordings_dir, default_fps=TARGET_FPS)
    logger.info(f"📹 Recording manager initialized, output: {recordings_dir}")

    # Initialize global transcribers BEFORE radio service (so model is pre-loaded)
    whisper_model_path = os.getenv("WHISPER_MODEL_PATH", "models/whisper-large-v3-hebrew-ct2")
    save_transcription_audio = os.getenv("SAVE_TRANSCRIPTION_AUDIO", "false").lower() == "true"
    init_transcribers(
        whisper_model_path=whisper_model_path,
        save_audio=save_transcription_audio,
        preload_whisper=True  # Pre-load model at startup
    )

    # Start radio service for RTP audio transcription via EC2 relay
    ec2_host = os.getenv("EC2_RTP_HOST")
    ec2_port = int(os.getenv("EC2_RTP_PORT", "5005"))
    transcription_chunk_duration = float(os.getenv("TRANSCRIPTION_CHUNK_DURATION", "8.0"))
    transcription_silence_threshold = float(os.getenv("TRANSCRIPTION_SILENCE_THRESHOLD", "500.0"))
    transcription_silence_duration = float(os.getenv("TRANSCRIPTION_SILENCE_DURATION", "1.5"))
    transcription_min_duration = float(os.getenv("TRANSCRIPTION_MIN_DURATION", "1.5"))
    transcription_idle_timeout = float(os.getenv("TRANSCRIPTION_IDLE_TIMEOUT", "2.0"))
    save_transcription_audio = os.getenv("SAVE_TRANSCRIPTION_AUDIO", "true").lower() == "true"
    use_vad = os.getenv("USE_VAD", "false").lower() == "true"

    if ec2_host:
        try:
            await init_radio_service(
                ec2_host=ec2_host,
                ec2_port=ec2_port,
                sample_rate=16000,
                backend_url=BACKEND_URL,
                chunk_duration=transcription_chunk_duration,
                silence_threshold=transcription_silence_threshold,
                silence_duration=transcription_silence_duration,
                min_duration=transcription_min_duration,
                idle_timeout=transcription_idle_timeout,
                save_audio=save_transcription_audio,
                use_vad=use_vad
            )
            logger.info(
                f"📻 Radio service started - EC2 relay: {ec2_host}:{ec2_port}, "
                f"VAD: {'enabled' if use_vad else 'disabled'}"
            )
        except Exception as e:
            logger.warning(f"Could not start radio service: {e}")
    else:
        logger.info("📻 Radio service not started (EC2_RTP_HOST not configured)")


async def update_camera_status(camera_id: str, status: str, error_msg: str = None):
    """Update camera status in the backend."""
    import httpx

    try:
        async with httpx.AsyncClient() as client:
            payload = {"status": status}
            if error_msg:
                payload["error"] = error_msg
            await client.patch(
                f"{BACKEND_URL}/api/cameras/{camera_id}/status",
                json=payload,
                timeout=5.0
            )
            logger.debug(f"Updated camera {camera_id} status to {status}")
    except Exception as e:
        logger.debug(f"Failed to update camera status: {e}")


async def auto_load_cameras():
    """Load cameras from backend and start RTSP readers."""
    import httpx

    logger.info(f"🔄 Auto-loading cameras from backend: {BACKEND_URL}/api/cameras")
    await asyncio.sleep(2)  # Wait for backend to be ready

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BACKEND_URL}/api/cameras", timeout=10.0)

            if response.status_code != 200:
                logger.error(f"Failed to load cameras: {response.status_code}")
                return

            cameras = response.json()
            logger.info(f"📋 Found {len(cameras)} cameras from backend")
            rtsp_manager = get_rtsp_manager()
            detection_loop = get_detection_loop()

            # Ensure the frame callback is registered
            if detection_loop and detection_loop.on_frame not in rtsp_manager._frame_callbacks:
                rtsp_manager.add_frame_callback(detection_loop.on_frame)
                logger.info(f"Registered detection loop frame callback")

            for camera in cameras:
                rtsp_url = camera.get('rtspUrl')
                if not rtsp_url:
                    continue

                camera_id = camera.get('cameraId') or str(camera.get('_id'))
                camera_type = camera.get('type', 'rtsp')
                ai_enabled = camera.get('aiEnabled', True)

                if not ai_enabled:
                    logger.info(f"Skipping camera {camera_id} (AI disabled)")
                    continue

                logger.info(f"📹 Starting camera: {camera.get('name', camera_id)} (type: {camera_type})")

                # Set status to connecting
                await update_camera_status(camera_id, "connecting")

                # Handle webcam type - use go2rtc stream
                stream_url = rtsp_url
                if camera_type == 'webcam':
                    from services.streaming.webcam_reader import get_best_webcam_settings
                    device_index = int(rtsp_url) if rtsp_url.isdigit() else 0

                    # Get best settings for this webcam
                    webcam_width, webcam_height, webcam_fps = get_best_webcam_settings(device_index)
                    logger.info(f"Webcam {camera_id}: Detected settings {webcam_width}x{webcam_height}@{webcam_fps}fps")

                    # Register with go2rtc
                    ffmpeg_source = f"ffmpeg:device?video={device_index}&video_size={webcam_width}x{webcam_height}&framerate={webcam_fps}#video=h264"
                    GO2RTC_URL = os.getenv("GO2RTC_URL", "http://localhost:1984")
                    GO2RTC_RTSP_PORT = os.getenv("GO2RTC_RTSP_PORT", "8554")

                    try:
                        async with httpx.AsyncClient(timeout=5.0) as go2rtc_client:
                            response = await go2rtc_client.put(
                                f"{GO2RTC_URL}/api/streams",
                                params={"name": camera_id, "src": ffmpeg_source}
                            )
                            if response.status_code in [200, 201]:
                                stream_url = f"rtsp://localhost:{GO2RTC_RTSP_PORT}/{camera_id}"
                                logger.info(f"Webcam {camera_id}: Registered with go2rtc -> {stream_url}")
                            else:
                                logger.warning(f"Webcam {camera_id}: go2rtc registration failed: {response.status_code}")
                                await update_camera_status(camera_id, "error", "go2rtc registration failed")
                                continue
                    except Exception as e:
                        logger.warning(f"Webcam {camera_id}: go2rtc error: {e}")
                        await update_camera_status(camera_id, "error", str(e))
                        continue

                config = RTSPConfig(
                    width=1280,
                    height=720,
                    fps=TARGET_FPS,
                    tcp_transport=True
                )

                try:
                    rtsp_manager.add_camera(camera_id, stream_url, config)
                    # Wait a bit for connection to establish
                    await asyncio.sleep(1.0)
                    # Check if we got a frame
                    frame = rtsp_manager.get_frame(camera_id)
                    if frame is not None:
                        await update_camera_status(camera_id, "online")
                        logger.info(f"✅ Camera {camera_id} online")
                    else:
                        await update_camera_status(camera_id, "error", "No frames received")
                        logger.warning(f"⚠️ Camera {camera_id} - no frames yet")
                except Exception as e:
                    await update_camera_status(camera_id, "error", str(e))
                    logger.error(f"❌ Failed to start camera {camera_id}: {e}")

            logger.info(f"✅ Loaded {len(rtsp_manager.get_active_cameras())} cameras")

    except Exception as e:
        logger.error(f"Error loading cameras: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down...")

    # Stop detection loop
    detection_loop = get_detection_loop()
    if detection_loop:
        await detection_loop.stop()

    # Stop RTSP readers
    rtsp_manager = get_rtsp_manager()
    rtsp_manager.stop_all()

    # Stop FFmpeg streams
    ffmpeg_manager.stop_all()

    # Stop radio service
    await stop_radio_service()

    # Stop recording manager
    recording_manager = get_recording_manager()
    if recording_manager:
        await recording_manager.stop()

    logger.info("Shutdown complete")


# ============== DETECTION LOOP ENDPOINTS ==============

@app.get("/detection/stats")
async def detection_stats():
    """Get detection loop statistics."""
    detection_loop = get_detection_loop()
    if not detection_loop:
        return {"error": "Detection loop not initialized"}

    rtsp_manager = get_rtsp_manager()

    return {
        "detection_loop": detection_loop.get_stats(),
        "rtsp_readers": rtsp_manager.get_all_stats()
    }


@app.get("/detection/gemini-debug")
async def gemini_debug_status():
    """Get Gemini analyzer debug information.

    Use this to diagnose why Gemini analysis might not be running.
    """
    detection_loop = get_detection_loop()

    result = {
        "gemini_configured": gemini.is_configured() if gemini else False,
        "gemini_api_calls": gemini.get_call_count() if gemini else 0,
        "gemini_model": str(gemini.model) if gemini and gemini.model else None,
    }

    if detection_loop:
        # Get analysis buffer stats
        if hasattr(detection_loop, 'analysis_buffer') and detection_loop.analysis_buffer:
            result["analysis_buffer"] = detection_loop.analysis_buffer.get_stats()
        else:
            result["analysis_buffer"] = "not_initialized"

        # Check if Gemini is being used in the loop
        result["loop_has_gemini"] = detection_loop.gemini is not None
        result["loop_gemini_configured"] = detection_loop.gemini.is_configured() if detection_loop.gemini else False

    return result


@app.get("/api/stats/realtime")
async def realtime_stats():
    """Get comprehensive real-time statistics for all AI components.

    Returns detailed performance metrics including:
    - YOLO detection timing and counts
    - ReID feature extraction timing
    - Tracker (BoT-SORT) performance
    - Recovery attempts and successes
    - Frame processing rates
    - Memory/CPU pressure indicators
    """
    detection_loop = get_detection_loop()
    rtsp_manager = get_rtsp_manager()
    recording_manager = get_recording_manager()

    # Get detection loop stats (includes timing)
    detection_stats = detection_loop.get_stats() if detection_loop else {}

    # Get ReID tracker stats
    reid_stats = tracker.get_stats() if tracker else {}

    # Get stable tracker stats
    stable = get_stable_tracker()
    stable_stats = stable.get_stats() if stable else {}

    # Get recording stats
    recording_stats = {}
    if recording_manager:
        recording_stats = recording_manager.get_stats()

    # Get RTSP reader stats
    rtsp_stats = rtsp_manager.get_all_stats() if rtsp_manager else {}

    # Get frame buffer stats
    from services.recording import get_frame_buffer
    frame_buffer = get_frame_buffer()
    buffer_stats = frame_buffer.get_stats() if frame_buffer else {}

    # Build comprehensive response
    timing = detection_stats.get("timing", {})
    config = detection_stats.get("config", {})
    pipeline_breakdown = detection_stats.get("pipeline_breakdown", {})
    bottlenecks = detection_stats.get("bottlenecks", [])

    return {
        "timestamp": time.time(),
        "uptime_seconds": detection_stats.get("uptime_seconds", 0),

        # Performance metrics (timing in ms)
        "performance": {
            # Main pipeline stages
            "yolo_ms": round(timing.get("yolo_ms", 0), 1),
            "yolo_postprocess_ms": round(timing.get("yolo_postprocess_ms", 0), 1),
            "reid_ms": round(timing.get("reid_extract_ms", 0), 1),  # renamed for clarity
            "reid_extract_ms": round(timing.get("reid_extract_ms", 0), 1),
            "reid_per_detection_ms": round(timing.get("reid_per_detection_ms", 0), 1),
            "tracker_ms": round(timing.get("tracker_ms", 0), 1),
            "recovery_ms": round(timing.get("recovery_ms", 0), 1),
            "weapon_ms": round(timing.get("weapon_ms", 0), 1),
            "drawing_ms": round(timing.get("drawing_ms", 0), 1),
            "total_frame_ms": round(timing.get("total_frame_ms", 0), 1),
            # Gemini analysis timing
            "gemini_vehicle_ms": round(timing.get("gemini_vehicle_ms", 0), 1),
            "gemini_person_ms": round(timing.get("gemini_person_ms", 0), 1),
            "image_enhance_ms": round(timing.get("image_enhance_ms", 0), 1),
            "cutout_gen_ms": round(timing.get("cutout_gen_ms", 0), 1),
            "backend_sync_ms": round(timing.get("backend_sync_ms", 0), 1),
            "frame_quality_ms": round(timing.get("frame_quality_ms", 0), 1),
            # FPS metrics
            "actual_fps": detection_stats.get("actual_fps", 0),
            "theoretical_fps": detection_stats.get("theoretical_fps", 0),
            "target_fps": config.get("detection_fps", 15),
        },

        # Pipeline breakdown (percentage of total time per stage)
        "pipeline_breakdown": pipeline_breakdown,
        "bottlenecks": bottlenecks,

        # Counters
        "counters": {
            "frames_processed": detection_stats.get("frames_processed", 0),
            "detections": detection_stats.get("detections", 0),
            "reid_extractions": detection_stats.get("reid_extractions", 0),
            "reid_recoveries": detection_stats.get("reid_recoveries", 0),
            "events_sent": detection_stats.get("events_sent", 0),
            "gemini_calls": gemini.get_call_count() if gemini else 0,
            "frames_dropped": detection_stats.get("frames_dropped_stale", 0),
            # Event-based processing stats
            "event_waits": detection_stats.get("event_waits", 0),
            "event_timeouts": detection_stats.get("event_timeouts", 0),
            "event_signals": detection_stats.get("event_signals", 0),
        },

        # Tracker stats
        "tracker": {
            "bot_sort": detection_stats.get("bot_sort", {}),
            "reid": reid_stats,
            "stable": stable_stats,
        },

        # Queue/buffer pressure
        "pressure": {
            "pending_frames": detection_stats.get("pending_frames", 0),
            "result_queue_size": detection_stats.get("result_queue_size", 0),
            "active_cameras": detection_stats.get("active_cameras", []),
        },

        # Configuration
        "config": config,

        # Recording stats
        "recording": recording_stats,

        # Buffer stats
        "frame_buffer": buffer_stats,

        # RTSP readers
        "rtsp_readers": rtsp_stats,

        # Frame selection stats
        "frame_selection": detection_stats.get("frame_selection", {}),

        # Image enhancement stats
        "image_enhancement": detection_stats.get("image_enhancement", {}),

        # ReID cache stats
        "reid_cache": detection_stats.get("reid_cache", {}),

        # Backend sync stats (rate limiting, retries, sync operations)
        "backend_sync": _get_backend_sync_stats(),
    }


def _get_backend_sync_stats() -> dict:
    """Get backend sync statistics safely."""
    try:
        from services import backend_sync
        return backend_sync.get_sync_stats()
    except Exception as e:
        logger.debug(f"Failed to get backend sync stats: {e}")
        return {}


@app.get("/detection/fps")
async def get_fps_config():
    """
    Get current FPS configuration.

    Returns:
        target_fps: Master FPS for RTSP readers (MUST match camera output!)
        detection_fps: How often to run AI detection (can be lower than target)
        stream_fps: Output stream FPS to frontend (can be lower than target)

    Example:
        Camera outputs 25 FPS → target_fps=25
        Run detection at 15 FPS → detection_fps=15 (saves CPU)
        Stream to UI at 15 FPS → stream_fps=15 (saves bandwidth)
    """
    return {
        "target_fps": TARGET_FPS,
        "detection_fps": DETECTION_FPS,
        "stream_fps": STREAM_FPS,
        "description": {
            "target_fps": "Master FPS setting - MUST match camera output FPS",
            "detection_fps": "AI detection processing FPS (can be lower to save CPU)",
            "stream_fps": "Output stream FPS to frontend (can be lower to save bandwidth)"
        }
    }


@app.post("/detection/fps")
async def update_fps_config(
    target_fps: Optional[int] = None,
    detection_fps: Optional[int] = None,
    stream_fps: Optional[int] = None
):
    """
    Update FPS configuration.

    Note: This updates the in-memory configuration. To persist changes,
    update the .env file and restart the service.

    Args:
        target_fps: Target FPS for RTSP readers and cameras (5-30)
        detection_fps: Detection processing FPS (5-30)
        stream_fps: Stream output FPS (5-30)
    """
    global TARGET_FPS, DETECTION_FPS, STREAM_FPS

    updated = {}

    if target_fps is not None:
        if not 5 <= target_fps <= 30:
            raise HTTPException(400, "target_fps must be between 5 and 30")
        TARGET_FPS = target_fps
        updated["target_fps"] = target_fps

    if detection_fps is not None:
        if not 5 <= detection_fps <= 30:
            raise HTTPException(400, "detection_fps must be between 5 and 30")
        DETECTION_FPS = detection_fps
        updated["detection_fps"] = detection_fps

    if stream_fps is not None:
        if not 5 <= stream_fps <= 30:
            raise HTTPException(400, "stream_fps must be between 5 and 30")
        STREAM_FPS = stream_fps
        updated["stream_fps"] = stream_fps

    if not updated:
        raise HTTPException(400, "No FPS values provided to update")

    logger.info(f"FPS configuration updated: {updated}")
    return {
        "status": "updated",
        "changes": updated,
        "current": {
            "target_fps": TARGET_FPS,
            "detection_fps": DETECTION_FPS,
            "stream_fps": STREAM_FPS
        },
        "note": "Changes are in-memory only. Update .env file to persist across restarts."
    }


@app.get("/detection/active")
async def get_active_cameras():
    """Get list of cameras with active detection streams.

    Used by backend to sync camera status on startup.
    """
    rtsp_manager = get_rtsp_manager()
    active_cameras = rtsp_manager.get_active_cameras()
    return {
        "active_cameras": active_cameras,
        "count": len(active_cameras)
    }


@app.post("/detection/reload")
async def reload_cameras():
    """Reload cameras from backend."""
    await auto_load_cameras()
    return {"status": "reloaded", "cameras": get_rtsp_manager().get_active_cameras()}


@app.post("/detection/cleanup")
async def cleanup_cameras():
    """
    Stop and cleanup all camera streams and detection loops.

    Use this to stop zombie processes when cameras are deleted from UI
    or when experiencing reconnect loops.
    """
    logger.info("Cleaning up all camera streams...")

    # Stop all RTSP readers
    rtsp_manager = get_rtsp_manager()
    active_cameras = rtsp_manager.get_active_cameras()
    for camera_id in active_cameras:
        try:
            rtsp_manager.remove_camera(camera_id)
            logger.info(f"Stopped RTSP reader for {camera_id}")
        except Exception as e:
            logger.error(f"Error stopping {camera_id}: {e}")

    # Stop FFmpeg manager streams
    try:
        ffmpeg_manager = get_ffmpeg_manager()
        ffmpeg_manager.stop_all()
        logger.info("Stopped all FFmpeg streams")
    except Exception as e:
        logger.error(f"Error stopping FFmpeg manager: {e}")

    # Stop detection loop
    try:
        detection_loop = get_detection_loop()
        if detection_loop:
            await detection_loop.stop()
            logger.info("Stopped detection loop")
    except Exception as e:
        logger.error(f"Error stopping detection loop: {e}")

    return {
        "status": "cleaned",
        "stopped_cameras": active_cameras,
        "message": "All camera streams and processes stopped"
    }


@app.delete("/detection/camera/{camera_id}")
async def stop_camera(camera_id: str):
    """
    Stop and remove a specific camera stream.

    Use this when deleting a camera from the UI to ensure
    all associated processes are properly terminated.
    """
    logger.info(f"Stopping camera: {camera_id}")

    # Stop RTSP reader
    try:
        rtsp_manager = get_rtsp_manager()
        rtsp_manager.remove_camera(camera_id)
        logger.info(f"Stopped RTSP reader for {camera_id}")
    except Exception as e:
        logger.warning(f"RTSP reader not found for {camera_id}: {e}")

    # Stop FFmpeg stream
    try:
        ffmpeg_manager = get_ffmpeg_manager()
        stream = ffmpeg_manager.get_stream(camera_id)
        if stream:
            stream.stop()
            logger.info(f"Stopped FFmpeg stream for {camera_id}")
    except Exception as e:
        logger.warning(f"FFmpeg stream not found for {camera_id}: {e}")

    # CRITICAL: Clear tracks for this camera to prevent ghost tracks and stale events
    try:
        from services.detection import get_bot_sort_tracker
        tracker = get_bot_sort_tracker()
        if tracker:
            tracker.clear_camera_tracks(camera_id)
            logger.info(f"Cleared tracks for {camera_id}")
    except Exception as e:
        logger.warning(f"Error clearing tracks for {camera_id}: {e}")

    # Stop any active recordings for this camera
    try:
        from services.recording import get_recording_manager
        recording_manager = get_recording_manager()
        if recording_manager and recording_manager.is_recording(camera_id):
            recording_manager.stop_recording(camera_id)
            logger.info(f"Stopped recording for {camera_id}")
    except Exception as e:
        logger.warning(f"Recording stop error for {camera_id}: {e}")

    # Clear scenario hooks state for this camera (prevents armed count carryover)
    try:
        from services.scenario.scenario_hooks import get_scenario_hooks
        hooks = get_scenario_hooks()
        hooks.clear_camera_state(camera_id)
        logger.info(f"Cleared scenario state for {camera_id}")
    except Exception as e:
        logger.debug(f"Scenario hooks clear error for {camera_id}: {e}")

    return {
        "status": "stopped",
        "camera_id": camera_id,
        "message": f"Camera {camera_id} stopped and removed"
    }


@app.post("/detection/start/{camera_id}")
async def start_camera_detection(
    camera_id: str,
    rtsp_url: Optional[str] = None,
    camera_type: Optional[str] = None,
    use_go2rtc: Optional[bool] = True
):
    """Start detection for a specific camera.

    Args:
        camera_id: Camera identifier
        rtsp_url: Optional RTSP URL, video file path, or webcam device index. If not provided, fetches from backend.
        camera_type: Camera type - "rtsp", "file", "webcam", or "simulator". If not provided, fetches from backend.
        use_go2rtc: If True and go2rtc is available, use go2rtc's RTSP re-stream (default: True)
                    This prevents multiple connections to the camera.

    Examples:
        POST /detection/start/cam-1?rtsp_url=assets/test2.mp4
        POST /detection/start/cam-1?rtsp_url=rtsp://192.168.1.100/stream
        POST /detection/start/webcam-1?camera_type=webcam&rtsp_url=0
    """
    import httpx

    GO2RTC_URL = os.getenv("GO2RTC_URL", "http://localhost:1984")
    GO2RTC_RTSP_PORT = os.getenv("GO2RTC_RTSP_PORT", "8554")

    async def check_go2rtc_available(max_retries: int = 3, retry_delay: float = 1.0):
        """Check if go2rtc is running with retry logic.

        Args:
            max_retries: Number of retries if connection fails
            retry_delay: Delay between retries in seconds

        Returns:
            True if go2rtc is available, False otherwise
        """
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    response = await client.get(f"{GO2RTC_URL}/api")
                    if response.status_code == 200:
                        return True
                    elif response.status_code in [500, 502, 503, 504]:
                        # Server error - might be starting up, retry
                        logger.debug(f"go2rtc returned {response.status_code}, attempt {attempt + 1}/{max_retries}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_delay)
                            continue
                    else:
                        # Other error (400, 404, etc.) - don't retry
                        logger.debug(f"go2rtc returned {response.status_code}")
                        return False
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                # Connection failed - might be starting up, retry
                logger.debug(f"go2rtc connection failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    continue
            except Exception as e:
                logger.debug(f"go2rtc check error: {e}")
                return False
        return False

    async def register_with_go2rtc(cam_id: str, original_url: str, max_retries: int = 2):
        """Register stream with go2rtc and return the re-stream URL.

        Args:
            cam_id: Camera ID to register
            original_url: Original RTSP URL or FFmpeg source
            max_retries: Number of retries on failure

        Returns:
            go2rtc re-stream URL or None if registration failed
        """
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    # Add TCP transport for RTSP URLs
                    src_url = original_url
                    if original_url.startswith("rtsp://") and "#" not in original_url:
                        src_url = f"{original_url}#transport=tcp"
                    elif original_url.startswith("rtsp://") and "transport=" not in original_url:
                        src_url = f"{original_url}&transport=tcp"

                    # Add stream to go2rtc
                    response = await client.put(
                        f"{GO2RTC_URL}/api/streams",
                        params={"name": cam_id, "src": src_url}
                    )

                    if response.status_code in [200, 201]:
                        # Return go2rtc's RTSP re-stream URL
                        return f"rtsp://localhost:{GO2RTC_RTSP_PORT}/{cam_id}"
                    elif response.status_code in [500, 502, 503, 504]:
                        # Server error - retry
                        logger.warning(f"go2rtc registration returned {response.status_code} for {cam_id}, attempt {attempt + 1}/{max_retries}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(1.0)
                            continue
                    else:
                        # Log response body to understand the error
                        error_body = response.text[:500] if response.text else "No response body"
                        logger.warning(f"go2rtc registration failed for {cam_id}: {response.status_code} - {error_body}")
                        return None
            except Exception as e:
                logger.warning(f"Failed to register {cam_id} with go2rtc (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1.0)
                    continue
        return None

    try:
        rtsp_manager = get_rtsp_manager()
        detection_loop = get_detection_loop()

        # Ensure the frame callback is registered (in case it wasn't during startup)
        if detection_loop and detection_loop.on_frame not in rtsp_manager._frame_callbacks:
            rtsp_manager.add_frame_callback(detection_loop.on_frame)
            logger.info(f"Re-registered detection loop frame callback")

        # If rtsp_url provided directly, use it; otherwise fetch from backend
        original_url = rtsp_url
        resolved_type = camera_type

        if not rtsp_url or not camera_type:
            # Fetch from backend
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{BACKEND_URL}/api/cameras/{camera_id}")
                if response.status_code != 200:
                    raise HTTPException(404, f"Camera {camera_id} not found in backend. Use ?rtsp_url= to provide URL directly.")

                camera = response.json()
                if not original_url:
                    original_url = camera.get('rtspUrl', '')
                if not resolved_type:
                    resolved_type = camera.get('type', 'rtsp')

        # Default to rtsp type
        resolved_type = resolved_type or 'rtsp'

        # Handle webcam type - register with go2rtc using FFmpeg device capture
        if resolved_type == 'webcam':
            from services.streaming.webcam_reader import get_best_webcam_settings

            # Parse device index from original_url (default "0")
            device_index = int(original_url) if original_url and original_url.isdigit() else 0

            # Get best settings for this webcam (auto-detect resolution and framerate)
            webcam_width, webcam_height, webcam_fps = get_best_webcam_settings(device_index)
            logger.info(f"Webcam {camera_id}: Detected best settings: {webcam_width}x{webcam_height}@{webcam_fps}fps")

            # go2rtc device source format: ffmpeg:device?video={index}&video_size={w}x{h}&framerate={fps}#video=h264
            ffmpeg_source = f"ffmpeg:device?video={device_index}&video_size={webcam_width}x{webcam_height}&framerate={webcam_fps}#video=h264"
            logger.info(f"Webcam {camera_id}: FFmpeg source for go2rtc: {ffmpeg_source}")

            # Register webcam with go2rtc
            stream_url = None
            using_go2rtc = False

            if use_go2rtc and await check_go2rtc_available():
                go2rtc_url = await register_with_go2rtc(camera_id, ffmpeg_source)
                if go2rtc_url:
                    stream_url = go2rtc_url
                    using_go2rtc = True
                    logger.info(f"Webcam {camera_id}: Using go2rtc re-stream: {go2rtc_url}")

            if not stream_url:
                raise HTTPException(500, f"Failed to register webcam with go2rtc. Ensure go2rtc is running.")

            # Use the go2rtc RTSP stream (same flow as regular cameras)
            config = RTSPConfig(width=1280, height=720, fps=TARGET_FPS, tcp_transport=True)
            rtsp_manager.add_camera(camera_id, stream_url, config)

            logger.info(f"Started webcam {camera_id} (device {device_index}) via go2rtc")
            return {
                "status": "started",
                "camera_id": camera_id,
                "camera_type": "webcam",
                "device_index": device_index,
                "rtsp_url": stream_url,
                "ffmpeg_source": ffmpeg_source,
                "using_go2rtc": using_go2rtc
            }

        # For non-webcam types, require a URL
        if not original_url:
            raise HTTPException(400, f"Camera {camera_id} has no URL configured")

        # Determine the URL to use for AI service
        stream_url = original_url
        using_go2rtc = False

        # For RTSP streams (not local files), try to use go2rtc if available
        if use_go2rtc and original_url.startswith("rtsp://"):
            if await check_go2rtc_available():
                go2rtc_url = await register_with_go2rtc(camera_id, original_url)
                if go2rtc_url:
                    stream_url = go2rtc_url
                    using_go2rtc = True
                    logger.info(f"Using go2rtc re-stream for {camera_id}: {go2rtc_url}")

        config = RTSPConfig(width=1280, height=720, fps=TARGET_FPS, tcp_transport=True)
        rtsp_manager.add_camera(camera_id, stream_url, config)

        logger.info(f"Started camera {camera_id} ({resolved_type}) with URL: {stream_url}")
        return {
            "status": "started",
            "camera_id": camera_id,
            "camera_type": resolved_type,
            "rtsp_url": stream_url,
            "original_url": original_url,
            "using_go2rtc": using_go2rtc
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/detection/stop/{camera_id}")
async def stop_camera_detection(camera_id: str):
    """Stop detection for a specific camera."""
    get_rtsp_manager().remove_camera(camera_id)

    # Clear tracks for this camera to prevent stale events
    try:
        from services.detection import get_bot_sort_tracker
        tracker = get_bot_sort_tracker()
        if tracker:
            tracker.clear_camera_tracks(camera_id)
    except Exception as e:
        logger.warning(f"Error clearing tracks: {e}")

    # Stop any active recordings for this camera
    from services.recording import get_recording_manager
    recording_manager = get_recording_manager()
    if recording_manager and recording_manager.is_recording(camera_id):
        recording_manager.stop_recording(camera_id)
        logger.info(f"Stopped recording for disconnected camera {camera_id}")

    return {"status": "stopped", "camera_id": camera_id}


@app.get("/detection/webcams")
async def list_available_webcams():
    """List available webcam devices.

    Returns a list of webcam devices that can be used with camera_type=webcam.
    """
    try:
        from services.streaming.webcam_reader import list_available_webcams
        devices = list_available_webcams()
        return {
            "status": "success",
            "devices": devices,
            "count": len(devices)
        }
    except Exception as e:
        logger.error(f"Error listing webcams: {e}")
        return {
            "status": "error",
            "error": str(e),
            "devices": []
        }


@app.get("/detection/config")
async def get_detection_config():
    """Get current detection configuration (FPS, ReID recovery, confidence thresholds, etc.)."""
    detection_loop = get_detection_loop()
    if not detection_loop:
        return {"error": "Detection loop not initialized"}

    return {
        "detection_fps": detection_loop.config.detection_fps,
        "stream_fps": detection_loop.config.stream_fps,
        "reader_fps": detection_loop.config.reader_fps,
        "recording_fps": detection_loop.config.recording_fps,
        "use_reid_recovery": detection_loop.config.use_reid_recovery,
        "use_bot_sort": detection_loop.config.use_bot_sort,
        "draw_bboxes": detection_loop.config.draw_bboxes,
        "send_events": detection_loop.config.send_events,
        "yolo_confidence": detection_loop.config.yolo_confidence,
        "weapon_confidence": detection_loop.config.weapon_confidence,
        "recovery_confidence": detection_loop.config.recovery_confidence,
    }


@app.post("/detection/config/fps")
async def set_detection_fps(
    detection_fps: Optional[int] = Query(None, description="Detection FPS (1-30)", ge=1, le=30),
    stream_fps: Optional[int] = Query(None, description="Stream FPS (1-30)", ge=1, le=30),
    reader_fps: Optional[int] = Query(None, description="Reader FPS (1-30)", ge=1, le=30),
    recording_fps: Optional[int] = Query(None, description="Recording FPS (1-30)", ge=1, le=30)
):
    """Change FPS settings dynamically.

    Note: Changes take effect immediately. Frontend streams may need to reconnect
    with new FPS parameter to see the change.

    Args:
        detection_fps: How often to run YOLO detection (affects CPU/GPU usage)
        stream_fps: FPS for streaming annotated video to frontend
        reader_fps: FPS for reading from RTSP camera (affects network/decoding)
        recording_fps: FPS for saved recordings
    """
    detection_loop = get_detection_loop()
    if not detection_loop:
        raise HTTPException(503, "Detection loop not running")

    if detection_fps is None and stream_fps is None and reader_fps is None and recording_fps is None:
        raise HTTPException(400, "Must provide at least one FPS value")

    detection_loop.set_fps(detection_fps, stream_fps, reader_fps, recording_fps)

    return {
        "message": "FPS settings updated",
        "detection_fps": detection_loop.config.detection_fps,
        "stream_fps": detection_loop.config.stream_fps,
        "reader_fps": detection_loop.config.reader_fps,
        "recording_fps": detection_loop.config.recording_fps,
        "note": "Frontend streams should reconnect to see changes"
    }


@app.post("/detection/config/demo-mode")
async def set_demo_mode(enabled: bool = Query(..., description="Enable demo mode (slower FPS for longer video duration)")):
    """Toggle demo mode on/off.

    Demo mode uses lower FPS values to make videos last longer,
    which is useful for demos and presentations.

    Demo mode settings:
    - detection_fps: 8 (instead of 20)
    - stream_fps: 10 (instead of 20)
    - reader_fps: 10 (instead of 25)
    - recording_fps: 10 (instead of 15)

    Production mode settings:
    - detection_fps: 20
    - stream_fps: 20
    - reader_fps: 25
    - recording_fps: 15
    """
    detection_loop = get_detection_loop()
    if not detection_loop:
        raise HTTPException(503, "Detection loop not running")

    if enabled:
        # Demo mode - slower FPS for longer video duration
        detection_loop.set_fps(
            detection_fps=8,
            stream_fps=10,
            reader_fps=10,
            recording_fps=10
        )
        mode = "demo"
    else:
        # Production mode - faster FPS
        detection_loop.set_fps(
            detection_fps=20,
            stream_fps=20,
            reader_fps=25,
            recording_fps=15
        )
        mode = "production"

    return {
        "message": f"{mode.capitalize()} mode {'enabled' if enabled else 'disabled'}",
        "mode": mode,
        "detection_fps": detection_loop.config.detection_fps,
        "stream_fps": detection_loop.config.stream_fps,
        "reader_fps": detection_loop.config.reader_fps,
        "recording_fps": detection_loop.config.recording_fps,
        "note": "Video processing speed adjusted. Lower FPS = longer video duration."
    }


@app.post("/detection/config/reid-recovery")
async def set_reid_recovery(enabled: bool = Query(..., description="Enable/disable ReID recovery")):
    """Toggle ReID-based detection recovery on/off.

    ReID recovery helps maintain tracking when objects are temporarily occluded,
    but may impact performance.

    Args:
        enabled: True to enable, False to disable
    """
    detection_loop = get_detection_loop()
    if not detection_loop:
        raise HTTPException(503, "Detection loop not running")

    detection_loop.set_reid_recovery(enabled)

    return {
        "message": f"ReID recovery {'enabled' if enabled else 'disabled'}",
        "use_reid_recovery": detection_loop.config.use_reid_recovery
    }


@app.post("/detection/config/confidence")
async def set_confidence_thresholds(
    yolo_confidence: Optional[float] = Query(None, description="YOLO confidence (0.0-1.0)", ge=0.0, le=1.0),
    weapon_confidence: Optional[float] = Query(None, description="Weapon confidence (0.0-1.0)", ge=0.0, le=1.0),
    recovery_confidence: Optional[float] = Query(None, description="Recovery confidence (0.0-1.0)", ge=0.0, le=1.0)
):
    """Change confidence thresholds dynamically.

    Lower thresholds = more detections (more false positives)
    Higher thresholds = fewer detections (more false negatives)

    Recommended ranges:
    - YOLO: 0.25-0.50 (default: 0.35)
    - Weapon: 0.30-0.60 (default: 0.40)
    - Recovery: 0.15-0.30 (default: 0.20)

    Args:
        yolo_confidence: Main YOLO detection confidence
        weapon_confidence: Weapon detection confidence
        recovery_confidence: Low-confidence recovery pass
    """
    detection_loop = get_detection_loop()
    if not detection_loop:
        raise HTTPException(503, "Detection loop not running")

    if yolo_confidence is None and weapon_confidence is None and recovery_confidence is None:
        raise HTTPException(400, "Must provide at least one confidence value")

    detection_loop.set_confidence(yolo_confidence, weapon_confidence, recovery_confidence)

    return {
        "message": "Confidence thresholds updated",
        "yolo_confidence": detection_loop.config.yolo_confidence,
        "weapon_confidence": detection_loop.config.weapon_confidence,
        "recovery_confidence": detection_loop.config.recovery_confidence,
        "note": "Changes take effect immediately on next frame"
    }


@app.post("/detection/reset-gemini")
async def reset_gemini_state(
    reset_tracker: bool = Query(True, description="Also reset BoT-SORT tracker (clears all track IDs)")
):
    """Reset Gemini analysis state to force re-analysis of all tracks.

    Use this when:
    - Starting a new session and want fresh Gemini analysis
    - AI Calls shows 0 but you expect analysis to happen
    - Tracks from previous session are preventing new analysis
    - You want to force re-analyze all currently visible objects

    This clears:
    - Gemini cooldown timers (allows immediate re-analysis)
    - Gemini API call counter (resets stats)
    - Alert and event cooldowns
    - Optionally: All BoT-SORT tracks (new track IDs assigned)

    Args:
        reset_tracker: If True (default), also reset tracker so objects get new IDs
    """
    detection_loop = get_detection_loop()
    if not detection_loop:
        raise HTTPException(503, "Detection loop not running")

    result = detection_loop.reset_gemini_state(reset_tracker=reset_tracker)

    return {
        "message": "Gemini analysis state reset - new tracks will be analyzed",
        "details": result,
        "note": "Objects will now be re-analyzed by Gemini as they appear"
    }


# ============== RULES ENDPOINTS ==============

@app.post("/api/rules/reload")
async def reload_event_rules():
    """Reload event rules from backend.

    Called by the backend when rules are created/updated/deleted.
    This avoids the need to poll for rule changes.
    """
    from services.rules import get_rule_engine

    try:
        engine = get_rule_engine()
        await engine.reload_rules()
        return {
            "status": "reloaded",
            "rules_count": len(engine.rules)
        }
    except Exception as e:
        logger.error(f"Failed to reload rules: {e}")
        raise HTTPException(500, str(e))


# ============== RADIO ENDPOINTS ==============

@app.get("/radio/stats")
async def radio_stats():
    """Get radio service and transcriber statistics."""
    service = get_radio_service()
    service_stats = service.get_stats() if service else {"error": "Radio service not running", "running": False}

    # Add global transcriber stats
    transcriber_stats = get_transcriber_stats()

    return {
        **service_stats,
        "transcriber_manager": transcriber_stats
    }


@app.get("/radio/transcribers")
async def get_transcriber_status():
    """Get status of available transcribers.

    Returns which transcribers are enabled/disabled and configured.
    Frontend can use this to show/hide transcription tabs.
    """
    gemini = get_gemini_transcriber()
    whisper = get_whisper_transcriber()
    stats = get_transcriber_stats()

    return {
        "gemini": {
            "enabled": not stats.get("gemini_disabled", False),
            "configured": gemini.is_configured() if gemini else False,
            "available": gemini is not None and gemini.is_configured(),
        },
        "whisper": {
            "enabled": not stats.get("whisper_disabled", False),
            "configured": whisper.is_configured() if whisper else False,
            "available": whisper is not None and whisper.is_configured(),
        },
        # Convenience: list of available transcribers for UI tabs
        "available_transcribers": [
            name for name, available in [
                ("gemini", gemini is not None and gemini.is_configured()),
                ("whisper", whisper is not None and whisper.is_configured()),
            ] if available
        ]
    }


@app.post("/radio/transcribe-file")
async def transcribe_audio_file(file: UploadFile = File(...)):
    """Transcribe an uploaded audio file (WAV format) using BOTH transcribers.

    This endpoint accepts a WAV file upload and returns transcriptions from
    both Gemini (cloud) and Whisper (local) transcribers.
    Both transcribers run in PARALLEL and emit socket events INDEPENDENTLY
    as each finishes (so UI updates immediately without waiting for both).

    Uses GLOBAL transcriber instances (shared with live radio service).
    Whisper uses a semaphore to prevent concurrent CPU-intensive processing.

    Args:
        file: WAV audio file to transcribe

    Returns:
        JSON with transcriptions from both transcribers
    """
    import tempfile
    import os as os_module
    import httpx
    import time

    # Backend URL for sending transcription events
    backend_url = os.getenv("BACKEND_URL", "http://localhost:3000")

    # Validate file type
    if not file.filename.lower().endswith('.wav'):
        raise HTTPException(400, "Only WAV files are supported. Please upload a .wav file.")

    # Check file size (limit to 25MB)
    MAX_SIZE = 25 * 1024 * 1024  # 25MB
    contents = await file.read()
    if len(contents) > MAX_SIZE:
        raise HTTPException(400, f"File too large. Maximum size is 25MB, got {len(contents) / 1024 / 1024:.1f}MB")

    # Save to temp file using async I/O to avoid blocking the event loop
    def write_temp_file():
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(contents)
            return tmp.name

    tmp_path = await asyncio.to_thread(write_temp_file)

    try:
        logger.info(f"Transcribing uploaded file: {file.filename} ({len(contents)} bytes)")

        # Get GLOBAL transcribers (shared with live radio service)
        # Note: Either or both may be None if disabled via environment variables
        gemini_transcriber = get_gemini_transcriber()
        whisper_transcriber = get_whisper_transcriber()
        whisper_semaphore = get_whisper_semaphore()

        # Require at least one transcriber to be available
        if not gemini_transcriber and not whisper_transcriber:
            raise HTTPException(500, "No transcribers available. Both are disabled or not initialized.")

        results = {
            "success": True,
            "filename": file.filename,
            "gemini": None,
            "whisper": None
        }

        # Helper to send transcription to backend (emits socket event immediately)
        async def send_to_backend(transcriber_type: str, data: dict):
            """Send transcription result to backend to emit socket event."""
            try:
                endpoint = f"{backend_url}/api/radio/transcription/{transcriber_type}"
                logger.info(f"[{transcriber_type.upper()}] Sending to endpoint: {endpoint}")
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.post(endpoint, json=data)
                    if response.status_code == 200:
                        logger.info(f"[{transcriber_type.upper()}] SUCCESS - sent to {endpoint}")
                    else:
                        logger.warning(f"[{transcriber_type.upper()}] Backend response: {response.status_code} - {response.text}")
            except httpx.TimeoutException:
                logger.error(f"[{transcriber_type.capitalize()}] Backend request timed out")
            except httpx.ConnectError as e:
                logger.error(f"[{transcriber_type.capitalize()}] Failed to connect to backend: {e}")
            except Exception as e:
                logger.error(f"[{transcriber_type.capitalize()}] Failed to send to backend: {type(e).__name__}: {e}")

        # Define async tasks for parallel execution
        async def transcribe_gemini():
            if not gemini_transcriber:
                return {"error": "Gemini transcriber disabled", "disabled": True}
            if not gemini_transcriber.is_configured():
                return {"error": "Gemini not configured (missing GEMINI_API_KEY)"}
            try:
                start_time = time.time()
                result = await gemini_transcriber.transcribe_file(tmp_path)
                processing_time_ms = (time.time() - start_time) * 1000

                if result and result.text:
                    result_data = {
                        "text": result.text,
                        "duration_seconds": result.duration_seconds,
                        "timestamp": result.timestamp.isoformat(),
                        "is_command": result.is_command,
                        "command_type": result.command_type,
                        "source": f"file:{file.filename}",
                        "processing_time_ms": processing_time_ms
                    }
                    logger.info(f"[Gemini] File transcription complete in {processing_time_ms:.0f}ms: '{result.text[:100] if result.text else '(empty)'}...'")

                    # Record statistics
                    record_transcription_stats("gemini", "file", processing_time_ms, success=True)

                    # Send to backend IMMEDIATELY (emits socket event)
                    await send_to_backend("gemini", result_data)

                    return result_data
                return {"error": "No transcription result"}
            except Exception as e:
                record_transcription_stats("gemini", "file", 0, success=False)
                logger.error(f"[Gemini] File transcription error: {type(e).__name__}: {e}")
                return {"error": str(e)}

        async def transcribe_whisper():
            if not whisper_transcriber:
                return {"error": "Whisper transcriber disabled", "disabled": True}
            if not whisper_transcriber.is_configured():
                return {"error": "Whisper not configured (model not found)"}

            # Use semaphore to prevent concurrent CPU-intensive processing
            if whisper_semaphore:
                if whisper_semaphore.locked():
                    logger.info("[Whisper] Waiting for semaphore (another transcription in progress)...")
                    record_whisper_queue_wait()

                async with whisper_semaphore:
                    return await _do_whisper_transcription()
            else:
                return await _do_whisper_transcription()

        async def _do_whisper_transcription():
            try:
                start_time = time.time()
                result = await whisper_transcriber.transcribe_file(tmp_path)
                processing_time_ms = (time.time() - start_time) * 1000

                if result and result.text:
                    result_data = {
                        "text": result.text,
                        "duration_seconds": result.duration_seconds,
                        "timestamp": result.timestamp.isoformat(),
                        "is_command": result.is_command,
                        "command_type": result.command_type,
                        "segments": result.segments,
                        "source": f"file:{file.filename}",
                        "processing_time_ms": processing_time_ms
                    }
                    logger.info(f"[Whisper] File transcription complete in {processing_time_ms:.0f}ms: '{result.text[:100] if result.text else '(empty)'}...'")

                    # Record statistics
                    record_transcription_stats("whisper", "file", processing_time_ms, success=True)

                    # Send to backend IMMEDIATELY (emits socket event)
                    await send_to_backend("whisper", result_data)

                    return result_data
                return {"error": "No transcription result"}
            except Exception as e:
                record_transcription_stats("whisper", "file", 0, success=False)
                logger.error(f"[Whisper] File transcription error: {type(e).__name__}: {e}")
                return {"error": str(e)}

        # Run BOTH transcribers in PARALLEL - each emits socket event when done
        logger.info("Running Gemini and Whisper transcription in parallel (independent socket events)...")
        gemini_result, whisper_result = await asyncio.gather(
            transcribe_gemini(),
            transcribe_whisper(),
            return_exceptions=True
        )

        # Handle results (could be exceptions if something went wrong)
        if isinstance(gemini_result, Exception):
            results["gemini"] = {"error": str(gemini_result)}
        else:
            results["gemini"] = gemini_result

        if isinstance(whisper_result, Exception):
            results["whisper"] = {"error": str(whisper_result)}
        else:
            results["whisper"] = whisper_result

        # Check if at least one transcriber worked
        gemini_ok = results["gemini"] and "text" in results["gemini"]
        whisper_ok = results["whisper"] and "text" in results["whisper"]

        if not gemini_ok and not whisper_ok:
            raise HTTPException(500, "Both transcribers failed. Check logs for details.")

        return results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File transcription error: {e}", exc_info=True)
        raise HTTPException(500, f"Transcription error: {str(e)}")
    finally:
        # Clean up temp file in background thread to avoid blocking
        def cleanup():
            try:
                os_module.unlink(tmp_path)
            except:
                pass
        asyncio.get_event_loop().run_in_executor(None, cleanup)


# ============== FFMPEG STREAMING ENDPOINTS ==============

@app.get("/recordings/{filename}")
async def get_recording(filename: str):
    """Serve recorded video files."""
    from pathlib import Path

    recordings_dir = os.getenv("RECORDINGS_DIR", "/tmp/hamal_recordings")
    video_path = Path(recordings_dir) / filename

    if not video_path.exists():
        raise HTTPException(404, "Recording not found")

    # Security: ensure filename doesn't escape the recordings directory
    try:
        video_path.resolve().relative_to(Path(recordings_dir).resolve())
    except ValueError:
        raise HTTPException(403, "Invalid path")

    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=filename
    )


@app.get("/recordings")
async def list_recordings():
    """List all recorded videos."""
    from pathlib import Path

    recordings_dir = os.getenv("RECORDINGS_DIR", "/tmp/hamal_recordings")
    recordings_path = Path(recordings_dir)

    if not recordings_path.exists():
        return {"recordings": []}

    recordings = []
    for video_file in sorted(recordings_path.glob("*.mp4"), reverse=True):
        stat = video_file.stat()
        recordings.append({
            "filename": video_file.name,
            "url": f"/recordings/{video_file.name}",
            "size_bytes": stat.st_size,
            "created_at": stat.st_ctime
        })

    return {"recordings": recordings}


@app.get("/recording/stats")
async def recording_stats():
    """Get recording manager statistics."""
    manager = get_recording_manager()
    if not manager:
        return {"error": "Recording manager not initialized"}

    from services.recording import get_frame_buffer
    frame_buffer = get_frame_buffer()

    return {
        "recording_manager": manager.get_stats(),
        "frame_buffer": frame_buffer.get_stats() if frame_buffer else None
    }


@app.get("/ffmpeg/stats")
async def ffmpeg_stats():
    """Get FFmpeg stream statistics."""
    return {
        "streams": ffmpeg_manager.get_all_stats(),
        "active_count": len(ffmpeg_manager.get_active_cameras())
    }


# ============== SCENARIO RULES ENDPOINTS ==============

@app.get("/scenario-rules")
async def get_scenario_rules():
    """Get all defined scenario rules."""
    try:
        from services.scenario import get_scenario_rule_engine
        engine = get_scenario_rule_engine()
        return {
            "scenarios": engine.get_all_scenarios(),
            "activeScenario": engine.get_active_scenario()
        }
    except Exception as e:
        logger.error(f"Failed to get scenario rules: {e}")
        return {"scenarios": [], "activeScenario": None, "error": str(e)}


@app.get("/scenario-rules/{scenario_id}")
async def get_scenario_rule(scenario_id: str):
    """Get a specific scenario rule by ID."""
    try:
        from services.scenario import get_scenario_rule_engine
        engine = get_scenario_rule_engine()
        scenario = engine.get_scenario(scenario_id)
        if scenario:
            return scenario
        raise HTTPException(status_code=404, detail="Scenario not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get scenario rule: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/scenario-rules/active/status")
async def get_active_scenario_status():
    """Get the status of the currently active scenario."""
    try:
        from services.scenario import get_scenario_rule_engine
        engine = get_scenario_rule_engine()
        return {
            "active": engine.is_active(),
            "scenario": engine.get_active_scenario()
        }
    except Exception as e:
        logger.error(f"Failed to get scenario status: {e}")
        return {"active": False, "scenario": None, "error": str(e)}


@app.post("/scenario-rules/reload")
async def reload_scenario_rules():
    """Reload scenario rules from config file."""
    try:
        from services.scenario import get_scenario_rule_engine
        engine = get_scenario_rule_engine()
        engine.reload_config()
        return {"message": "Scenario rules reloaded", "count": len(engine.get_all_scenarios())}
    except Exception as e:
        logger.error(f"Failed to reload scenario rules: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/scenario-rules/armed-person")
async def trigger_armed_person(person_data: dict):
    """
    Trigger armed person event on the ScenarioRuleEngine.

    This endpoint allows the backend to notify the AI service about armed persons
    detected via demo or other means that bypass the normal detection flow.

    Required fields in person_data:
    - trackId: Track ID of the person
    - armed: Boolean (should be True)
    - cameraId: Camera ID where detected

    Optional fields:
    - clothing, clothingColor, weaponType, confidence, bbox, etc.
    """
    try:
        from services.scenario import get_scenario_rule_engine
        engine = get_scenario_rule_engine()

        # Ensure armed is set
        person_data['armed'] = True

        # Trigger the rule engine
        triggered = await engine.handle_armed_person(person_data)

        return {
            "success": True,
            "triggered": triggered,
            "scenarioActive": engine.is_active(),
            "activeScenario": engine.get_active_scenario()
        }
    except Exception as e:
        logger.error(f"Failed to trigger armed person: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/scenario-rules/vehicle-detected")
async def trigger_vehicle_detected(vehicle_data: dict):
    """
    Trigger vehicle detection event on the ScenarioRuleEngine.

    This endpoint allows the backend to notify the AI service about stolen vehicles
    detected via demo or other means that bypass the normal detection flow.

    Required fields in vehicle_data:
    - licensePlate: License plate of the vehicle
    - cameraId: Camera ID where detected

    Optional fields:
    - color, make, model, vehicleType, confidence, bbox, trackId, etc.
    """
    try:
        from services.scenario import get_scenario_rule_engine
        engine = get_scenario_rule_engine()

        # Trigger the rule engine's vehicle detection handler
        triggered = await engine.handle_vehicle_detection(vehicle_data)

        return {
            "success": True,
            "triggered": triggered,
            "scenarioActive": engine.is_active(),
            "activeScenario": engine.get_active_scenario()
        }
    except Exception as e:
        logger.error(f"Failed to trigger vehicle detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/scenario-rules/start-recording")
async def trigger_start_recording(request_data: dict):
    """
    Manually trigger recording start for a specific camera.

    This endpoint allows the backend to start recording when the ScenarioRuleEngine
    is not managing the scenario (e.g., during demo mode).

    Required fields:
    - cameraId: Camera ID to record

    Optional fields:
    - duration: Recording duration in seconds (default: 60)
    - preBuffer: Pre-event buffer in seconds (default: 30)
    - reason: Trigger reason (default: "manual")
    """
    try:
        from services.recording.recording_manager import get_recording_manager

        recording_manager = get_recording_manager()
        if not recording_manager:
            raise HTTPException(status_code=503, detail="Recording manager not available")

        camera_id = request_data.get('cameraId')
        if not camera_id:
            raise HTTPException(status_code=400, detail="cameraId is required")

        duration = request_data.get('duration', 60)
        pre_buffer = request_data.get('preBuffer', 30)
        reason = request_data.get('reason', 'manual')
        metadata = request_data.get('metadata', {})

        recording_id = recording_manager.start_recording(
            camera_id=camera_id,
            duration=duration,
            pre_buffer=pre_buffer,
            trigger_reason=reason,
            metadata=metadata
        )

        if recording_id:
            return {
                "success": True,
                "recordingId": recording_id,
                "cameraId": camera_id,
                "duration": duration
            }
        else:
            return {
                "success": False,
                "message": "Could not start recording (may already be recording)",
                "cameraId": camera_id
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start recording: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ffmpeg/stop/{camera_id}")
async def ffmpeg_stop(camera_id: str):
    """Stop an FFmpeg stream."""
    ffmpeg_manager.remove_camera(camera_id)
    return {"status": "stopped", "camera_id": camera_id}


# ============== TRACKER ENDPOINTS ==============

@app.get("/tracker/stable/stats")
async def stable_tracker_stats():
    """Get StableTracker statistics."""
    tracker = get_stable_tracker()
    return {
        "stats": tracker.get_stats(),
        "vehicles": [v.to_dict() for v in tracker.get_all_vehicles()],
        "persons": [p.to_dict() for p in tracker.get_all_persons()],
        "armed_persons": [p.to_dict() for p in tracker.get_armed_persons()]
    }


@app.post("/tracker/stable/reset")
async def reset_stable_tracker():
    """Reset the StableTracker - clears all tracked objects."""
    tracker = get_stable_tracker()
    tracker.reset()
    return {"message": "Tracker reset", "stats": tracker.get_stats()}


@app.get("/api/stream/debug/{camera_id}")
async def stream_debug(camera_id: str):
    """Debug endpoint to check stream state."""
    detection_loop = get_detection_loop()
    rtsp_manager = get_rtsp_manager()

    state = {
        "detection_loop_running": detection_loop._running if detection_loop else False,
        "active_cameras": rtsp_manager.get_active_cameras(),
        "pending_frames": list(detection_loop._latest_frames.keys()) if detection_loop else [],
        "annotated_frames": list(detection_loop._annotated_frames.keys()) if detection_loop else [],
        "camera_requested": camera_id,
        "camera_in_active": camera_id in rtsp_manager.get_active_cameras(),
        "has_annotated_frame": camera_id in detection_loop._annotated_frames if detection_loop else False,
    }

    # Get RTSP reader stats if available
    reader = rtsp_manager.get_reader(camera_id) if hasattr(rtsp_manager, 'get_reader') else None
    if reader:
        state["reader_stats"] = reader.get_stats()

    # Get detection loop stats
    if detection_loop:
        state["detection_stats"] = detection_loop.get_stats()

    return state


@app.get("/api/detections/{camera_id}/latest")
async def get_latest_detections(camera_id: str):
    """
    Get latest detections for a camera (for WebRTC overlay).

    Returns JSON-serializable detection data including bounding boxes,
    class labels, confidence scores, and track IDs.

    This endpoint is optimized for low overhead to support polling
    at 5-10 FPS from the frontend WebRTC overlay.
    """
    detection_loop = get_detection_loop()
    if not detection_loop:
        raise HTTPException(503, "Detection loop not running")

    detections = detection_loop.get_latest_detections(camera_id)

    return {
        "camera_id": camera_id,
        "timestamp": time.time(),
        "detections": detections,
        "count": len(detections)
    }


@app.websocket("/api/detections/{camera_id}/ws")
async def websocket_detections(websocket: WebSocket, camera_id: str):
    """
    WebSocket endpoint for real-time detection streaming.

    Clients connect and receive detection updates pushed from the server
    whenever new detections are available. This eliminates polling overhead.

    Message format (JSON):
    {
        "camera_id": "test",
        "timestamp": 1703348000.123,
        "detections": [...],
        "count": 5
    }
    """
    from services.detection_loop import get_ws_manager

    ws_manager = get_ws_manager()

    try:
        await ws_manager.connect(camera_id, websocket)

        # Keep connection alive
        while True:
            try:
                # Wait for any message (ping/pong or close)
                # Timeout every 30s to check connection is still alive
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                # Client can send "ping" to keep alive
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                # Send ping to check connection
                try:
                    await websocket.send_text('{"type":"ping"}')
                except Exception:
                    break
            except WebSocketDisconnect:
                break
            except Exception:
                break
    finally:
        await ws_manager.disconnect(camera_id, websocket)


@app.post("/api/ai/{camera_id}/enable")
async def enable_ai(camera_id: str, rtsp_url: Optional[str] = None):
    """Enable AI processing for a specific camera.

    This will:
    1. Enable AI processing flag for the camera
    2. Ensure the camera stream is in the RTSP reader (starts it if needed)

    If camera not in RTSP reader, tries to get RTSP URL from:
    - rtsp_url parameter (if provided)
    - go2rtc streams API
    - Backend camera config
    """
    import httpx

    detection_loop = get_detection_loop()
    if not detection_loop:
        raise HTTPException(503, "Detection loop not running")

    rtsp_manager = get_rtsp_manager()
    active_cameras = rtsp_manager.get_active_cameras()

    # Check if camera already in RTSP reader
    camera_needs_start = camera_id not in active_cameras

    if camera_needs_start:
        logger.info(f"Camera {camera_id} not in RTSP reader, attempting to start...")

        # Try to get RTSP URL
        source_url = rtsp_url

        # If no URL provided, try go2rtc first
        if not source_url:
            try:
                GO2RTC_URL = os.getenv("GO2RTC_URL", "http://localhost:1984")
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(f"{GO2RTC_URL}/api/streams")
                    if response.status_code == 200:
                        streams = response.json()
                        if camera_id in streams:
                            # Get the source URL from go2rtc config
                            sources = streams[camera_id]
                            if isinstance(sources, list) and sources:
                                source_url = sources[0]
                                # Clean up go2rtc format (remove #transport=tcp etc for our reader)
                                if "#" in source_url:
                                    source_url = source_url.split("#")[0]
                                logger.info(f"Got RTSP URL from go2rtc: {source_url[:50]}...")
            except Exception as e:
                logger.debug(f"Could not get URL from go2rtc: {e}")

        # If still no URL, try backend
        if not source_url:
            try:
                BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:3000")
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(f"{BACKEND_URL}/api/cameras/{camera_id}")
                    if response.status_code == 200:
                        camera_config = response.json()
                        source_url = camera_config.get("rtspUrl")
                        if source_url:
                            logger.info(f"Got RTSP URL from backend")
            except Exception as e:
                logger.debug(f"Could not get URL from backend: {e}")

        if not source_url:
            raise HTTPException(400, f"Camera {camera_id} not found and no RTSP URL provided. "
                              "Add camera to go2rtc or provide rtsp_url parameter.")

        # Start the camera in RTSP reader
        config = RTSPConfig(
            width=1280,
            height=720,
            fps=TARGET_FPS,
            tcp_transport=True
        )

        try:
            rtsp_manager.add_camera(camera_id, source_url, config)
            # Wait for first frame
            await asyncio.sleep(1.0)
            logger.info(f"✅ Started RTSP reader for camera {camera_id}")
        except Exception as e:
            logger.error(f"Failed to start camera {camera_id}: {e}")
            raise HTTPException(503, f"Failed to start camera stream: {e}")

    # Enable AI processing
    detection_loop.set_ai_enabled(camera_id, True)

    return {
        "camera_id": camera_id,
        "ai_enabled": True,
        "stream_started": camera_needs_start,
        "message": "AI enabled" + (" and RTSP reader started" if camera_needs_start else "")
    }


@app.post("/api/ai/{camera_id}/disable")
async def disable_ai(camera_id: str):
    """Disable AI processing for a specific camera (saves resources)."""
    detection_loop = get_detection_loop()
    if not detection_loop:
        raise HTTPException(503, "Detection loop not running")

    detection_loop.set_ai_enabled(camera_id, False)
    return {"camera_id": camera_id, "ai_enabled": False}


@app.get("/api/ai/{camera_id}/status")
async def get_ai_status(camera_id: str):
    """Get AI processing status for a specific camera."""
    detection_loop = get_detection_loop()
    if not detection_loop:
        raise HTTPException(503, "Detection loop not running")

    enabled = detection_loop.is_ai_enabled(camera_id)
    return {"camera_id": camera_id, "ai_enabled": enabled}


@app.get("/api/ai/status")
async def get_all_ai_status():
    """Get AI processing status for all cameras."""
    detection_loop = get_detection_loop()
    if not detection_loop:
        raise HTTPException(503, "Detection loop not running")

    status = detection_loop.get_ai_status()
    return {"cameras": status}


@app.get("/api/stream/annotated/{camera_id}")
async def stream_annotated(camera_id: str, fps: Optional[int] = None):
    """
    Stream annotated frames with bounding boxes via SSE.
    This shows the AI detection results in real-time.
    Smooth delivery with frame deduplication.

    Args:
        camera_id: Camera to stream
        fps: Optional FPS override. If not provided, uses detection loop's stream_fps config.
    """
    import base64

    detection_loop = get_detection_loop()
    if not detection_loop:
        raise HTTPException(503, "Detection loop not running")

    # Debug: Check current state
    rtsp_manager = get_rtsp_manager()
    active_cameras = rtsp_manager.get_active_cameras()
    logger.info(f"[Stream] Request for {camera_id}, active cameras: {active_cameras}")
    logger.info(f"[Stream] Detection loop running: {detection_loop._running}")
    logger.info(f"[Stream] Annotated frames available: {list(detection_loop._annotated_frames.keys())}")

    # Use configured stream_fps if not explicitly provided
    if fps is None:
        fps = detection_loop.config.stream_fps

    fps = max(1, min(fps, 30))  # Allow up to 30 FPS
    frame_interval = 1.0 / fps

    async def generate():
        last_frame_id = None
        last_send_time = time.time()

        while True:
            current_time = time.time()
            elapsed = current_time - last_send_time

            # Only send if enough time has elapsed (smooth pacing)
            if elapsed >= frame_interval:
                frame = detection_loop.get_annotated_frame(camera_id)

                if frame is not None:
                    # Use frame hash to detect duplicates
                    frame_id = hash(frame.tobytes())

                    # Only send if it's a new frame (prevents bursts of duplicates)
                    if frame_id != last_frame_id:
                        ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
                        if ret:
                            frame_b64 = base64.b64encode(jpeg.tobytes()).decode('utf-8')
                            yield f"data: {{\"frame\": \"{frame_b64}\"}}\n\n"
                            last_frame_id = frame_id
                            last_send_time = current_time
                    else:
                        # Same frame, wait for next interval
                        last_send_time = current_time

            # Small sleep to prevent CPU spinning
            await asyncio.sleep(0.01)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    uvicorn.run(app, host=host, port=port)
