"""
HAMAL-AI Detection Service - Unified

FastAPI service for AI-powered detection using:
- YOLO for object detection
- ReID (DeepSort) for persistent tracking
- Gemini for ALL analysis (vehicle, person, threat assessment)
- Google Cloud TTS for Hebrew announcements

Supports Mac MPS acceleration.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
import threading
import time
import torch
from ultralytics import YOLO
import numpy as np
import cv2
import os
import sys
import asyncio
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

# Import services
from services.reid_tracker import ReIDTracker
from services.gemini_analyzer import GeminiAnalyzer
from services.tts_service import TTSService
from services.rtsp_reader import get_rtsp_manager, RTSPConfig
from services.detection_loop import init_detection_loop, get_detection_loop, LoopConfig
from services.radio import init_radio_service, get_radio_service, stop_radio_service
from services.detection import get_frame_buffer_manager, get_stable_tracker
from services.ffmpeg_rtsp import get_ffmpeg_manager, FFmpegConfig

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

# Model paths
MODEL_PATH = os.getenv("YOLO_MODEL", "yolo12n.pt")
CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", "0.25"))

# Try to find model in various locations
model_locations = [
    MODEL_PATH,
    f"models/{MODEL_PATH}",
    f"../models/{MODEL_PATH}",
    f"../../models/{MODEL_PATH}",
]

# Load main YOLO model
yolo = None
for model_path in model_locations:
    if os.path.exists(model_path):
        logger.info(f"Loading YOLO model from: {model_path}")
        yolo = YOLO(model_path)
        yolo.to(DEVICE)
        logger.info(f"✅ YOLO loaded on {DEVICE}")
        break

if yolo is None:
    logger.info("⚠️ YOLO model not found, downloading default")
    yolo = YOLO("yolo11n.pt")
    yolo.to(DEVICE)
    logger.info(f"✅ YOLO loaded on {DEVICE}")

# Optional: Load weapon detection model
weapon_detector = None
weapon_model_path = os.getenv("WEAPON_MODEL", "models/weapon_yolov8.pt")
if os.path.exists(weapon_model_path):
    try:
        weapon_detector = YOLO(weapon_model_path)
        weapon_detector.to(DEVICE)
        logger.info(f"✅ Weapon detector loaded")
    except Exception as e:
        logger.warning(f"Could not load weapon detector: {e}")

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
        """Start capturing from RTSP stream"""
        with self.lock:
            if camera_id in self.streams and self.streams[camera_id].get("running"):
                logger.info(f"Stream already running for {camera_id}")
                return True  # Already running

        # Check if this is a local file path (not a network URL)
        if not rtsp_url.startswith(('rtsp://', 'http://', 'https://', 'rtmp://', 'udp://')):
            # Local file path - resolve relative to Ronai-Vision root
            # Current file is at: Ronai-Vision/hamal-ai/ai-service/main.py
            # Root is two levels up: ../../
            script_dir = Path(__file__).parent
            project_root = script_dir.parent.parent  # Ronai-Vision root

            # Clean up the path (remove leading slash if present for relative paths)
            clean_path = rtsp_url.lstrip('/')

            # Try as relative path first (from project root)
            video_path = project_root / clean_path

            # If doesn't exist as relative, try as absolute
            if not video_path.exists() and Path(rtsp_url).is_absolute():
                video_path = Path(rtsp_url)

            # Convert to absolute path
            rtsp_url = str(video_path.absolute())

            # Check if file exists
            if not video_path.exists():
                logger.error(f"Video file not found: {rtsp_url}")
                logger.error(f"Looked in project root: {project_root}")
                return False

            logger.info(f"✓ Resolved local video path: {rtsp_url}")

        # Build full RTSP URL with credentials if not already in URL
        full_url = rtsp_url
        if username and password and "@" not in rtsp_url:
            if "://" in rtsp_url:
                protocol, rest = rtsp_url.split("://", 1)
                full_url = f"{protocol}://{username}:{password}@{rest}"

        logger.info(f"Starting RTSP stream for {camera_id}")
        logger.info(f"URL: {rtsp_url[:50]}...")

        # Set environment for FFmpeg
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

        # OpenCV VideoCapture with RTSP over TCP
        cap = cv2.VideoCapture(full_url, cv2.CAP_FFMPEG)

        # Configure capture for low latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 15)  # Limit FPS

        if not cap.isOpened():
            logger.error(f"Failed to open RTSP stream for {camera_id}")
            logger.error(f"Make sure FFmpeg is installed and the URL is correct")
            return False

        # Read first frame to verify connection
        ret, test_frame = cap.read()
        if not ret:
            logger.error(f"Connected but failed to read first frame for {camera_id}")
            cap.release()
            return False

        logger.info(f"First frame received: {test_frame.shape}")

        with self.lock:
            stream_data = {
                "cap": cap,
                "url": full_url,
                "frame": test_frame,  # Store first frame immediately
                "running": True,
                "last_update": time.time(),
                "error": None,
                "reconnect_count": 0
            }
            self.streams[camera_id] = stream_data

        # Start capture thread
        thread = threading.Thread(target=self._capture_loop, args=(camera_id,), daemon=True)
        with self.lock:
            self.streams[camera_id]["thread"] = thread
        thread.start()

        logger.info(f"✅ RTSP stream started for {camera_id}")
        return True

    def _capture_loop(self, camera_id: str):
        """Background thread to continuously capture frames"""
        consecutive_errors = 0
        max_consecutive_errors = 30  # ~3 seconds of errors before reconnect

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

            # Read frame
            ret, frame = cap.read()

            if ret and frame is not None:
                consecutive_errors = 0
                with self.lock:
                    if camera_id in self.streams:
                        self.streams[camera_id]["frame"] = frame
                        self.streams[camera_id]["last_update"] = time.time()
                        self.streams[camera_id]["error"] = None
            else:
                consecutive_errors += 1
                logger.warning(f"Failed to read frame for {camera_id} ({consecutive_errors}/{max_consecutive_errors})")

                if consecutive_errors >= max_consecutive_errors:
                    logger.warning(f"Too many errors, attempting reconnect for {camera_id}")
                    cap.release()

                    # Wait before reconnecting
                    time.sleep(2)

                    # Reconnect
                    new_cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
                    new_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                    if new_cap.isOpened():
                        with self.lock:
                            if camera_id in self.streams:
                                self.streams[camera_id]["cap"] = new_cap
                                self.streams[camera_id]["reconnect_count"] = \
                                    self.streams[camera_id].get("reconnect_count", 0) + 1
                        cap = new_cap
                        consecutive_errors = 0
                        logger.info(f"Reconnected to stream for {camera_id}")
                    else:
                        logger.error(f"Reconnect failed for {camera_id}")
                        with self.lock:
                            if camera_id in self.streams:
                                self.streams[camera_id]["error"] = "Connection lost"
                        time.sleep(5)  # Wait longer before next attempt

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
        "tts_engine": tts.engine_type if tts.is_configured() else None,
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
            # Encode frame as JPEG
            ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                frame_count += 1
                if frame_count % 100 == 0:
                    logger.debug(f"Streamed {frame_count} frames for {camera_id}")
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        else:
            await asyncio.sleep(0.1)


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
            # Encode frame as JPEG
            ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                frame_count += 1
                if frame_count % 100 == 0:
                    logger.debug(f"FFmpeg stream: {frame_count} frames for {camera_id}")
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        else:
            await asyncio.sleep(0.1)


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
            # Encode frame as JPEG
            ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                frame_count += 1
                if frame_count % 100 == 0:
                    logger.debug(f"Buffered stream: {frame_count} frames for {camera_id}")
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        else:
            await asyncio.sleep(0.1)


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
                config = RTSPConfig(width=1280, height=720, fps=15, tcp_transport=True)
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

    logger.info(f"Starting MJPEG response for {camera_id}")
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
    frame = stream_manager.get_frame(camera_id)
    if frame is None:
        # Try to start stream first
        camera_config = stream_manager.get_camera_config(camera_id)
        if camera_config and camera_config.get("rtspUrl"):
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

    ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ret:
        raise HTTPException(500, "Failed to encode frame")

    return StreamingResponse(
        iter([jpeg.tobytes()]),
        media_type="image/jpeg"
    )


@app.get("/api/stream/sse/{camera_id}")
async def stream_sse(camera_id: str, fps: int = 5):
    """
    Server-Sent Events streaming - stable single connection.
    Server controls frame rate (default 5 FPS).
    """
    import base64

    logger.info(f"SSE stream requested for camera: {camera_id}, fps: {fps}")

    # Clamp FPS to reasonable range
    fps = max(1, min(fps, 15))
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
                    # Encode as JPEG with moderate quality
                    ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
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
                    await asyncio.sleep(0.5)

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

                # Check if armed - trigger emergency
                if analysis.get("armed"):
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


@app.delete("/tracker/cleanup")
async def cleanup_old_tracks(
    max_age_seconds: int = Query(3600, description="Max age in seconds")
):
    """Remove old tracks that haven't been seen"""
    removed = tracker.cleanup_old_tracks(max_age_seconds)
    return {"removed": removed}


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
║   TTS: {tts.engine_type or '❌'}                                     ║
║   Weapon Det: {'✅' if weapon_detector else '❌'}                                   ║
║   Backend: {BACKEND_URL}                 ║
║                                                        ║
╚════════════════════════════════════════════════════════╝
    """)

    # Initialize detection loop
    loop_config = LoopConfig(
        backend_url=BACKEND_URL,
        detection_fps=15,  # Process all frames (15 FPS) for continuous analysis
        stream_fps=15,     # Stream at 15 fps for smooth viewing
        draw_bboxes=True,
        send_events=True
    )

    detection_loop = init_detection_loop(yolo, tracker, gemini, loop_config)

    # Start detection loop
    await detection_loop.start()

    # Set up RTSP manager to feed frames to detection loop
    rtsp_manager = get_rtsp_manager()
    rtsp_manager.add_frame_callback(detection_loop.on_frame)

    # Auto-load cameras from backend
    asyncio.create_task(auto_load_cameras())

    logger.info("✅ Detection loop initialized and ready")

    # Start radio service for RTP audio transcription via EC2 relay
    ec2_host = os.getenv("EC2_RTP_HOST")
    ec2_port = int(os.getenv("EC2_RTP_PORT", "5005"))

    if ec2_host:
        try:
            await init_radio_service(
                ec2_host=ec2_host,
                ec2_port=ec2_port,
                sample_rate=16000,
                backend_url=BACKEND_URL,
                chunk_duration=3.0
            )
            logger.info(f"📻 Radio service started - EC2 relay: {ec2_host}:{ec2_port}")
        except Exception as e:
            logger.warning(f"Could not start radio service: {e}")
    else:
        logger.info("📻 Radio service not started (EC2_RTP_HOST not configured)")


async def auto_load_cameras():
    """Load cameras from backend and start RTSP readers."""
    import httpx

    await asyncio.sleep(2)  # Wait for backend to be ready

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BACKEND_URL}/api/cameras", timeout=10.0)

            if response.status_code != 200:
                logger.error(f"Failed to load cameras: {response.status_code}")
                return

            cameras = response.json()
            rtsp_manager = get_rtsp_manager()

            for camera in cameras:
                rtsp_url = camera.get('rtspUrl')
                if not rtsp_url:
                    continue

                camera_id = camera.get('cameraId') or str(camera.get('_id'))
                ai_enabled = camera.get('aiEnabled', True)

                if not ai_enabled:
                    logger.info(f"Skipping camera {camera_id} (AI disabled)")
                    continue

                logger.info(f"📹 Starting camera: {camera.get('name', camera_id)}")

                config = RTSPConfig(
                    width=1280,
                    height=720,
                    fps=5,
                    tcp_transport=True
                )

                rtsp_manager.add_camera(camera_id, rtsp_url, config)

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


@app.post("/detection/reload")
async def reload_cameras():
    """Reload cameras from backend."""
    await auto_load_cameras()
    return {"status": "reloaded", "cameras": get_rtsp_manager().get_active_cameras()}


@app.post("/detection/start/{camera_id}")
async def start_camera_detection(camera_id: str):
    """Start detection for a specific camera."""
    import httpx

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BACKEND_URL}/api/cameras/{camera_id}")
            if response.status_code != 200:
                raise HTTPException(404, f"Camera {camera_id} not found")

            camera = response.json()
            rtsp_url = camera.get('rtspUrl')
            if not rtsp_url:
                raise HTTPException(400, f"Camera {camera_id} has no RTSP URL")

            config = RTSPConfig(width=1280, height=720, fps=5, tcp_transport=True)
            get_rtsp_manager().add_camera(camera_id, rtsp_url, config)

            return {"status": "started", "camera_id": camera_id}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/detection/stop/{camera_id}")
async def stop_camera_detection(camera_id: str):
    """Stop detection for a specific camera."""
    get_rtsp_manager().remove_camera(camera_id)
    return {"status": "stopped", "camera_id": camera_id}


# ============== RADIO ENDPOINTS ==============

@app.get("/radio/stats")
async def radio_stats():
    """Get radio service statistics."""
    service = get_radio_service()
    if not service:
        return {"error": "Radio service not running", "running": False}
    return service.get_stats()


# ============== FFMPEG STREAMING ENDPOINTS ==============

@app.get("/ffmpeg/stats")
async def ffmpeg_stats():
    """Get FFmpeg stream statistics."""
    return {
        "streams": ffmpeg_manager.get_all_stats(),
        "active_count": len(ffmpeg_manager.get_active_cameras())
    }


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


@app.get("/api/stream/annotated/{camera_id}")
async def stream_annotated(camera_id: str, fps: int = 15):
    """
    Stream annotated frames with bounding boxes via SSE.
    This shows the AI detection results in real-time.
    """
    import base64

    detection_loop = get_detection_loop()
    if not detection_loop:
        raise HTTPException(503, "Detection loop not running")

    fps = max(1, min(fps, 15))
    frame_interval = 1.0 / fps

    async def generate():
        while True:
            frame = detection_loop.get_annotated_frame(camera_id)

            if frame is not None:
                ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ret:
                    frame_b64 = base64.b64encode(jpeg.tobytes()).decode('utf-8')
                    yield f"data: {{\"frame\": \"{frame_b64}\"}}\n\n"

            await asyncio.sleep(frame_interval)

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
