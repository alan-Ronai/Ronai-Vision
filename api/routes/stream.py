from typing import AsyncIterator
from fastapi import APIRouter, WebSocket, HTTPException, Request
from starlette.responses import StreamingResponse
import base64
import asyncio

from services.output.broadcaster import broadcaster
from services import viewer_tracker

router = APIRouter()


async def mjpeg_generator(cam_id: str) -> AsyncIterator[bytes]:
    """MJPEG frame generator that yields frames with proper client disconnect handling."""
    import threading

    # Increment viewer count when client connects (cross-process safe)
    viewer_count = viewer_tracker.increment_viewer(cam_id)
    print(f"[MJPEG] Viewer connected to {cam_id} (total viewers: {viewer_count})")

    boundary = b"--frame"
    idle_count = 0
    max_idle = 100  # Max 10 seconds of idle waiting (100 * 0.1s sleep)
    cleaned_up = False

    try:
        while True:
            # Check if server is shutting down
            try:
                from api.server import _runner_stop_event

                if _runner_stop_event is not None and _runner_stop_event.is_set():
                    break
            except (ImportError, AttributeError):
                pass

            try:
                # Use short timeout (0.1s) to be responsive to Ctrl+C
                jpeg = await broadcaster.wait_for_frame(cam_id, timeout=0.1)
            except asyncio.CancelledError:
                # Handle task cancellation (Ctrl+C, client disconnect, etc.)
                print(f"[MJPEG] {cam_id} cancelled")
                break
            except Exception as e:
                # If anything goes wrong, exit the generator
                print(f"[MJPEG] {cam_id} wait_for_frame error: {e}")
                break

            if jpeg is None:
                idle_count += 1
                if idle_count > max_idle:
                    # No frames received; timeout
                    print(
                        f"[MJPEG] {cam_id} timeout: no frames after {max_idle * 0.1:.1f}s"
                    )
                    break
                # Don't log progress for every iteration, it's too noisy
                continue

            idle_count = 0
            try:
                part = b"\r\n" + boundary + b"\r\n"
                part += b"Content-Type: image/jpeg\r\n"
                part += f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                part += jpeg
                yield part
            except Exception as e:
                # Client disconnected or other error; stop generator
                print(f"[MJPEG] {cam_id} send error: {e}")
                break
    finally:
        # Decrement viewer count when client disconnects (cross-process safe)
        # Only decrement once
        if not cleaned_up:
            cleaned_up = True
            viewer_count = viewer_tracker.decrement_viewer(cam_id)
            print(
                f"[MJPEG] Viewer disconnected from {cam_id} (remaining viewers: {viewer_count})"
            )


@router.get("/mjpeg/{cam_id}")
async def mjpeg_stream(cam_id: str, request: Request):
    """HTTP MJPEG stream for a camera. Open in browser or VLC.

    Example: http://localhost:8000/api/stream/mjpeg/cam1
    """

    async def stream_with_disconnect_detection():
        """Wrapper that detects client disconnection."""
        async for chunk in mjpeg_generator(cam_id):
            # Check if client has disconnected
            if await request.is_disconnected():
                print(f"[MJPEG] {cam_id} client disconnected (detected)")
                break
            yield chunk

    return StreamingResponse(
        stream_with_disconnect_detection(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.websocket("/ws/{cam_id}")
async def websocket_stream(websocket: WebSocket, cam_id: str):
    await websocket.accept()

    # Increment viewer count when client connects
    viewer_count = broadcaster.increment_viewer(cam_id)
    print(f"[WebSocket] Viewer connected to {cam_id} (total viewers: {viewer_count})")

    try:
        while True:
            jpeg = await broadcaster.wait_for_frame(cam_id, timeout=10.0)
            if jpeg is None:
                # send ping
                try:
                    await websocket.send_json({"type": "ping"})
                except Exception:
                    break
                continue

            # send base64-encoded jpeg to keep websocket text-safe
            payload = base64.b64encode(jpeg).decode("ascii")
            try:
                await websocket.send_json({"type": "frame", "data": payload})
            except Exception:
                break

    finally:
        # Decrement viewer count when client disconnects
        viewer_count = broadcaster.decrement_viewer(cam_id)
        print(
            f"[WebSocket] Viewer disconnected from {cam_id} (remaining viewers: {viewer_count})"
        )
        await websocket.close()


@router.post("/publish/{cam_id}")
async def publish_frame(cam_id: str, request: Request):
    """Publish a JPEG frame to the server broadcaster.

    Accepts either raw `image/jpeg` body or multipart/form-data `file`.
    Optional header: `X-STREAM-TOKEN` for simple auth (compare to env STREAM_PUBLISH_TOKEN).
    """
    # simple token-based auth
    import os

    token = os.getenv("STREAM_PUBLISH_TOKEN")
    if token:
        hdr = request.headers.get("x-stream-token") or request.headers.get(
            "authorization"
        )
        if hdr is None:
            raise HTTPException(status_code=401, detail="missing publish token")
        # allow `Bearer <token>` or raw header
        if hdr.lower().startswith("bearer "):
            hdr_val = hdr.split(" ", 1)[1]
        else:
            hdr_val = hdr
        if hdr_val != token:
            raise HTTPException(status_code=403, detail="invalid publish token")

    content_type = request.headers.get("content-type", "")
    body = None
    if content_type.startswith("image/"):
        body = await request.body()
    else:
        # try multipart
        form = await request.form()
        if "file" in form:
            f = form["file"]
            body = await f.read()

    if not body:
        raise HTTPException(status_code=400, detail="no image data received")

    # publish to broadcaster
    try:
        # decode JPEG to BGR numpy and republish via broadcaster
        import numpy as np
        import cv2

        arr = np.frombuffer(body, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("invalid jpeg")
        broadcaster.publish_frame(cam_id, img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"publish failed: {e}")

    return {"ok": True, "cam_id": cam_id}


@router.get("/cameras")
def list_cameras():
    """List all configured cameras with their status and viewer counts.

    Returns:
        List of camera objects with:
        - camera_id: Camera identifier
        - status: 'inactive' (no viewers) or 'active' (has viewers)
        - viewer_count: Number of active connections
        - last_frame_ts: Timestamp of last received frame (epoch seconds, or null)
    """
    # Get camera IDs from broadcaster (if populated)
    camera_ids = broadcaster.get_all_camera_ids()

    # Fallback 1: if broadcaster is empty, try to get from camera manager
    if not camera_ids:
        try:
            from scripts.run_multi_camera import get_camera_manager

            cm = get_camera_manager()
            if cm:
                camera_ids = cm.get_camera_ids()
                # Register them with broadcaster for tracking
                for cam_id in camera_ids:
                    broadcaster.register_camera(cam_id)
        except (ImportError, AttributeError):
            pass

    # Fallback 2: Load directly from config file
    if not camera_ids:
        try:
            from config.pipeline_config import PipelineConfig
            import json
            import os

            if os.path.exists(PipelineConfig.CAMERA_CONFIG):
                with open(PipelineConfig.CAMERA_CONFIG, "r") as f:
                    config = json.load(f)
                    cameras_dict = config.get("cameras", {})
                    camera_ids = list(cameras_dict.keys())

                    # Register with broadcaster
                    for cam_id in camera_ids:
                        broadcaster.register_camera(cam_id)
        except Exception:
            pass

    status_map = broadcaster.status()

    cameras = []
    for cam_id in camera_ids:
        # Use file-based viewer tracker for viewer count (cross-process safe)
        viewer_count = viewer_tracker.get_viewer_count(cam_id)
        last_ts = status_map.get(cam_id)

        cameras.append(
            {
                "camera_id": cam_id,
                "status": "active" if viewer_count > 0 else "inactive",
                "viewer_count": viewer_count,
                "last_frame_ts": last_ts,
            }
        )

    return {"cameras": cameras, "total": len(cameras)}


@router.get("/status")
def stream_status():
    """Return last-publish timestamps per camera to help debug whether frames are arriving."""
    stats = broadcaster.status()
    return {"last_published": stats}
