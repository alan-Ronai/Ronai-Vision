from typing import AsyncIterator
from fastapi import APIRouter, WebSocket, HTTPException, Request
from starlette.responses import StreamingResponse
import base64
import asyncio

from services.output.broadcaster import broadcaster

router = APIRouter()


async def mjpeg_generator(cam_id: str) -> AsyncIterator[bytes]:
    boundary = b"--frame"
    while True:
        jpeg = await broadcaster.wait_for_frame(cam_id, timeout=5.0)
        if jpeg is None:
            # send a short heartbeat comment to keep connection alive
            await asyncio.sleep(0.1)
            continue
        part = b"\r\n" + boundary + b"\r\n"
        part += b"Content-Type: image/jpeg\r\n"
        part += f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
        part += jpeg
        yield part


@router.get("/mjpeg/{cam_id}")
def mjpeg_stream(cam_id: str):
    """HTTP MJPEG stream for a camera. Open in browser or VLC.

    Example: http://localhost:8000/api/stream/mjpeg/cam1
    """

    async def app_gen():
        async for chunk in mjpeg_generator(cam_id):
            yield chunk

    return StreamingResponse(
        app_gen(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.websocket("/ws/{cam_id}")
async def websocket_stream(websocket: WebSocket, cam_id: str):
    await websocket.accept()
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


@router.get("/status")
def stream_status():
    """Return last-publish timestamps per camera to help debug whether frames are arriving."""
    stats = broadcaster.status()
    return {"last_published": stats}
