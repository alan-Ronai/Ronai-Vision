from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from services.ptz.ptz_interface import PTZSimulator
from api.websocket import websocket_endpoint

router = APIRouter(prefix="/api")

ptz = PTZSimulator()


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/ptz")
async def get_ptz():
    return ptz.state()


@router.post("/ptz/move")
async def move_ptz(
    pan_delta: float = 0.0, tilt_delta: float = 0.0, zoom_delta: float = 0.0
):
    ptz.move(pan_delta=pan_delta, tilt_delta=tilt_delta, zoom_delta=zoom_delta)
    return ptz.state()


@router.websocket("/ws")
async def websocket_route(websocket: WebSocket):
    await websocket_endpoint(websocket)
