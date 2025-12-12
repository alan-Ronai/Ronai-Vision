from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from api.websocket import websocket_endpoint

router = APIRouter(prefix="/api")


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.websocket("/ws")
async def websocket_route(websocket: WebSocket):
    await websocket_endpoint(websocket)
