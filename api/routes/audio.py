"""API endpoints for RTP/RTSP audio server control and monitoring."""

import os
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Optional

from services.audio import RTPAudioServer


router = APIRouter()

# Global audio server instance
_audio_server: Optional[RTPAudioServer] = None


def get_audio_server() -> RTPAudioServer:
    """Get or create audio server instance."""
    global _audio_server

    if _audio_server is None:
        # Load config
        config_path = os.getenv("AUDIO_CONFIG", "config/audio_settings.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
        else:
            config = {}

        # Create server
        rtsp_config = config.get("rtsp", {})
        rtp_config = config.get("rtp", {})
        storage_config = config.get("storage", {})

        _audio_server = RTPAudioServer(
            rtsp_host=rtsp_config.get("host", "0.0.0.0"),
            rtsp_port=rtsp_config.get("port", 8554),
            rtp_base_port=rtp_config.get("base_port", 5004),
            storage_path=storage_config.get("base_path", "audio_storage/recordings"),
            session_timeout=rtsp_config.get("session_timeout", 60),
            jitter_buffer_ms=rtp_config.get("jitter_buffer_ms", 100)
        )

    return _audio_server


# Response models
class ServerStatus(BaseModel):
    """Audio server status."""
    running: bool
    rtsp_host: str
    rtsp_port: int
    rtp_base_port: int
    active_sessions: int
    active_receivers: int


class SessionInfo(BaseModel):
    """Audio session information."""
    session_id: str
    client_address: tuple
    state: str
    codec: Optional[str]
    sample_rate: Optional[int]
    duration: float
    packets_received: int
    packets_lost: int
    bytes_received: int


class RecordingInfo(BaseModel):
    """Audio recording information."""
    filename: str
    session_id: str
    codec: str
    sample_rate: int
    channels: int
    duration: float
    size_bytes: int
    created_at: float


# Endpoints

@router.get("/status", response_model=ServerStatus)
async def get_status():
    """Get audio server status."""
    server = get_audio_server()
    status = server.get_status()

    return ServerStatus(
        running=status["running"],
        rtsp_host=status["rtsp_host"],
        rtsp_port=status["rtsp_port"],
        rtp_base_port=status["rtp_base_port"],
        active_sessions=status["active_sessions"],
        active_receivers=status["active_receivers"]
    )


@router.post("/start")
async def start_server():
    """Start audio server."""
    server = get_audio_server()

    if server.get_status()["running"]:
        return {"message": "Server already running"}

    try:
        server.start()
        return {"message": "Server started successfully", "status": server.get_status()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start server: {str(e)}")


@router.post("/stop")
async def stop_server():
    """Stop audio server."""
    server = get_audio_server()

    if not server.get_status()["running"]:
        return {"message": "Server not running"}

    try:
        server.stop()
        return {"message": "Server stopped successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop server: {str(e)}")


@router.get("/sessions", response_model=List[SessionInfo])
async def get_sessions():
    """Get list of active sessions."""
    server = get_audio_server()
    sessions = server.session_manager.get_active_sessions()

    result = []
    for session_id, session in sessions.items():
        result.append(SessionInfo(
            session_id=session.session_id,
            client_address=session.client_address,
            state=session.state.value,
            codec=session.codec,
            sample_rate=session.sample_rate,
            duration=session.last_activity - session.created_at,
            packets_received=session.packets_received,
            packets_lost=session.packets_lost,
            bytes_received=session.bytes_received
        ))

    return result


@router.get("/recordings", response_model=List[RecordingInfo])
async def get_recordings():
    """Get list of audio recordings."""
    server = get_audio_server()
    storage_path = server.storage_path

    if not os.path.exists(storage_path):
        return []

    recordings = []

    for filename in os.listdir(storage_path):
        if not filename.endswith(".wav"):
            continue

        # Read metadata JSON
        json_path = os.path.join(storage_path, filename.replace(".wav", ".json"))
        if not os.path.exists(json_path):
            continue

        try:
            with open(json_path, "r") as f:
                metadata = json.load(f)

            # Get file size
            wav_path = os.path.join(storage_path, filename)
            size_bytes = os.path.getsize(wav_path)

            recordings.append(RecordingInfo(
                filename=filename,
                session_id=metadata["session_id"],
                codec=metadata["codec"],
                sample_rate=metadata["sample_rate"],
                channels=metadata["channels"],
                duration=metadata.get("duration", 0.0),
                size_bytes=size_bytes,
                created_at=metadata["start_time"]
            ))

        except Exception as e:
            print(f"[API] Failed to read metadata for {filename}: {e}")
            continue

    # Sort by created_at (newest first)
    recordings.sort(key=lambda x: x.created_at, reverse=True)

    return recordings


@router.get("/recordings/{filename}")
async def download_recording(filename: str):
    """Download audio recording file."""
    server = get_audio_server()
    filepath = os.path.join(server.storage_path, filename)

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Recording not found")

    if not filepath.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Invalid file type")

    return FileResponse(
        filepath,
        media_type="audio/wav",
        filename=filename
    )


@router.delete("/recordings/{filename}")
async def delete_recording(filename: str):
    """Delete audio recording file and metadata."""
    server = get_audio_server()
    wav_path = os.path.join(server.storage_path, filename)
    json_path = wav_path.replace(".wav", ".json")

    if not os.path.exists(wav_path):
        raise HTTPException(status_code=404, detail="Recording not found")

    try:
        # Delete WAV file
        os.unlink(wav_path)

        # Delete JSON metadata
        if os.path.exists(json_path):
            os.unlink(json_path)

        return {"message": f"Deleted {filename}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete recording: {str(e)}")
