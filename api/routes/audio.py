"""API endpoints for RTP/RTSP audio server control and monitoring."""

import os
import json
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Optional

from services.audio import RTPAudioServer
from services.audio.audio_pipeline import AudioPipeline
from services.audio.transcriber import HebrewTranscriber
from services.audio.command_processor import CommandProcessor
from services.audio.tts import HebrewTTS
from services.audio.unified_receiver import UnifiedAudioReceiver, AudioProtocol


router = APIRouter()

# Global audio server instance
_audio_server: Optional[RTPAudioServer] = None

# Global audio pipeline instance
_audio_pipeline: Optional[AudioPipeline] = None

# Global unified receiver instance
_unified_receiver: Optional[UnifiedAudioReceiver] = None


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
            jitter_buffer_ms=rtp_config.get("jitter_buffer_ms", 100),
        )


def get_audio_pipeline() -> AudioPipeline:
    """Get or create audio pipeline instance."""
    global _audio_pipeline

    if _audio_pipeline is None:
        server = get_audio_server()
        _audio_pipeline = AudioPipeline(rtp_server=server, enable_auto_response=True)

    return _audio_pipeline


def get_unified_receiver() -> UnifiedAudioReceiver:
    """Get or create unified audio receiver instance."""
    global _unified_receiver

    if _unified_receiver is None:
        pipeline = get_audio_pipeline()
        _unified_receiver = UnifiedAudioReceiver(audio_pipeline=pipeline)

    return _unified_receiver


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
        active_receivers=status["active_receivers"],
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
        result.append(
            SessionInfo(
                session_id=session.session_id,
                client_address=session.client_address,
                state=session.state.value,
                codec=session.codec,
                sample_rate=session.sample_rate,
                duration=session.last_activity - session.created_at,
                packets_received=session.packets_received,
                packets_lost=session.packets_lost,
                bytes_received=session.bytes_received,
            )
        )

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

            recordings.append(
                RecordingInfo(
                    filename=filename,
                    session_id=metadata["session_id"],
                    codec=metadata["codec"],
                    sample_rate=metadata["sample_rate"],
                    channels=metadata["channels"],
                    duration=metadata.get("duration", 0.0),
                    size_bytes=size_bytes,
                    created_at=metadata["start_time"],
                )
            )

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

    return FileResponse(filepath, media_type="audio/wav", filename=filename)


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
        raise HTTPException(
            status_code=500, detail=f"Failed to delete recording: {str(e)}"
        )


# ============================================================================
# TRANSCRIPTION & COMMAND PROCESSING ENDPOINTS
# ============================================================================


class TranscriptionControl(BaseModel):
    """Transcription control request."""

    enabled: bool = True


class SendTextRequest(BaseModel):
    """Send text as speech request."""

    session_id: str
    text: str


class SendAudioRequest(BaseModel):
    """Send audio file request."""

    session_id: str


class PipelineStats(BaseModel):
    """Audio pipeline statistics."""

    running: bool
    chunks_processed: int
    transcriptions: int
    commands_detected: int
    responses_sent: int
    queue_size: int
    transcriber_ready: bool
    tts_ready: bool


class CommandInfo(BaseModel):
    """Command information."""

    command_id: str
    keyword: str
    text: str
    confidence: float
    timestamp: str
    parameters: Dict


@router.post("/transcription/start")
async def start_transcription():
    """Start audio transcription pipeline."""
    pipeline = get_audio_pipeline()

    if pipeline._running:
        return {"message": "Transcription already running"}

    pipeline.start()
    return {"message": "Transcription started"}


@router.post("/transcription/stop")
async def stop_transcription():
    """Stop audio transcription pipeline."""
    pipeline = get_audio_pipeline()

    if not pipeline._running:
        return {"message": "Transcription not running"}

    pipeline.stop()
    return {"message": "Transcription stopped"}


@router.get("/transcription/status", response_model=PipelineStats)
async def get_transcription_status():
    """Get transcription pipeline status and statistics."""
    pipeline = get_audio_pipeline()
    stats = pipeline.get_stats()

    return PipelineStats(**stats)


@router.get("/commands/history")
async def get_command_history(limit: int = 10) -> List[CommandInfo]:
    """Get recent command history.

    Args:
        limit: Maximum number of commands to return (default 10)
    """
    pipeline = get_audio_pipeline()
    history = pipeline.command_processor.get_history(limit=limit)

    return [
        CommandInfo(
            command_id=cmd.command_id,
            keyword=cmd.keyword,
            text=cmd.text,
            confidence=cmd.confidence,
            timestamp=cmd.timestamp.isoformat(),
            parameters=cmd.parameters,
        )
        for cmd in history
    ]


@router.delete("/commands/history")
async def clear_command_history():
    """Clear command history."""
    pipeline = get_audio_pipeline()
    pipeline.command_processor.clear_history()
    return {"message": "Command history cleared"}


@router.post("/speak")
async def send_text_as_speech(request: SendTextRequest):
    """Convert Hebrew text to speech and send to field device.

    Args:
        request: Contains session_id and Hebrew text
    """
    pipeline = get_audio_pipeline()

    # Validate session exists
    server = get_audio_server()
    if request.session_id not in server.get_active_sessions():
        raise HTTPException(
            status_code=404, detail=f"Session {request.session_id} not found"
        )

    # Send text as speech
    success = pipeline.send_text_response(request.session_id, request.text)

    if not success:
        raise HTTPException(status_code=500, detail="Failed to send speech")

    return {
        "message": "Speech sent successfully",
        "session_id": request.session_id,
        "text": request.text,
    }


@router.post("/send-audio/{session_id}")
async def send_audio_file(session_id: str, audio_file: UploadFile = File(...)):
    """Send pre-recorded audio file to field device.

    Args:
        session_id: Target session identifier
        audio_file: Audio file (WAV format)
    """
    pipeline = get_audio_pipeline()

    # Validate session exists
    server = get_audio_server()
    if session_id not in server.get_active_sessions():
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    # Validate file type
    if not audio_file.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only WAV files are supported")

    try:
        # Load audio file
        import soundfile as sf
        import io

        content = await audio_file.read()
        audio, sample_rate = sf.read(io.BytesIO(content))

        # Convert to int16 if needed
        if audio.dtype in (np.float32, np.float64):
            audio = (audio * 32767).astype(np.int16)

        # Send audio
        success = pipeline.send_audio_response(session_id, audio, sample_rate)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to send audio")

        return {
            "message": "Audio sent successfully",
            "session_id": session_id,
            "filename": audio_file.filename,
            "samples": len(audio),
            "sample_rate": sample_rate,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to process audio file: {str(e)}"
        )


@router.get("/sessions")
async def list_active_sessions() -> List[str]:
    """Get list of active RTP session IDs."""
    server = get_audio_server()
    return server.get_active_sessions()


# ============================================================================
# UNIFIED RECEIVER & MULTI-PROTOCOL SUPPORT
# ============================================================================


class ProtocolConfig(BaseModel):
    """Protocol configuration."""

    host: Optional[str] = "0.0.0.0"
    port: Optional[int] = None


class RawRTPConfig(BaseModel):
    """Raw RTP receiver configuration."""

    receiver_id: str
    listen_port: int
    codec: str = "g711_ulaw"


class FFmpegConfig(BaseModel):
    """FFmpeg receiver configuration."""

    receiver_id: str
    input_url: str
    sample_rate: int = 16000


class GStreamerConfig(BaseModel):
    """GStreamer receiver configuration."""

    receiver_id: str
    pipeline_string: Optional[str] = None
    input_url: Optional[str] = None
    sample_rate: int = 16000


@router.get("/protocols/status")
async def get_protocols_status():
    """Get status of all audio protocols."""
    receiver = get_unified_receiver()
    return receiver.get_status()


@router.post("/protocols/rtsp/enable")
async def enable_rtsp(config: Optional[ProtocolConfig] = None):
    """Enable RTSP audio server."""
    receiver = get_unified_receiver()

    if config:
        receiver.enable_rtsp(
            rtsp_host=config.host or "0.0.0.0", rtsp_port=config.port or 8554
        )
    else:
        receiver.enable_rtsp()

    return {"message": "RTSP enabled"}


@router.post("/protocols/rtsp/disable")
async def disable_rtsp():
    """Disable RTSP audio server."""
    receiver = get_unified_receiver()
    receiver.disable_rtsp()
    return {"message": "RTSP disabled"}


@router.post("/protocols/sip/enable")
async def enable_sip(config: Optional[ProtocolConfig] = None):
    """Enable SIP + RTP server."""
    receiver = get_unified_receiver()

    if config:
        receiver.enable_sip(
            sip_host=config.host or "0.0.0.0", sip_port=config.port or 5060
        )
    else:
        receiver.enable_sip()

    return {"message": "SIP enabled"}


@router.post("/protocols/sip/disable")
async def disable_sip():
    """Disable SIP server."""
    receiver = get_unified_receiver()
    receiver.disable_sip()
    return {"message": "SIP disabled"}


@router.post("/receivers/raw-rtp/add")
async def add_raw_rtp_receiver(config: RawRTPConfig):
    """Add raw RTP/UDP receiver."""
    receiver = get_unified_receiver()

    receiver_id = receiver.add_raw_rtp_receiver(
        receiver_id=config.receiver_id,
        listen_port=config.listen_port,
        codec=config.codec,
    )

    return {
        "message": "Raw RTP receiver added",
        "receiver_id": receiver_id,
        "listen_port": config.listen_port,
    }


@router.delete("/receivers/raw-rtp/{receiver_id}")
async def remove_raw_rtp_receiver(receiver_id: str):
    """Remove raw RTP receiver."""
    receiver = get_unified_receiver()
    receiver.remove_raw_rtp_receiver(receiver_id)
    return {"message": f"Receiver {receiver_id} removed"}


@router.post("/receivers/ffmpeg/add")
async def add_ffmpeg_receiver(config: FFmpegConfig):
    """Add FFmpeg audio receiver."""
    receiver = get_unified_receiver()

    receiver_id = receiver.add_ffmpeg_receiver(
        receiver_id=config.receiver_id,
        input_url=config.input_url,
        sample_rate=config.sample_rate,
    )

    return {
        "message": "FFmpeg receiver added",
        "receiver_id": receiver_id,
        "input_url": config.input_url,
    }


@router.delete("/receivers/ffmpeg/{receiver_id}")
async def remove_ffmpeg_receiver(receiver_id: str):
    """Remove FFmpeg receiver."""
    receiver = get_unified_receiver()
    receiver.remove_ffmpeg_receiver(receiver_id)
    return {"message": f"Receiver {receiver_id} removed"}


@router.post("/receivers/gstreamer/add")
async def add_gstreamer_receiver(config: GStreamerConfig):
    """Add GStreamer audio receiver."""
    receiver = get_unified_receiver()

    receiver_id = receiver.add_gstreamer_receiver(
        receiver_id=config.receiver_id,
        pipeline_string=config.pipeline_string,
        input_url=config.input_url,
        sample_rate=config.sample_rate,
    )

    return {"message": "GStreamer receiver added", "receiver_id": receiver_id}


@router.delete("/receivers/gstreamer/{receiver_id}")
async def remove_gstreamer_receiver(receiver_id: str):
    """Remove GStreamer receiver."""
    receiver = get_unified_receiver()
    receiver.remove_gstreamer_receiver(receiver_id)
    return {"message": f"Receiver {receiver_id} removed"}


@router.post("/receivers/stop-all")
async def stop_all_receivers():
    """Stop all audio receivers."""
    receiver = get_unified_receiver()
    receiver.stop_all()
    return {"message": "All receivers stopped"}


# ============================================================================
# TTS (Text-to-Speech) ENDPOINTS
# ============================================================================

# Global TTS instance
_tts_engine: Optional[HebrewTTS] = None


def get_tts_engine() -> HebrewTTS:
    """Get or create TTS engine instance."""
    global _tts_engine
    if _tts_engine is None:
        _tts_engine = HebrewTTS(sample_rate=16000, engine="pyttsx3")
    return _tts_engine


class TTSRequest(BaseModel):
    """TTS synthesis request."""

    text: str
    save_file: bool = False  # Save to file in addition to returning audio


class TTSResponse(BaseModel):
    """TTS synthesis response."""

    success: bool
    message: str
    audio_file: Optional[str] = None  # Path to saved audio file
    sample_rate: int
    duration: Optional[float] = None


@router.post("/tts/synthesize", response_model=TTSResponse)
async def synthesize_text_to_speech(request: TTSRequest):
    """Synthesize Hebrew text to speech.

    Args:
        request: TTS request with text

    Returns:
        TTS response with audio file path if saved
    """
    try:
        tts = get_tts_engine()

        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Text is required")

        # Generate audio
        audio = tts.synthesize(request.text)

        if audio is None:
            raise HTTPException(status_code=500, detail="TTS synthesis failed")

        # Calculate duration
        duration = len(audio) / tts.sample_rate

        # Optionally save to file
        audio_file = None
        if request.save_file:
            import time
            timestamp = int(time.time())
            output_dir = "output/tts"
            os.makedirs(output_dir, exist_ok=True)
            audio_file = f"{output_dir}/tts_{timestamp}.wav"

            # Save using soundfile
            import soundfile as sf
            sf.write(audio_file, audio, tts.sample_rate)

        return TTSResponse(
            success=True,
            message="Speech synthesized successfully",
            audio_file=audio_file,
            sample_rate=tts.sample_rate,
            duration=duration,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS error: {str(e)}")


@router.post("/tts/speak-and-stream")
async def speak_and_stream_to_rtp(text: str, session_id: str):
    """Synthesize text and stream to active RTP session.

    Args:
        text: Hebrew text to speak
        session_id: RTP session ID to stream to

    Returns:
        Status of streaming operation
    """
    try:
        tts = get_tts_engine()
        server = get_audio_server()

        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Text is required")

        # Check if session exists
        session = server.session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        # Generate audio with RTP codec
        result = tts.text_to_rtp_payload(text, codec="g711_ulaw")

        if result is None:
            raise HTTPException(status_code=500, detail="TTS synthesis failed")

        audio_bytes, sample_rate = result

        # Stream to session
        # Note: This is a simplified implementation
        # In production, you'd want to chunk the audio and send as RTP packets
        # with proper timing and sequence numbers

        return {
            "success": True,
            "message": f"Speech synthesized and ready to stream to session {session_id}",
            "audio_size_bytes": len(audio_bytes),
            "sample_rate": sample_rate,
            "note": "Full RTP streaming implementation requires additional packet handling",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stream error: {str(e)}")


@router.get("/tts/status")
async def get_tts_status():
    """Get TTS engine status."""
    try:
        tts = get_tts_engine()
        return {
            "initialized": tts.is_ready(),
            "engine": tts.engine_name,
            "sample_rate": tts.sample_rate,
        }
    except Exception as e:
        return {
            "initialized": False,
            "error": str(e),
        }
