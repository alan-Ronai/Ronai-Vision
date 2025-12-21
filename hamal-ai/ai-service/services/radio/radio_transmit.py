"""Radio Transmission API Routes.

FastAPI routes for sending audio to the radio system via RTP.
Supports:
- Direct audio transmission (base64 encoded)
- File upload transmission
- PTT signaling
- Transmission status/stats
"""

import base64
import io
import logging
import os
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from .rtp_tcp_sender import get_sender, send_audio_to_radio

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/radio/transmit", tags=["radio-transmit"])


class TransmitRequest(BaseModel):
    """Request to transmit audio."""
    audio_base64: str  # Base64 encoded 16-bit PCM audio
    sample_rate: int = 16000
    auto_ptt: bool = True


class PTTRequest(BaseModel):
    """Request to send PTT signal."""
    state: str  # 'START' or 'STOP'


class TransmitResponse(BaseModel):
    """Response from transmission."""
    success: bool
    message: str
    packets_sent: Optional[int] = None
    bytes_sent: Optional[int] = None


class StatsResponse(BaseModel):
    """Transmission statistics."""
    connected: bool
    host: str
    port: int
    sample_rate: int
    packets_sent: int
    bytes_sent: int
    ptt_signals: int
    errors: int
    last_send_time: Optional[float] = None


@router.post("/audio", response_model=TransmitResponse)
async def transmit_audio(request: TransmitRequest):
    """Transmit base64-encoded audio to the radio.

    The audio should be 16-bit PCM format.
    """
    try:
        # Decode base64 audio
        audio_data = base64.b64decode(request.audio_base64)

        if len(audio_data) == 0:
            raise HTTPException(status_code=400, detail="Empty audio data")

        logger.info(
            f"📤 Transmitting {len(audio_data)} bytes of audio "
            f"@ {request.sample_rate}Hz, PTT={request.auto_ptt}"
        )

        # Send to radio
        success = send_audio_to_radio(
            audio_data=audio_data,
            sample_rate=request.sample_rate,
            auto_ptt=request.auto_ptt
        )

        if success:
            sender = get_sender()
            stats = sender.get_stats()
            return TransmitResponse(
                success=True,
                message="Audio transmitted successfully",
                packets_sent=stats.get("packets_sent"),
                bytes_sent=stats.get("bytes_sent")
            )
        else:
            return TransmitResponse(
                success=False,
                message="Failed to transmit audio"
            )

    except Exception as e:
        logger.error(f"Transmission error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/file", response_model=TransmitResponse)
async def transmit_file(
    file: UploadFile = File(...),
    sample_rate: int = Form(16000),
    auto_ptt: bool = Form(True)
):
    """Transmit an uploaded audio file to the radio.

    Supports:
    - WAV files (will extract PCM data)
    - Raw PCM files
    """
    try:
        # Read file content
        content = await file.read()

        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty file")

        logger.info(
            f"📤 Transmitting file '{file.filename}' "
            f"({len(content)} bytes) @ {sample_rate}Hz"
        )

        # Check if it's a WAV file
        audio_data = content
        actual_sample_rate = sample_rate

        if content[:4] == b'RIFF' and content[8:12] == b'WAVE':
            # Parse WAV header to extract PCM data
            audio_data, actual_sample_rate = _parse_wav(content)
            logger.info(f"Parsed WAV: {len(audio_data)} bytes @ {actual_sample_rate}Hz")

        # Send to radio
        success = send_audio_to_radio(
            audio_data=audio_data,
            sample_rate=actual_sample_rate,
            auto_ptt=auto_ptt
        )

        if success:
            sender = get_sender()
            stats = sender.get_stats()
            return TransmitResponse(
                success=True,
                message=f"File '{file.filename}' transmitted successfully",
                packets_sent=stats.get("packets_sent"),
                bytes_sent=stats.get("bytes_sent")
            )
        else:
            return TransmitResponse(
                success=False,
                message="Failed to transmit file"
            )

    except Exception as e:
        logger.error(f"File transmission error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ptt", response_model=TransmitResponse)
async def send_ptt(request: PTTRequest):
    """Send a PTT (Push-to-Talk) signal.

    Used to manually control PTT when streaming audio.
    """
    try:
        state = request.state.upper()
        if state not in ("START", "STOP"):
            raise HTTPException(
                status_code=400,
                detail="State must be 'START' or 'STOP'"
            )

        sender = get_sender()
        success = sender.send_ptt_signal(state)

        return TransmitResponse(
            success=success,
            message=f"PTT {state} signal {'sent' if success else 'failed'}"
        )

    except Exception as e:
        logger.error(f"PTT signal error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=StatsResponse)
async def get_transmission_stats():
    """Get transmission statistics."""
    try:
        sender = get_sender()
        stats = sender.get_stats()

        # Check both TCP command connection and UDP audio readiness
        cmd_connected = stats.get("cmd_connected", False)
        audio_ready = stats.get("audio_ready", False)
        # Consider connected if either TCP is connected or audio is ready
        # (UDP doesn't have a connection state, so audio_ready means socket is created)
        connected = cmd_connected or audio_ready

        return StatsResponse(
            connected=connected,
            host=stats.get("host", ""),
            port=stats.get("cmd_port", stats.get("port", 0)),
            sample_rate=stats.get("sample_rate", 16000),
            packets_sent=stats.get("packets_sent", 0),
            bytes_sent=stats.get("bytes_sent", 0),
            ptt_signals=stats.get("ptt_signals", 0),
            errors=stats.get("errors", 0),
            last_send_time=stats.get("last_send_time")
        )

    except Exception as e:
        logger.error(f"Stats error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/connect", response_model=TransmitResponse)
async def connect_sender():
    """Manually connect to the transmission server."""
    try:
        sender = get_sender()
        success = sender.connect()

        return TransmitResponse(
            success=success,
            message="Connected to TX server" if success else "Failed to connect"
        )

    except Exception as e:
        logger.error(f"Connect error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/disconnect", response_model=TransmitResponse)
async def disconnect_sender():
    """Disconnect from the transmission server."""
    try:
        sender = get_sender()
        sender.disconnect()

        return TransmitResponse(
            success=True,
            message="Disconnected from TX server"
        )

    except Exception as e:
        logger.error(f"Disconnect error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def _parse_wav(data: bytes) -> tuple[bytes, int]:
    """Parse WAV file and extract PCM data.

    Args:
        data: Raw WAV file bytes

    Returns:
        Tuple of (pcm_data, sample_rate)
    """
    import struct

    # Verify RIFF header
    if data[:4] != b'RIFF' or data[8:12] != b'WAVE':
        raise ValueError("Not a valid WAV file")

    # Find fmt chunk
    pos = 12
    sample_rate = 16000
    bits_per_sample = 16
    num_channels = 1

    while pos < len(data) - 8:
        chunk_id = data[pos:pos + 4]
        chunk_size = struct.unpack("<I", data[pos + 4:pos + 8])[0]

        if chunk_id == b'fmt ':
            # Parse format chunk
            audio_format = struct.unpack("<H", data[pos + 8:pos + 10])[0]
            num_channels = struct.unpack("<H", data[pos + 10:pos + 12])[0]
            sample_rate = struct.unpack("<I", data[pos + 12:pos + 16])[0]
            bits_per_sample = struct.unpack("<H", data[pos + 22:pos + 24])[0]

            logger.debug(
                f"WAV format: {audio_format}, channels={num_channels}, "
                f"rate={sample_rate}, bits={bits_per_sample}"
            )

        elif chunk_id == b'data':
            # Found data chunk - extract PCM
            pcm_data = data[pos + 8:pos + 8 + chunk_size]

            # Convert stereo to mono if needed
            if num_channels == 2:
                pcm_data = _stereo_to_mono(pcm_data, bits_per_sample)

            return pcm_data, sample_rate

        pos += 8 + chunk_size
        # Align to word boundary
        if chunk_size % 2 == 1:
            pos += 1

    raise ValueError("No data chunk found in WAV file")


def _stereo_to_mono(data: bytes, bits_per_sample: int = 16) -> bytes:
    """Convert stereo PCM to mono by averaging channels."""
    import numpy as np

    if bits_per_sample == 16:
        dtype = np.int16
    elif bits_per_sample == 8:
        dtype = np.uint8
    else:
        dtype = np.int32

    samples = np.frombuffer(data, dtype=dtype)

    # Reshape to (num_samples, 2) and average
    samples = samples.reshape(-1, 2)
    mono = samples.mean(axis=1).astype(dtype)

    return mono.tobytes()


async def transmit_audio_internal(
    audio_data: bytes,
    format: str = "wav",
    priority: str = "normal",
    sample_rate: int = 24000,
    auto_ptt: bool = True
) -> dict:
    """
    Internal function to transmit audio to radio without HTTP.

    Called by TTS service and scenario rule engine to transmit
    generated audio directly.

    Args:
        audio_data: Raw audio bytes (WAV or PCM format)
        format: Audio format - "wav" or "pcm"
        priority: Priority level - "high", "normal", "low"
        sample_rate: Sample rate for PCM data (ignored for WAV)
        auto_ptt: Whether to auto-trigger PTT

    Returns:
        Dict with success status and transmission info
    """
    try:
        if len(audio_data) == 0:
            logger.warning("📻 transmit_audio_internal: Empty audio data")
            return {"success": False, "error": "Empty audio data"}

        pcm_data = audio_data
        actual_sample_rate = sample_rate

        # Parse WAV if needed
        if format == "wav" or (len(audio_data) > 12 and audio_data[:4] == b'RIFF' and audio_data[8:12] == b'WAVE'):
            try:
                pcm_data, actual_sample_rate = _parse_wav(audio_data)
                logger.info(f"📻 Parsed WAV: {len(pcm_data)} bytes PCM @ {actual_sample_rate}Hz")
            except Exception as e:
                logger.error(f"📻 WAV parse error, using raw data: {e}")
                # Skip WAV header if parse failed (assume 44-byte header)
                if len(audio_data) > 44:
                    pcm_data = audio_data[44:]

        logger.info(
            f"📻 transmit_audio_internal: {len(pcm_data)} bytes PCM "
            f"@ {actual_sample_rate}Hz, priority={priority}, PTT={auto_ptt}"
        )

        # Send to radio
        success = send_audio_to_radio(
            audio_data=pcm_data,
            sample_rate=actual_sample_rate,
            auto_ptt=auto_ptt
        )

        if success:
            sender = get_sender()
            stats = sender.get_stats()
            logger.info(f"📻 Audio transmitted successfully: {stats.get('packets_sent')} packets")
            return {
                "success": True,
                "message": "Audio transmitted to radio",
                "packets_sent": stats.get("packets_sent"),
                "bytes_sent": stats.get("bytes_sent")
            }
        else:
            logger.warning("📻 Radio transmission failed")
            return {
                "success": False,
                "error": "Failed to transmit audio to radio"
            }

    except Exception as e:
        logger.error(f"📻 transmit_audio_internal error: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }
