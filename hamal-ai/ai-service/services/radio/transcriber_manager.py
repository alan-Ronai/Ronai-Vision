"""Global Transcriber Manager - Singleton transcribers shared across the application.

This module provides centralized access to transcriber instances, ensuring:
1. Models are loaded ONCE at startup
2. Both file upload and live radio use the SAME instances
3. Semaphore prevents concurrent Whisper processing (CPU-intensive)
4. Proper statistics tracking across all uses
"""

import os
import asyncio
import logging
import time
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# Global singleton instances
_gemini_transcriber = None
_whisper_transcriber = None
_initialized = False

# Semaphore to prevent concurrent Whisper processing (it's CPU-intensive)
_whisper_semaphore: Optional[asyncio.Semaphore] = None

# Global statistics
_stats = {
    "gemini": {
        "file_transcriptions": 0,
        "live_transcriptions": 0,
        "total_processing_time_ms": 0,
        "errors": 0,
        "last_processing_time_ms": 0,
        "avg_processing_time_ms": 0,
    },
    "whisper": {
        "file_transcriptions": 0,
        "live_transcriptions": 0,
        "total_processing_time_ms": 0,
        "errors": 0,
        "last_processing_time_ms": 0,
        "avg_processing_time_ms": 0,
        "model_load_time_ms": 0,
        "queue_wait_count": 0,  # Times had to wait for semaphore
    },
    "initialized_at": None,
}


def init_transcribers(
    whisper_model_path: str = "models/whisper-large-v3-turbo-ct2",
    save_audio: bool = False,
    preload_whisper: bool = True,
) -> bool:
    """Initialize global transcriber instances at startup.

    Should be called once during application startup.

    Args:
        whisper_model_path: Path to Whisper CT2 model
        save_audio: Whether to save audio files for debugging
        preload_whisper: Whether to pre-load Whisper model (recommended)

    Returns:
        True if initialization successful
    """
    global \
        _gemini_transcriber, \
        _whisper_transcriber, \
        _initialized, \
        _whisper_semaphore, \
        _stats

    if _initialized:
        logger.warning("Transcribers already initialized, skipping")
        return True

    logger.info("=" * 60)
    logger.info("📻 Initializing Global Transcriber Manager")
    logger.info("=" * 60)

    try:
        from .gemini_transcriber import GeminiTranscriber, StreamingGeminiTranscriber
        from .whisper_transcriber import WhisperTranscriber

        # Initialize Gemini transcriber (fast, cloud-based)
        _gemini_transcriber = GeminiTranscriber(save_audio=save_audio)
        gemini_status = (
            "✅ configured"
            if _gemini_transcriber.is_configured()
            else "❌ not configured (missing GEMINI_API_KEY)"
        )
        logger.info(f"  Gemini Transcriber: {gemini_status}")

        # Initialize Whisper transcriber (slow, local)
        _whisper_transcriber = WhisperTranscriber(
            model_path=whisper_model_path,
            device="auto",
            compute_type="int8",
            save_audio=save_audio,
        )
        whisper_status = (
            "✅ configured"
            if _whisper_transcriber.is_configured()
            else "❌ not configured (model not found)"
        )
        logger.info(f"  Whisper Transcriber: {whisper_status}")
        logger.info(f"    Model path: {whisper_model_path}")
        logger.info(f"    Device: {_whisper_transcriber.device}")

        # Note: Whisper model is now loaded automatically in __init__
        # Just record the load time if available
        if _whisper_transcriber.is_configured() and _whisper_transcriber._initialized:
            logger.info("  Whisper model: already loaded at initialization")

        # Initialize semaphore (allow only 1 concurrent Whisper transcription)
        _whisper_semaphore = asyncio.Semaphore(1)

        _initialized = True
        _stats["initialized_at"] = datetime.now().isoformat()

        logger.info("=" * 60)
        logger.info("✅ Global Transcriber Manager initialized")
        logger.info("=" * 60)

        return True

    except Exception as e:
        logger.error(f"Failed to initialize transcribers: {e}", exc_info=True)
        return False


def get_gemini_transcriber():
    """Get the global Gemini transcriber instance."""
    global _gemini_transcriber
    if not _initialized:
        logger.warning("Transcribers not initialized, call init_transcribers() first")
    return _gemini_transcriber


def get_whisper_transcriber():
    """Get the global Whisper transcriber instance."""
    global _whisper_transcriber
    if not _initialized:
        logger.warning("Transcribers not initialized, call init_transcribers() first")
    return _whisper_transcriber


def get_whisper_semaphore() -> Optional[asyncio.Semaphore]:
    """Get the Whisper processing semaphore."""
    return _whisper_semaphore


def is_initialized() -> bool:
    """Check if transcribers are initialized."""
    return _initialized


def record_transcription_stats(
    transcriber_type: str,
    source: str,  # "file" or "live"
    processing_time_ms: float,
    success: bool = True,
):
    """Record statistics for a transcription operation.

    Args:
        transcriber_type: "gemini" or "whisper"
        source: "file" for file upload, "live" for live radio
        processing_time_ms: Time taken to process
        success: Whether transcription succeeded
    """
    global _stats

    if transcriber_type not in _stats:
        return

    stats = _stats[transcriber_type]

    if success:
        if source == "file":
            stats["file_transcriptions"] += 1
        else:
            stats["live_transcriptions"] += 1

        stats["total_processing_time_ms"] += processing_time_ms
        stats["last_processing_time_ms"] = processing_time_ms

        total_count = stats["file_transcriptions"] + stats["live_transcriptions"]
        if total_count > 0:
            stats["avg_processing_time_ms"] = (
                stats["total_processing_time_ms"] / total_count
            )
    else:
        stats["errors"] += 1


def record_whisper_queue_wait():
    """Record that a request had to wait for Whisper semaphore."""
    global _stats
    _stats["whisper"]["queue_wait_count"] += 1


def get_transcriber_stats() -> Dict[str, Any]:
    """Get comprehensive transcriber statistics."""
    global _stats, _gemini_transcriber, _whisper_transcriber, _initialized

    result = {
        "initialized": _initialized,
        **_stats,
    }

    # Add instance-specific stats
    if _gemini_transcriber:
        result["gemini"]["instance_stats"] = _gemini_transcriber.get_stats()
        result["gemini"]["configured"] = _gemini_transcriber.is_configured()

    if _whisper_transcriber:
        result["whisper"]["instance_stats"] = _whisper_transcriber.get_stats()
        result["whisper"]["configured"] = _whisper_transcriber.is_configured()
        result["whisper"]["model_loaded"] = _whisper_transcriber._initialized

    return result
