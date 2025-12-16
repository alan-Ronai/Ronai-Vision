from fastapi import FastAPI
from contextlib import asynccontextmanager
from api.routes import router as api_router

# ReID store wiring (now uses singleton)
from api.routes import tracks as tracks_route_module
from api.routes import stream as stream_route_module
from api.routes import webrtc as webrtc_route_module
from api.routes import perf as perf_route_module
from api.routes import audio as audio_route_module
from api.routes import ptz as ptz_route_module
from api.routes import metadata as metadata_route_module
from api.routes import recorder as recorder_route_module
from api.routes import vision_analysis as vision_analysis_module
from api.routes import logs as logs_module
from api.routes import videos as videos_module
import os
import threading
import logging

logger = logging.getLogger(__name__)

# Module-level variables for runner thread management
_runner_stop_event = None
_runner_thread = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI.
    Handles startup (loading persisted data, cleaning up stale tracks) and shutdown (saving data gracefully).
    """
    # Startup: Load persisted metadata and clean up stale tracks
    global _runner_stop_event, _runner_thread

    # Load metadata from persistence
    try:
        from services.tracker.metadata_manager import get_metadata_manager

        manager = get_metadata_manager()
        # Metadata is auto-loaded on manager init, cleanup old entries
        cleaned = manager.cleanup_expired()
        if cleaned > 0:
            logger.info(
                f"[Startup] Cleaned up {cleaned} expired tracks from metadata store"
            )
    except Exception as e:
        logger.warning(f"Failed to load/cleanup metadata on startup: {e}")

    # Reset viewer state on startup
    try:
        from services import viewer_tracker

        viewer_tracker.cleanup()
        print("[Server] Cleared viewer state on startup")
    except Exception as e:
        logger.warning(f"Failed to clear viewer state: {e}")

    start_flag = os.getenv("START_RUNNER", "false").lower()
    if start_flag in ("1", "true", "yes"):
        _runner_stop_event = threading.Event()

        def _run_loop():
            try:
                parallel_flag = os.getenv("PARALLEL", "false").lower() in (
                    "1",
                    "true",
                    "yes",
                )
                logger.info(
                    f"Starting embedded multi-camera runner (background thread, parallel={parallel_flag})"
                )
                from scripts.run_multi_camera import run_loop

                run_loop(stop_event=_runner_stop_event, max_frames=None)
                logger.info("Embedded runner exited cleanly")
            except Exception:
                logger.exception("Embedded runner exited with error")

        # daemon=False so we can properly wait for shutdown
        _runner_thread = threading.Thread(
            target=_run_loop, daemon=False, name="embedded-runner"
        )
        _runner_thread.start()
        logger.info("AI pipeline worker started")

    yield  # Server is running

    # Shutdown: Stop the AI pipeline, save metadata, and decay ReID store
    if _runner_stop_event is not None:
        try:
            logger.info("Signaling AI pipeline to stop...")
            _runner_stop_event.set()
        except Exception:
            logger.exception("Failed to set runner stop event")

    if _runner_thread is not None:
        try:
            logger.info("Waiting for AI pipeline worker to exit (timeout=30s)...")
            _runner_thread.join(timeout=30.0)
            if _runner_thread.is_alive():
                logger.warning("AI pipeline worker did not exit within 30s timeout")
            else:
                logger.info("AI pipeline worker exited cleanly")
        except Exception:
            logger.exception("Error while joining runner thread")

    # Save metadata to persistence file on shutdown
    try:
        from services.tracker.metadata_manager import get_metadata_manager

        manager = get_metadata_manager()
        manager.save_metadata()
        logger.info("[Shutdown] Saved metadata persistence file")
    except Exception as e:
        logger.warning(f"Failed to save metadata on shutdown: {e}")

    # Decay stale tracks in ReID store (mark inactive if not seen for >1 hour)
    # This allows re-identification if camera comes back online
    try:
        from services.reid.reid_store import ReIDStore

        # Access the reid_store instance created below (we'll pass it via a module var)
        # For now, we'll get it from the app state
        if hasattr(app, "state") and hasattr(app.state, "reid_store"):
            store = app.state.reid_store
            removed = store.decay_stale(max_age_seconds=3600)  # 1 hour TTL
            if removed > 0:
                logger.info(
                    f"[Shutdown] Deactivated {removed} stale global IDs in ReID store"
                )
    except Exception as e:
        logger.warning(f"Failed to decay ReID store on shutdown: {e}")


app = FastAPI(lifespan=lifespan)
app.include_router(api_router)

# Use global singleton ReIDStore instance (also used by worker managers)
from services.reid.reid_store import get_reid_store

reid_store = get_reid_store()
app.state.reid_store = reid_store  # Store in app state for lifecycle access
tracks_route_module.set_reid_store(reid_store)
# include the tracks router under /api/tracks
app.include_router(tracks_route_module.router, prefix="/api/tracks")
app.include_router(stream_route_module.router, prefix="/api/stream")
app.include_router(webrtc_route_module.router, prefix="/api/webrtc")
app.include_router(perf_route_module.router, prefix="/api/status")
app.include_router(audio_route_module.router, prefix="/api/audio", tags=["audio"])
app.include_router(ptz_route_module.router, prefix="/api/ptz", tags=["ptz"])
app.include_router(
    metadata_route_module.router
)  # Metadata routes (already has /api/metadata prefix)
app.include_router(
    recorder_route_module.router, prefix="/api/recorder", tags=["recorder"]
)
app.include_router(
    vision_analysis_module.router
)  # Vision routes (already has /api/vision prefix)
app.include_router(logs_module.router)  # Log routes (already has /api/logs prefix)
app.include_router(
    videos_module.router
)  # Video routes (already has /api/videos prefix)
