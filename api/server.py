from fastapi import FastAPI
from contextlib import asynccontextmanager
from api.routes import router as api_router

# ReID store wiring
from services.reid.reid_store import ReIDStore
from api.routes import tracks as tracks_route_module
from api.routes import stream as stream_route_module
from api.routes import webrtc as webrtc_route_module
from api.routes import perf as perf_route_module
from api.routes import audio as audio_route_module
from api.routes import ptz as ptz_route_module
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
    Handles startup (starting AI pipeline workers) and shutdown (stopping workers gracefully).
    """
    # Startup: Start the AI pipeline if requested
    global _runner_stop_event, _runner_thread

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

    # Shutdown: Stop the AI pipeline and wait for clean exit
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


app = FastAPI(lifespan=lifespan)
app.include_router(api_router)

# create a global ReIDStore instance and wire it into the tracks router
reid_store = ReIDStore()
tracks_route_module.set_reid_store(reid_store)
# include the tracks router under /api/tracks
app.include_router(tracks_route_module.router, prefix="/api/tracks")
app.include_router(stream_route_module.router, prefix="/api/stream")
app.include_router(webrtc_route_module.router, prefix="/api/webrtc")
app.include_router(perf_route_module.router, prefix="/api/status")
app.include_router(audio_route_module.router, prefix="/api/audio", tags=["audio"])
app.include_router(ptz_route_module.router, prefix="/api/ptz", tags=["ptz"])
