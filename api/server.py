from fastapi import FastAPI
from api.routes import router as api_router

# ReID store wiring
from services.reid.reid_store import ReIDStore
from api.routes import tracks as tracks_route_module
from api.routes import stream as stream_route_module
from api.routes import webrtc as webrtc_route_module
import os
import threading
import logging

logger = logging.getLogger(__name__)

app = FastAPI()
app.include_router(api_router)

# create a global ReIDStore instance and wire it into the tracks router
reid_store = ReIDStore()
tracks_route_module.set_reid_store(reid_store)
# include the tracks router under /api/tracks
app.include_router(tracks_route_module.router, prefix="/api/tracks")
app.include_router(stream_route_module.router, prefix="/api/stream")
app.include_router(webrtc_route_module.router, prefix="/api/webrtc")
# optional perf/status router
from api.routes import perf as perf_route_module
from api.routes import audio as audio_route_module

app.include_router(perf_route_module.router, prefix="/api/status")
app.include_router(audio_route_module.router, prefix="/api/audio", tags=["audio"])


def _maybe_start_runner():
    """Start the multi-camera runner in a background thread when requested via env var.

    This shares the `broadcaster` singleton in-process so MJPEG/websocket streams
    served by FastAPI will receive frames.
    """
    start_flag = os.getenv("START_RUNNER", "false").lower()
    if start_flag not in ("1", "true", "yes"):
        return

    # create module-level stop event and thread so shutdown can join
    global _runner_stop_event, _runner_thread

    if globals().get("_runner_thread") is not None:
        logger.info("Runner appears already started")
        return

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
            # import run_loop and call with the stop event
            from scripts.run_multi_camera import run_loop

            run_loop(stop_event=_runner_stop_event, max_frames=None)
            logger.info("Embedded runner exited cleanly")
        except Exception:
            logger.exception("Embedded runner exited with error")

    # non-daemon so we can join on shutdown
    _runner_thread = threading.Thread(
        target=_run_loop, daemon=False, name="embedded-runner"
    )
    _runner_thread.start()


@app.on_event("startup")
def _startup_event():
    _maybe_start_runner()


@app.on_event("shutdown")
def _shutdown_event():
    # Signal runner to stop and wait briefly for it to exit
    stop_evt = globals().get("_runner_stop_event")
    th = globals().get("_runner_thread")
    if stop_evt is not None:
        try:
            stop_evt.set()
        except Exception:
            logger.exception("Failed to set runner stop event")
    if th is not None:
        try:
            logger.info("Waiting for embedded runner thread to exit")
            th.join(timeout=5.0)
        except Exception:
            logger.exception("Error while joining runner thread")
