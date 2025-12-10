# TODO — Ronai-Vision

This file mirrors the project's tracked work items and acts as a readable checklist for contributors.

## Completed

-   [x] Add .gitignore
-   [x] Scaffold minimal modules
-   [x] Run quick smoke test
-   [x] Expand scaffolding (FastAPI routes + WebSocket)
-   [x] Update Dockerfile and README
-   [x] Implement YOLO detector wrapper (version-agnostic)
-   [x] Upgrade YOLO to v12 and integrate models
-   [x] Implement SAM2 segmenter and load checkpoint
-   [x] Move models to `models/` folder
-   [x] Implement CentroidTracker
-   [x] Implement HistogramReID (CPU fallback)
-   [x] Implement output renderer
-   [x] Add camera manager (multi-stream ingestion)
-   [x] Add OSNet ReID wrapper (`services/reid/osnet_reid.py`) — loads local `models/osnet_*.pth` via `torchreid`
-   [x] Integrate BoT-SORT per-camera tracker (`services/tracker/bot_sort.py`)
-   [x] Implement FAISS-based cross-camera ReID store (`services/reid/reid_store.py`)
-   [x] Create plugin registry for additional detectors (`services/detector/plugins.py`)
-   [x] Build multi-camera end-to-end runner and sample outputs (`scripts/run_multi_camera.py`)
-   [x] Add API endpoints to query global track IDs and recent tracks (`/api/tracks`)
-   [x] Add OpenAPI docs and Pydantic models for tracks endpoints
-   [x] Persist latest embedding per `gid` in `ReIDStore` and implement `remove_gid`/`decay_stale`
-   [x] Add unit tests for `ReIDStore` (upsert/query/decay)
-   [x] Parallelize per-camera processing with aggregator and a queue (`PARALLEL=true` mode)
-   [x] Add cooperative `run_loop(stop_event, max_frames)` to `scripts/run_multi_camera.py`
-   [x] Start/stop runner from `api/server.py` cleanly (FastAPI startup/shutdown handlers)
-   [x] Produce clean final frames (only boxes + IDs) and make mask overlay optional (`show_masks` parameter)
-   [x] Testing, validation, and small integration tests for graceful shutdown
-   [x] Performance tuning: model warmup and per-stage profiling metrics (`/api/status/perf`)

## In Progress / Next

-   [ ] Add example detectors for weapons/car-color (placeholders + training recipes)
-   [ ] Write docs + runbook (README updates, runbook for macOS/conda)
-   [ ] Persist full `ReIDStore` state to disk and restore on startup (checkpoint save/load methods)
-   [ ] Add admin endpoint to trigger index rebuild or state restore
-   [ ] Add runtime tuning endpoints for `match_threshold` and `max_history`
-   [ ] Clarify SAM2 prompting options + add optional CLIP/text-filter helper
-   [ ] Performance tuning: advanced optimizations (improved batching, device selection strategies)

## Notes

-   Use `config/dev.env` (or `config/prod.env`) to set `DEVICE=cpu` or `DEVICE=cuda` to control model/device selection.
-   Place large model files in `models/` (gitignored). Example names: `yolo12n.pt`, `sam2_small.pt`, `osnet_x1_0_imagenet.pth`.

## How to contribute

1. Create a branch `feature/<task>` and open a PR.
2. Run unit tests in `tests/` and include a small smoke test for new services.
3. Update this `TODO.md` when tasks are completed.
