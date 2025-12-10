## Issue Fixes Summary (December 10, 2025)

### Issue 1: `/api/status/perf` returns empty metrics

**Root Cause:**
The perf endpoint was importing `RUN_METRICS` at import time, but the dict is only populated when the runner thread starts and calls `run_loop()`. The module-level `RUN_METRICS` dict wasn't being updated correctly across thread boundaries.

**Fix:**

-   Changed perf endpoint to use `getattr(runner_module, "RUN_METRICS", {})` instead of directly importing
-   This ensures it reads the live dict as updated by the runner thread
-   Added a better note when metrics are not yet available
-   Metrics will appear after the runner has processed several frames (checks count > 0)

**Test:**

```bash
START_RUNNER=true uvicorn api.server:app &
sleep 5  # Let runner process some frames
curl http://localhost:8000/api/status/perf
# Should return timing averages instead of empty dict
```

---

### Issue 2: "Address already in use: 127.0.0.1:8000"

**Root Cause:**
When uvicorn crashes or is not properly terminated (e.g., killed with SIGKILL instead of SIGTERM), the process lingers and holds the port.

**Fix:**

-   Created helper script `scripts/kill_port.sh` to forcibly free a port
-   Updated README with clear instructions on how to clean up:

    ```bash
    # Quick kill:
    lsof -i :8000 | grep LISTEN | awk '{print $2}' | xargs kill -9

    # Or use helper:
    bash scripts/kill_port.sh 8000
    ```

-   Improved server shutdown logging (runner now logs when exiting cleanly)

**Test:**

```bash
START_RUNNER=true uvicorn api.server:app &
sleep 2
kill -9 <PID>  # Force kill
bash scripts/kill_port.sh 8000  # Clean up
START_RUNNER=true uvicorn api.server:app  # Restart
```

---

### Issue 3: No logs when using PARALLEL mode

**Root Cause:**
In parallel mode, per-camera workers and the aggregator thread run silently. There are no periodic progress logs, making it appear frozen.

**Fixes Applied:**

-   Added periodic logging in per-camera workers (every 5 frames): `[cam_id] processed N frames`
-   Added periodic logging in aggregator (every 5 batches): `[aggregator] processed N embedding batches`
-   Added exception logging in aggregator so errors are visible
-   Improved server startup log to indicate `parallel=true` when enabled

**Test:**

```bash
PARALLEL=true python scripts/run_multi_camera.py
# You should see logs like:
# [run_loop] model warmup complete (parallel=true)
# [cam1] processed 5 frames
# [cam2] processed 5 frames
# [aggregator] processed 5 embedding batches
```

Or with embedded runner:

```bash
START_RUNNER=true PARALLEL=true uvicorn api.server:app
# Server logs will show:
# Starting embedded multi-camera runner (background thread, parallel=true)
```

---

### Issue 4: "YOLO should be part of the pipeline, is it not?"

**Clarification:**
YOLO **is definitely part of the pipeline**. It's the first stage (object detection).

**What the pipeline does:**

1. **YOLO** detects objects and produces bounding boxes
2. **SAM2** creates masks for those boxes
3. **OSNet** extracts feature vectors from the detected regions
4. **BoT-SORT** tracks objects frame-to-frame
5. **FAISS** links tracks across cameras

**Added to README:**

-   New "Pipeline Overview" section explicitly lists all stages
-   Clarifies that YOLO (detection) is the entry point
-   Explains flow through segmentation → ReID → tracking → output
-   Documents how each component works together

**Verification:**
Check `scripts/run_multi_camera.py` line ~50:

```python
detector = YOLODetector(model_name="yolo12n.pt", device="cpu")
```

The detector runs on every frame and its output (boxes) drives the rest of the pipeline.

---

## Files Modified

1. **api/routes/perf.py** — Fixed metrics visibility using `getattr()`
2. **scripts/run_multi_camera.py** — Added logging in parallel mode workers and aggregator
3. **api/server.py** — Improved startup logging with parallel flag indication
4. **scripts/kill_port.sh** — New helper script for port cleanup
5. **README.md** — Added "Pipeline Overview" section, port cleanup instructions, clarified YOLO role

## Testing Recommendations

```bash
# Test 1: Metrics endpoint
START_RUNNER=true uvicorn api.server:app &
sleep 3
curl http://localhost:8000/api/status/perf  # Should show timing data
kill %1

# Test 2: Parallel mode logging
PARALLEL=true python scripts/run_multi_camera.py  # Should show periodic progress logs

# Test 3: Port cleanup
bash scripts/kill_port.sh 8000
# Then restart server

# Test 4: Verify YOLO is running
python -c "from services.detector import YOLODetector; d = YOLODetector(); print('YOLO loaded')"
```
