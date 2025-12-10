"""Multi-camera runner: orchestrates multiple camera simulators,
runs detector -> segmenter -> reid -> per-camera tracker -> global reid store,
and writes overlays + JSON metadata.

This runner uses available local models and runs on CPU for development.
"""

import os
import time
import json
import threading
from typing import Optional

import cv2
import numpy as np
import queue

from services.camera.manager import CameraManager
from services.detector import YOLODetector
from services.segmenter.sam2_segmenter import SAM2Segmenter
from services.reid import get_reid
from services.tracker.bot_sort import BoTSortTracker
from services.reid.reid_store import ReIDStore
from services.output.renderer import FrameRenderer
from services.output.broadcaster import broadcaster
import requests

# Optional: if STREAM_SERVER_URL is set, POST frames to the server publish endpoint
STREAM_SERVER_URL = os.getenv("STREAM_SERVER_URL")
STREAM_PUBLISH_TOKEN = os.getenv("STREAM_PUBLISH_TOKEN")
# Optionally save frames+metadata locally when enabled (set SAVE_FRAMES=true in env)
SAVE_FRAMES = os.getenv("SAVE_FRAMES", "false").lower() in ("1", "true", "yes")

# ============================================================================
# DETECTION & SEGMENTATION CONFIGURATION
# ============================================================================
# Configure which object classes to detect and segment
# Default: all classes (None). Override via env: ALLOWED_CLASSES=person,car,dog
# Leave empty/unset to process all 80 COCO classes
ALLOWED_CLASSES_STR = os.getenv("ALLOWED_CLASSES", "")
ALLOWED_CLASSES = ALLOWED_CLASSES_STR.split(",") if ALLOWED_CLASSES_STR else None

# YOLO confidence threshold (0.0-1.0, lower = more sensitive)
# Override via env: YOLO_CONFIDENCE=0.5
YOLO_CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", "0.25"))

# Public metrics structure populated at runtime by `run_loop` for diagnostics.
RUN_METRICS = {}


def run_loop(
    stop_event: Optional[threading.Event] = None, max_frames: Optional[int] = 30
) -> None:
    """Run the multi-camera processing loop.

    stop_event: optional threading.Event instance that can be set to request a clean shutdown.
    max_frames: how many iterations to run (default 30 to preserve current behaviour). Use None for unlimited.
    """

    out_dir = "output/multi"
    os.makedirs(out_dir, exist_ok=True)

    # Load cameras from config file
    camera_config_path = os.getenv("CAMERA_CONFIG", "config/camera_settings.json")
    cm = CameraManager()

    if os.path.exists(camera_config_path):
        with open(camera_config_path, "r") as f:
            camera_config = json.load(f)
        cm.add_from_config(camera_config)
        print(f"[INFO] Loaded cameras from {camera_config_path}")
    else:
        print(f"[WARNING] Camera config not found at {camera_config_path}, no cameras loaded")

    cm.start_all()

    # Get device from environment (cpu or cuda)
    device = os.getenv("DEVICE", "cpu")
    detector = YOLODetector(model_name="yolo12n.pt", device=device)
    segmenter = SAM2Segmenter(model_type="small", device=device)
    reid = get_reid()

    # Initialize trackers for each camera from config
    trackers = {cam_id: BoTSortTracker() for cam_id in cm.workers.keys()}
    store = ReIDStore()
    renderer = FrameRenderer()

    # Warm-up models to reduce first-inference latency
    try:
        dummy = np.zeros((360, 640, 3), dtype=np.uint8)
        _ = detector.predict(dummy, confidence=0.25)
        # fake box covering center for reid warmup
        try:
            _ = reid.extract_features(dummy, np.array([[100, 50, 200, 300]]))
        except Exception:
            # some reid backends may require different shapes — ignore warmup failures
            pass
        print("model warmup complete")
    except Exception:
        print("model warmup failed (continuing)")

    frame_idx = 0
    # Parallel mode toggle (opt-in): set PARALLEL=true to enable per-camera workers
    PARALLEL = os.getenv("PARALLEL", "false").lower() in ("1", "true", "yes")

    # Structures for parallel processing
    agg_queue: "queue.Queue" = queue.Queue()
    global_id_map = {}
    global_id_lock = threading.Lock()

    def aggregator_worker():
        """Consume embedding batches from camera workers and update ReIDStore."""
        agg_count = 0
        while True:
            try:
                item = agg_queue.get(timeout=0.5)
            except Exception:
                # check stop_event
                if stop_event is not None and stop_event.is_set():
                    break
                continue

            if item is None:
                break

            try:
                cam_id = item.get("cam_id")
                emb_arr = item.get("emb")
                meta = item.get("meta")
                local_ids = item.get("local_ids")
                if emb_arr is not None and len(emb_arr) > 0:
                    gids = store.upsert(emb_arr.astype(np.float32), meta)
                    # map local track ids to gids (1:1 order assumed)
                    with global_id_lock:
                        for lid, gid in zip(local_ids, gids):
                            global_id_map[(cam_id, lid)] = gid
                    agg_count += 1
                    if agg_count % 5 == 0:
                        print(f"[aggregator] processed {agg_count} embedding batches")
            except Exception as e:
                print(f"[aggregator] exception: {e}")
            finally:
                agg_queue.task_done()

    worker_threads = {}
    try:
        # timing accumulators for perf analysis
        timing = {
            "detect": 0.0,
            "segment": 0.0,
            "reid": 0.0,
            "track": 0.0,
            "render": 0.0,
            "publish": 0.0,
            "count": 0,
        }

        # expose timing dictionary for external inspection
        global RUN_METRICS
        RUN_METRICS = timing

        if PARALLEL:
            # start aggregator
            agg_thread = threading.Thread(target=aggregator_worker, daemon=True)
            agg_thread.start()

            # per-camera worker
            def make_camera_worker(cid: str):
                def _worker():
                    last_ts = 0.0
                    local_frame_idx = 0
                    while max_frames is None or local_frame_idx < int(max_frames):
                        if stop_event is not None and stop_event.is_set():
                            break
                        w = cm.workers.get(cid)
                        if w is None:
                            time.sleep(0.01)
                            continue
                        latest = w.get_latest()
                        if latest is None:
                            time.sleep(0.01)
                            continue
                        frame, ts = latest
                        if ts is None or ts <= last_ts:
                            time.sleep(0.01)
                            continue
                        last_ts = ts

                        # same processing as before, but per-camera
                        if not isinstance(frame, np.ndarray):
                            frame = np.array(frame, dtype=np.uint8)
                        elif frame.dtype != np.uint8:
                            if frame.dtype == np.float32 or frame.dtype == np.float64:
                                frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
                            else:
                                frame = frame.astype(np.uint8)

                        det = detector.predict(frame, confidence=YOLO_CONFIDENCE)
                        boxes = det.boxes
                        scores = det.scores
                        class_ids = det.class_ids

                        # Filter boxes/class_ids/scores to only process allowed classes
                        if ALLOWED_CLASSES is not None:
                            segmented_mask = np.isin(
                                class_ids,
                                [
                                    i
                                    for i, cn in enumerate(det.class_names)
                                    if cn in ALLOWED_CLASSES
                                ],
                            )
                            filtered_boxes = boxes[segmented_mask]
                            filtered_class_ids = class_ids[segmented_mask]
                            filtered_scores = scores[segmented_mask]
                        else:
                            filtered_boxes = boxes
                            filtered_class_ids = class_ids
                            filtered_scores = scores

                        # Segment filtered boxes only
                        masks = np.zeros(
                            (0, frame.shape[0], frame.shape[1]), dtype=np.uint8
                        )
                        if len(filtered_boxes) > 0:
                            seg = segmenter.segment(frame, boxes=filtered_boxes)
                            masks = seg.masks
                            if masks.ndim == 4 and masks.shape[1] == 1:
                                masks = masks[:, 0, :, :]

                        # Extract ReID features from filtered boxes
                        feats = None
                        if len(filtered_boxes) > 0:
                            feats = reid.extract_features(frame, filtered_boxes)

                        # Track only filtered boxes
                        tracks = trackers[cid].update(filtered_boxes, filtered_class_ids, filtered_scores, feats)

                        # prepare embeddings for aggregator
                        meta = []
                        emb = []
                        local_ids = []
                        for tr in tracks:
                            tid = tr.track_id
                            if feats is not None and len(feats) >= 1:
                                matched_idx = None
                                for i, b in enumerate(filtered_boxes):
                                    if np.allclose(b, tr.box, atol=2.0):
                                        matched_idx = i
                                        break
                                if matched_idx is not None:
                                    emb.append(feats[matched_idx])
                                    meta.append(
                                        {"camera_id": cid, "track_id": tid, "ts": ts}
                                    )
                                    local_ids.append(tid)

                        if len(emb) > 0:
                            agg_queue.put(
                                {
                                    "cam_id": cid,
                                    "emb": np.vstack(emb),
                                    "meta": meta,
                                    "local_ids": local_ids,
                                }
                            )

                        # attach any known global ids for rendering
                        with global_id_lock:
                            for t in tracks:
                                gid = global_id_map.get((cid, t.track_id))
                                if gid is not None:
                                    setattr(t, "global_id", gid)

                        out = renderer.render_masks(frame, masks)
                        out = renderer.render_tracks(out, tracks, det.class_names)

                        # publish
                        try:
                            ok, buf = cv2.imencode(
                                ".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                            )
                            if ok:
                                if STREAM_SERVER_URL:
                                    url = f"{STREAM_SERVER_URL.rstrip('/')}/api/stream/publish/{cid}"
                                    headers = {"Content-Type": "image/jpeg"}
                                    if STREAM_PUBLISH_TOKEN:
                                        headers["x-stream-token"] = STREAM_PUBLISH_TOKEN
                                    resp = requests.post(
                                        url,
                                        data=buf.tobytes(),
                                        headers=headers,
                                        timeout=2.0,
                                    )
                                    if resp.status_code != 200:
                                        broadcaster.publish_frame(cid, out)
                                else:
                                    broadcaster.publish_frame(cid, out)
                        except Exception:
                            broadcaster.publish_frame(cid, out)

                        local_frame_idx += 1
                        if local_frame_idx % 5 == 0:
                            print(f"[{cid}] processed {local_frame_idx} frames")
                        time.sleep(0.01)

                return _worker

            # start workers for each camera
            for cid in list(cm.workers.keys()):
                th = threading.Thread(target=make_camera_worker(cid), daemon=True)
                worker_threads[cid] = th
                th.start()

            # wait until stop_event or until workers finish
            while True:
                if stop_event is not None and stop_event.is_set():
                    break
                all_dead = all(not t.is_alive() for t in worker_threads.values())
                if all_dead:
                    break
                time.sleep(0.1)

            # signal aggregator to stop
            agg_queue.put(None)
            agg_thread.join(timeout=2.0)

        else:
            while max_frames is None or frame_idx < int(max_frames):
                # cooperative stop check
                if stop_event is not None and stop_event.is_set():
                    print("run_loop: stop_event set, exiting loop")
                    break

                frames = cm.get_frames()
                timestamp = time.time()
                for cam_id, val in frames.items():
                    # check stop between camera iterations
                    if stop_event is not None and stop_event.is_set():
                        break

                    if val is None:
                        continue
                    frame, ts = val
                    if frame is None:
                        print(
                            f"[{cam_id}] frame_idx={frame_idx}: frame is None, skipping"
                        )
                        continue

                    # Ensure frame is numpy array, uint8, BGR
                    if not isinstance(frame, np.ndarray):
                        print(f"[{cam_id}] frame type is {type(frame)}, converting...")
                        frame = np.array(frame, dtype=np.uint8)
                    elif frame.dtype != np.uint8:
                        if frame.dtype == np.float32 or frame.dtype == np.float64:
                            frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
                        else:
                            frame = frame.astype(np.uint8)

                    print(
                        f"[{cam_id}] frame_idx={frame_idx}: shape={frame.shape}, dtype={frame.dtype}"
                    )

                    # Run detection
                    t0 = time.time()
                    det = detector.predict(frame, confidence=YOLO_CONFIDENCE)
                    timing["detect"] += time.time() - t0
                    boxes = det.boxes
                    scores = det.scores
                    class_ids = det.class_ids

                    # Segment (optional) - with optional class filtering
                    t0 = time.time()
                    masks = np.zeros(
                        (0, frame.shape[0], frame.shape[1]), dtype=np.uint8
                    )
                    if len(boxes) > 0:
                        if ALLOWED_CLASSES is not None:
                            # Use class filtering (only segment specified classes)
                            seg = segmenter.segment_from_detections(
                                frame,
                                boxes=boxes,
                                class_ids=class_ids,
                                class_names=det.class_names,
                                allowed_class_names=ALLOWED_CLASSES,
                            )
                        else:
                            # Segment all detected boxes (no filtering)
                            seg = segmenter.segment(frame, boxes=boxes)
                        masks = seg.masks
                        if masks.ndim == 4 and masks.shape[1] == 1:
                            masks = masks[:, 0, :, :]
                    timing["segment"] += time.time() - t0

                    # Filter boxes/class_ids/scores to only process allowed classes
                    if ALLOWED_CLASSES is not None:
                        # Only process boxes for specified classes (e.g., person)
                        segmented_mask = np.isin(
                            class_ids,
                            [
                                i
                                for i, cn in enumerate(det.class_names)
                                if cn in ALLOWED_CLASSES
                            ],
                        )
                        filtered_boxes = boxes[segmented_mask]
                        filtered_class_ids = class_ids[segmented_mask]
                        filtered_scores = scores[segmented_mask]
                    else:
                        # Use all boxes
                        filtered_boxes = boxes
                        filtered_class_ids = class_ids
                        filtered_scores = scores

                    # ReID embeddings (only on filtered boxes)
                    t0 = time.time()
                    feats = None
                    if len(filtered_boxes) > 0:
                        feats = reid.extract_features(frame, filtered_boxes)
                    timing["reid"] += time.time() - t0

                    # Per-camera tracking (only track filtered boxes)
                    t0 = time.time()
                    tracks = trackers[cam_id].update(
                        filtered_boxes, filtered_class_ids, filtered_scores, feats
                    )
                    timing["track"] += time.time() - t0

                    # Upsert track embeddings to global ReID store
                    # Prepare metadata and embeddings
                    meta = []
                    emb = []
                    for tr in tracks:
                        tid = tr.track_id
                        # if we have an embedding for this track, use it
                        if feats is not None and len(feats) >= 1:
                            # find matching det index by bbox equality (approx)
                            matched_idx = None
                            for i, b in enumerate(filtered_boxes):
                                if np.allclose(b, tr.box, atol=2.0):
                                    matched_idx = i
                                    break
                            if matched_idx is not None:
                                emb.append(feats[matched_idx])
                                meta.append(
                                    {
                                        "camera_id": cam_id,
                                        "track_id": tid,
                                        "ts": timestamp,
                                    }
                                )
                    if len(emb) > 0:
                        emb_arr = np.vstack(emb).astype(np.float32)
                        gids = store.upsert(emb_arr, meta)
                    else:
                        gids = []

                    # Render and save — final stream output should be boxes+IDs only by default
                    t0 = time.time()
                    out = renderer.render_masks(frame, masks)
                    out = renderer.render_tracks(out, tracks, det.class_names)
                    timing["render"] += time.time() - t0

                    # publish to stream server if configured, otherwise publish locally
                    t0 = time.time()
                    if STREAM_SERVER_URL:
                        try:
                            ok, buf = cv2.imencode(
                                ".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                            )
                            if ok:
                                url = f"{STREAM_SERVER_URL.rstrip('/')}/api/stream/publish/{cam_id}"
                                headers = {"Content-Type": "image/jpeg"}
                                if STREAM_PUBLISH_TOKEN:
                                    headers["x-stream-token"] = STREAM_PUBLISH_TOKEN
                                resp = requests.post(
                                    url,
                                    data=buf.tobytes(),
                                    headers=headers,
                                    timeout=2.0,
                                )
                                if resp.status_code != 200:
                                    print(
                                        f"publish failed: {resp.status_code} {resp.text}"
                                    )
                                    # fallback to local publish
                                    broadcaster.publish_frame(cam_id, out)
                            else:
                                broadcaster.publish_frame(cam_id, out)
                        except Exception as e:
                            print(f"publish exception: {e}")
                            broadcaster.publish_frame(cam_id, out)
                    else:
                        # publish to in-memory broadcaster for streaming (do not save each frame)
                        broadcaster.publish_frame(cam_id, out)
                    timing["publish"] += time.time() - t0

                    # Optionally save rendered frame and metadata (disabled by default)
                    if SAVE_FRAMES:
                        fname = os.path.join(
                            out_dir, f"{cam_id}_frame_{frame_idx:04d}.png"
                        )
                        jsonf = os.path.join(
                            out_dir, f"{cam_id}_frame_{frame_idx:04d}.json"
                        )
                        try:
                            cv2.imwrite(fname, out)
                        except Exception:
                            pass

                        meta_out = {
                            "camera_id": cam_id,
                            "frame_idx": frame_idx,
                            "timestamp": timestamp,
                            "detections": [
                                {
                                    "box": b.tolist(),
                                    "score": float(s),
                                    "class_id": int(c),
                                }
                                for b, s, c in zip(boxes, scores, class_ids)
                            ],
                            "tracks": [
                                {
                                    "track_id": t.track_id,
                                    "box": t.box.tolist(),
                                    "class_id": t.class_id,
                                    "confidence": t.confidence,
                                }
                                for t in tracks
                            ],
                            "global_ids": gids,
                        }
                        try:
                            with open(jsonf, "w") as f:
                                json.dump(meta_out, f)
                        except Exception:
                            pass

                frame_idx += 1
                timing["count"] += 1

                # log timing every 10 frames
                if timing["count"] % 10 == 0:
                    avg_detect = timing["detect"] / timing["count"]
                    avg_segment = timing["segment"] / timing["count"]
                    avg_reid = timing["reid"] / timing["count"]
                    avg_track = timing["track"] / timing["count"]
                    avg_render = timing["render"] / timing["count"]
                    avg_publish = timing["publish"] / timing["count"]
                    print(
                        f"perf avg (s): detect={avg_detect:.3f}, segment={avg_segment:.3f}, reid={avg_reid:.3f}, track={avg_track:.3f}, render={avg_render:.3f}, publish={avg_publish:.3f}"
                    )

                time.sleep(0.05)
    finally:
        cm.stop_all()


def main():
    run_loop()


if __name__ == "__main__":
    main()
