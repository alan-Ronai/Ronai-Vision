from fastapi import APIRouter

router = APIRouter()


@router.get("/perf")
def perf_status():
    """Return runtime perf metrics from the embedded runner if available."""
    try:
        import scripts.run_multi_camera as runner_module

        metrics = getattr(runner_module, "RUN_METRICS", {})
        if not metrics or metrics.get("count", 0) == 0:
            return {
                "ok": True,
                "metrics": {},
                "note": "no runner metrics available (runner may not be active yet)",
            }

        # provide averages based on count
        count = metrics.get("count", 0) or 1
        avg = {
            "detect_s": metrics.get("detect", 0.0) / count,
            "segment_s": metrics.get("segment", 0.0) / count,
            "reid_s": metrics.get("reid", 0.0) / count,
            "track_s": metrics.get("track", 0.0) / count,
            "render_s": metrics.get("render", 0.0) / count,
            "publish_s": metrics.get("publish", 0.0) / count,
            "frames": count,
        }
        return {"ok": True, "metrics": avg}
    except Exception as e:
        return {"ok": False, "error": str(e)}
