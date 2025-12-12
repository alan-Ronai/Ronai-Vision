from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from services.profiler import profiler
import threading

router = APIRouter()


def _check_runner_active():
    """Check if the embedded runner thread is active."""
    for thread in threading.enumerate():
        if thread.name == "embedded-runner":
            return {"active": True, "alive": thread.is_alive()}
    return {"active": False, "alive": False}


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


@router.get("/perf/detailed", response_class=PlainTextResponse)
def detailed_perf_status():
    """Return detailed profiler statistics as formatted text report."""
    try:
        # Try to load from file (cross-process)
        from services.profiler_export import load_stats

        stats_data = load_stats()
        stats = stats_data.get("global", {})

        # Fallback to in-memory profiler if file is empty
        if not stats:
            stats = profiler.get_summary()

        if not stats:
            cameras_available = (
                stats_data.get("cameras", {}).keys() if stats_data else []
            )
            return (
                "No profiling data available yet.\n\n"
                f"Profiler stats count: {len(profiler.stats)}\n"
                f"Cameras with stats: {list(cameras_available)}\n"
                f"Runner active: {_check_runner_active()}\n"
            )

        # Build formatted report similar to profiler.print_report()
        output = []
        output.append("=" * 100)
        output.append("PERFORMANCE PROFILER REPORT")
        output.append("=" * 100)

        # Sort by total time (biggest impact)
        sorted_ops = sorted(
            stats.items(), key=lambda x: x[1]["total_time_s"], reverse=True
        )

        # Header
        output.append("")
        output.append(
            f"{'Operation':<30} {'Count':>8} {'Avg(ms)':>10} {'Med(ms)':>10} "
            f"{'P95(ms)':>10} {'FPS':>8} {'Total(s)':>10}"
        )
        output.append("-" * 100)

        # All operations
        total_time = sum(op["total_time_s"] for _, op in sorted_ops)
        for name, op in sorted_ops:
            pct = (op["total_time_s"] / total_time * 100) if total_time > 0 else 0
            output.append(
                f"{name:<30} {op['count']:>8} "
                f"{op['avg_ms']:>10.2f} {op['median_ms']:>10.2f} "
                f"{op['p95_ms']:>10.2f} {op['fps']:>8.1f} "
                f"{op['total_time_s']:>10.2f} ({pct:>5.1f}%)"
            )

        output.append("-" * 100)
        total_count = sum(op["count"] for _, op in sorted_ops)
        output.append(
            f"{'TOTAL':<30} {total_count:>8} "
            f"{'':>10} {'':>10} {'':>10} {'':>8} {total_time:>10.2f}"
        )

        # Bottlenecks
        top_n = min(5, len(sorted_ops))
        output.append("")
        output.append(f"🔴 TOP {top_n} BOTTLENECKS (by total time):")
        for i, (name, op) in enumerate(sorted_ops[:top_n], 1):
            pct = (op["total_time_s"] / total_time * 100) if total_time > 0 else 0
            output.append(
                f"  {i}. {name}: {op['total_time_s']:.2f}s ({pct:.1f}%) - "
                f"avg {op['avg_ms']:.2f}ms, {op['count']} calls"
            )

        output.append("")
        output.append("=" * 100)

        return "\n".join(output)
    except Exception as e:
        return f"Error generating profiler report: {str(e)}"


@router.get("/perf/detailed/json")
def detailed_perf_status_json(camera_id: str = None):
    """Return detailed profiler statistics with percentiles and bottleneck analysis (JSON format).

    Query parameters:
        camera_id: Optional camera ID to get stats for a specific camera
    """
    try:
        # Try to load from file (cross-process)
        from services.profiler_export import load_stats

        stats_data = load_stats()

        if camera_id:
            # Get specific camera stats
            stats = stats_data.get("cameras", {}).get(camera_id, {})
        else:
            # Get global stats
            stats = stats_data.get("global", {})

        # Fallback to in-memory profiler if file is empty
        if not stats:
            stats = profiler.get_summary(camera_id)

        if not stats:
            # Return debug info about why no data
            cameras = (
                list(stats_data.get("cameras", {}).keys())
                if stats_data
                else profiler.get_all_cameras()
            )
            return {
                "ok": True,
                "camera_id": camera_id,
                "operations": {},
                "note": "no profiling data available yet",
                "debug": {
                    "profiler_stats_count": len(profiler.stats),
                    "profiler_camera_count": len(profiler.camera_stats),
                    "cameras_with_stats": cameras,
                    "runner_active": _check_runner_active(),
                },
            }

        # Calculate total time and identify bottlenecks
        total_time = sum(op["total_time_s"] for op in stats.values())
        bottlenecks = sorted(
            [
                {
                    "operation": name,
                    "total_time_s": op["total_time_s"],
                    "percentage": (op["total_time_s"] / total_time * 100)
                    if total_time > 0
                    else 0,
                    "avg_ms": op["avg_ms"],
                    "p95_ms": op["p95_ms"],
                    "fps": op["fps"],
                }
                for name, op in stats.items()
            ],
            key=lambda x: x["total_time_s"],
            reverse=True,
        )

        return {
            "ok": True,
            "camera_id": camera_id,
            "operations": stats,
            "bottlenecks": bottlenecks[:5],
            "total_time_s": total_time,
            "total_samples": sum(op["count"] for op in stats.values()),
            "available_cameras": profiler.get_all_cameras(),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.post("/perf/reset")
def reset_profiler():
    """Reset all profiler statistics."""
    try:
        profiler.reset()
        return {"ok": True, "message": "Profiler statistics reset"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.get("/perf/debug")
def debug_profiler():
    """Debug endpoint to check profiler and runner status."""
    import os

    return {
        "ok": True,
        "environment": {
            "START_RUNNER": os.getenv("START_RUNNER", "not set"),
            "PARALLEL": os.getenv("PARALLEL", "not set"),
            "DEVICE": os.getenv("DEVICE", "not set"),
        },
        "profiler": {
            "stats_count": len(profiler.stats),
            "operations": list(profiler.stats.keys()),
            "total_samples": sum(s.count for s in profiler.stats.values()),
        },
        "runner": _check_runner_active(),
        "threads": [
            {"name": t.name, "alive": t.is_alive(), "daemon": t.daemon}
            for t in threading.enumerate()
        ],
    }
