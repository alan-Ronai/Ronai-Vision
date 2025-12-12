"""Performance profiler for pipeline stages.

Tracks timing statistics for each stage of the detection pipeline
and provides detailed performance reports.
"""

import time
from typing import Dict, List, Optional
from contextlib import contextmanager
from dataclasses import dataclass, field
import statistics
from collections import defaultdict


@dataclass
class TimingStats:
    """Statistics for a single profiled operation."""

    name: str
    count: int = 0
    total_time: float = 0.0
    min_time: float = float("inf")
    max_time: float = 0.0
    samples: List[float] = field(default_factory=list)

    def add_sample(self, duration: float):
        """Add a timing sample."""
        self.count += 1
        self.total_time += duration
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)

        # Keep last 100 samples for statistics
        self.samples.append(duration)
        if len(self.samples) > 100:
            self.samples.pop(0)

    @property
    def avg_time(self) -> float:
        """Average time per operation."""
        return self.total_time / self.count if self.count > 0 else 0.0

    @property
    def median_time(self) -> float:
        """Median time per operation."""
        return statistics.median(self.samples) if self.samples else 0.0

    @property
    def stddev_time(self) -> float:
        """Standard deviation of operation time."""
        if len(self.samples) < 2:
            return 0.0
        return statistics.stdev(self.samples)

    @property
    def p95_time(self) -> float:
        """95th percentile time."""
        if not self.samples:
            return 0.0
        sorted_samples = sorted(self.samples)
        idx = int(len(sorted_samples) * 0.95)
        return sorted_samples[min(idx, len(sorted_samples) - 1)]

    @property
    def fps(self) -> float:
        """Frames per second (inverse of average time)."""
        return 1.0 / self.avg_time if self.avg_time > 0 else 0.0


class PerformanceProfiler:
    """Performance profiler for pipeline stages."""

    def __init__(self):
        self.stats: Dict[str, TimingStats] = {}  # Global stats
        self.camera_stats: Dict[str, Dict[str, TimingStats]] = defaultdict(
            dict
        )  # Per-camera stats
        self._start_times: Dict[str, float] = {}
        self._current_camera: Optional[str] = None  # Track current camera context

    @contextmanager
    def profile(self, operation_name: str):
        """Context manager for profiling an operation.

        Usage:
            with profiler.profile("yolo_detection"):
                results = detector.predict(frame)
        """
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            self.record(operation_name, duration)

    def start(self, operation_name: str):
        """Start timing an operation manually."""
        self._start_times[operation_name] = time.time()

    def stop(self, operation_name: str):
        """Stop timing an operation manually."""
        if operation_name not in self._start_times:
            return

        duration = time.time() - self._start_times[operation_name]
        self.record(operation_name, duration)
        del self._start_times[operation_name]

    def set_camera_context(self, camera_id: Optional[str]):
        """Set the current camera context for per-camera stats."""
        self._current_camera = camera_id

    def record(
        self, operation_name: str, duration: float, camera_id: Optional[str] = None
    ):
        """Record a timing sample.

        Args:
            operation_name: Name of the operation
            duration: Duration in seconds
            camera_id: Optional camera ID for per-camera stats
        """
        # Global stats
        if operation_name not in self.stats:
            self.stats[operation_name] = TimingStats(name=operation_name)
        self.stats[operation_name].add_sample(duration)

        # Per-camera stats
        cam_id = camera_id or self._current_camera
        if cam_id:
            if operation_name not in self.camera_stats[cam_id]:
                self.camera_stats[cam_id][operation_name] = TimingStats(
                    name=operation_name
                )
            self.camera_stats[cam_id][operation_name].add_sample(duration)

    def get_stats(self, operation_name: str) -> Optional[TimingStats]:
        """Get statistics for a specific operation."""
        return self.stats.get(operation_name)

    def get_summary(self, camera_id: Optional[str] = None) -> Dict[str, Dict]:
        """Get summary of statistics.

        Args:
            camera_id: If provided, return only stats for that camera. Otherwise return global stats.

        Returns:
            Dictionary of operation stats
        """
        if camera_id:
            stats_dict = self.camera_stats.get(camera_id, {})
        else:
            stats_dict = self.stats

        summary = {}
        for name, stat in stats_dict.items():
            summary[name] = {
                "count": stat.count,
                "avg_ms": stat.avg_time * 1000,
                "median_ms": stat.median_time * 1000,
                "min_ms": stat.min_time * 1000,
                "max_ms": stat.max_time * 1000,
                "p95_ms": stat.p95_time * 1000,
                "stddev_ms": stat.stddev_time * 1000,
                "fps": stat.fps,
                "total_time_s": stat.total_time,
            }
        return summary

    def export_to_file(self):
        """Export stats to file for cross-process access."""
        try:
            from services.profiler_export import export_stats

            # Export both global and per-camera stats
            global_summary = self.get_summary()
            camera_summaries = {}
            for cam_id in self.camera_stats.keys():
                camera_summaries[cam_id] = self.get_summary(cam_id)

            export_stats(global_summary, camera_summaries)
        except Exception:
            # Don't fail if export fails
            pass

    def get_all_cameras(self) -> List[str]:
        """Get list of all cameras with stats."""
        return list(self.camera_stats.keys())

    def print_report(self, top_n: int = 10):
        """Print a formatted performance report.

        Args:
            top_n: Number of slowest operations to highlight
        """
        if not self.stats:
            print("No profiling data available")
            return

        print("\n" + "=" * 100)
        print("PERFORMANCE PROFILER REPORT")
        print("=" * 100)

        # Sort by total time (biggest impact)
        sorted_stats = sorted(
            self.stats.values(), key=lambda s: s.total_time, reverse=True
        )

        # Print header
        print(
            f"\n{'Operation':<30} {'Count':>8} {'Avg(ms)':>10} {'Med(ms)':>10} {'P95(ms)':>10} {'FPS':>8} {'Total(s)':>10}"
        )
        print("-" * 100)

        # Print all operations
        total_time = sum(s.total_time for s in sorted_stats)
        for stat in sorted_stats:
            pct = (stat.total_time / total_time * 100) if total_time > 0 else 0
            print(
                f"{stat.name:<30} {stat.count:>8} "
                f"{stat.avg_time * 1000:>10.2f} {stat.median_time * 1000:>10.2f} "
                f"{stat.p95_time * 1000:>10.2f} {stat.fps:>8.1f} "
                f"{stat.total_time:>10.2f} ({pct:>5.1f}%)"
            )

        print("-" * 100)
        print(
            f"{'TOTAL':<30} {sum(s.count for s in sorted_stats):>8} "
            f"{'':>10} {'':>10} {'':>10} {'':>8} {total_time:>10.2f}"
        )

        # Highlight bottlenecks
        print(f"\n🔴 TOP {min(top_n, len(sorted_stats))} BOTTLENECKS (by total time):")
        for i, stat in enumerate(sorted_stats[:top_n], 1):
            pct = (stat.total_time / total_time * 100) if total_time > 0 else 0
            print(
                f"  {i}. {stat.name}: {stat.total_time:.2f}s ({pct:.1f}%) - "
                f"avg {stat.avg_time * 1000:.2f}ms, {stat.count} calls"
            )

        print("\n" + "=" * 100 + "\n")

    def reset(self, camera_id: Optional[str] = None):
        """Reset statistics.

        Args:
            camera_id: If provided, reset only that camera's stats. Otherwise reset everything.
        """
        if camera_id:
            self.camera_stats.pop(camera_id, None)
        else:
            self.stats.clear()
            self.camera_stats.clear()
        self._start_times.clear()

    def export_json(self) -> dict:
        """Export statistics as JSON-serializable dict."""
        return {
            "operations": self.get_summary(),
            "total_operations": len(self.stats),
            "total_samples": sum(s.count for s in self.stats.values()),
        }


# Global profiler instance
profiler = PerformanceProfiler()
