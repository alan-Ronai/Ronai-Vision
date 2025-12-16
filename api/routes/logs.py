"""
API routes for operational event logging and querying.

Provides endpoints for:
- Querying operational events with filtering
- Real-time event streaming (WebSocket)
- Event statistics and analytics
- Log file management
"""

import logging
import json
import time
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from pathlib import Path
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from services.logging.operational_logger import get_operational_logger

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/logs", tags=["logs"])


# WebSocket manager for real-time event streaming
class EventStreamManager:
    """Manages WebSocket connections for real-time event streaming."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, event: Dict):
        """Broadcast event to all connected clients."""
        for connection in self.active_connections:
            try:
                await connection.send_json(event)
            except Exception as e:
                logger.error(f"Failed to broadcast event: {e}")


event_stream_manager = EventStreamManager()


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================


class LogEventResponse(BaseModel):
    """Log event response."""

    timestamp: float
    event_type: str
    severity: str
    camera_id: Optional[str]
    track_id: Optional[int]
    message: str
    details: Optional[Dict] = None


class LogQueryRequest(BaseModel):
    """Request to query logs."""

    start_time: Optional[float] = None
    end_time: Optional[float] = None
    event_type: Optional[str] = None
    severity: Optional[str] = None
    camera_id: Optional[str] = None
    track_id: Optional[int] = None
    limit: int = 100


# ============================================================================
# LOG QUERY LOGIC
# ============================================================================


def _read_operational_logs() -> List[Dict]:
    """Read operational logs from JSONL file.

    Returns:
        List of log events
    """
    op_logger = get_operational_logger()
    log_file = Path(op_logger.log_file)

    events = []
    if log_file.exists():
        try:
            with open(log_file, "r") as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())
                        events.append(event)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"Failed to read operational logs: {e}")

    return events


def _filter_events(
    events: List[Dict],
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    event_type: Optional[str] = None,
    severity: Optional[str] = None,
    camera_id: Optional[str] = None,
    track_id: Optional[int] = None,
) -> List[Dict]:
    """Filter events by various criteria.

    Args:
        events: List of events
        start_time: Minimum timestamp (Unix time)
        end_time: Maximum timestamp (Unix time)
        event_type: Filter by event type
        severity: Filter by severity
        camera_id: Filter by camera ID
        track_id: Filter by track ID

    Returns:
        Filtered events
    """
    filtered = []

    for event in events:
        # Time range filter
        if start_time and event.get("timestamp", 0) < start_time:
            continue
        if end_time and event.get("timestamp", 0) > end_time:
            continue

        # Event type filter
        if event_type and event.get("event_type") != event_type:
            continue

        # Severity filter
        if severity and event.get("severity") != severity:
            continue

        # Camera ID filter
        if camera_id and event.get("camera_id") != camera_id:
            continue

        # Track ID filter
        if track_id and event.get("track_id") != track_id:
            continue

        filtered.append(event)

    return filtered


def _cleanup_old_logs(retention_days: int = 2):
    """Remove log entries older than retention period.

    Args:
        retention_days: Number of days to retain (default 2 days)
    """
    op_logger = get_operational_logger()
    log_file = Path(op_logger.log_file)

    if not log_file.exists():
        return

    try:
        cutoff_time = time.time() - (retention_days * 86400)
        events = _read_operational_logs()

        # Keep only recent events
        recent_events = [e for e in events if e.get("timestamp", 0) > cutoff_time]

        # Rewrite log file
        with open(log_file, "w") as f:
            for event in recent_events:
                f.write(json.dumps(event) + "\n")

        deleted_count = len(events) - len(recent_events)
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old log entries")

    except Exception as e:
        logger.error(f"Failed to cleanup old logs: {e}")


# ============================================================================
# API ENDPOINTS
# ============================================================================


@router.get("/events")
async def get_events(
    start_time: Optional[float] = Query(
        None, description="Start time (Unix timestamp)"
    ),
    end_time: Optional[float] = Query(None, description="End time (Unix timestamp)"),
    event_type: Optional[str] = Query(None, description="Event type to filter"),
    severity: Optional[str] = Query(
        None, description="Severity level (info/warning/critical)"
    ),
    camera_id: Optional[str] = Query(None, description="Camera ID to filter"),
    track_id: Optional[int] = Query(None, description="Track ID to filter"),
    limit: int = Query(100, description="Maximum results"),
) -> Dict:
    """Query operational events with filtering.

    Example queries:
    - `/api/logs/events?severity=critical` - Get all critical events
    - `/api/logs/events?camera_id=cam1&event_type=person_detected` - Get person detections
    - `/api/logs/events?start_time=1702080000&end_time=1702166400` - Get events in time range
    - `/api/logs/events?track_id=42` - Get all events for track 42

    Args:
        start_time: Minimum timestamp
        end_time: Maximum timestamp
        event_type: Filter by event type
        severity: Filter by severity
        camera_id: Filter by camera
        track_id: Filter by track
        limit: Max results

    Returns:
        List of matching events
    """
    # Default time range: last 2 days
    if end_time is None:
        end_time = time.time()
    if start_time is None:
        start_time = end_time - (2 * 86400)  # 2 days

    # Read and filter events
    all_events = _read_operational_logs()
    filtered = _filter_events(
        all_events,
        start_time=start_time,
        end_time=end_time,
        event_type=event_type,
        severity=severity,
        camera_id=camera_id,
        track_id=track_id,
    )

    # Sort by timestamp (newest first)
    filtered.sort(key=lambda x: x.get("timestamp", 0), reverse=True)

    # Apply limit
    filtered = filtered[:limit]

    return {
        "total": len(all_events),
        "matched": len(filtered),
        "limit": limit,
        "filters": {
            "start_time": start_time,
            "end_time": end_time,
            "event_type": event_type,
            "severity": severity,
            "camera_id": camera_id,
            "track_id": track_id,
        },
        "events": filtered,
    }


@router.get("/events/stats")
async def get_event_stats(
    start_time: Optional[float] = Query(
        None, description="Start time (Unix timestamp)"
    ),
    end_time: Optional[float] = Query(None, description="End time (Unix timestamp)"),
) -> Dict:
    """Get statistics on events.

    Args:
        start_time: Start time for stats
        end_time: End time for stats

    Returns:
        Statistics about events
    """
    if end_time is None:
        end_time = time.time()
    if start_time is None:
        start_time = end_time - (2 * 86400)

    all_events = _read_operational_logs()
    events = _filter_events(all_events, start_time=start_time, end_time=end_time)

    # Aggregate stats
    stats = {
        "total_events": len(events),
        "by_event_type": {},
        "by_severity": {},
        "by_camera": {},
        "critical_events": 0,
    }

    for event in events:
        event_type = event.get("event_type", "unknown")
        severity = event.get("severity", "info")
        camera_id = event.get("camera_id", "unknown")

        stats["by_event_type"][event_type] = (
            stats["by_event_type"].get(event_type, 0) + 1
        )
        stats["by_severity"][severity] = stats["by_severity"].get(severity, 0) + 1
        stats["by_camera"][camera_id] = stats["by_camera"].get(camera_id, 0) + 1

        if severity == "critical":
            stats["critical_events"] += 1

    return stats


@router.get("/events/critical")
async def get_critical_events(
    limit: int = Query(50, description="Max results"),
) -> Dict:
    """Get recent critical events (alerts).

    Args:
        limit: Maximum results

    Returns:
        List of critical events
    """
    events = _read_operational_logs()
    critical = [e for e in events if e.get("severity") == "critical"]
    critical.sort(key=lambda x: x.get("timestamp", 0), reverse=True)

    return {
        "total_critical": len(critical),
        "recent": critical[:limit],
    }


@router.delete("/events/cleanup")
async def cleanup_logs(
    retention_days: int = Query(2, description="Retention period in days"),
):
    """Clean up old log entries.

    Removes events older than retention period.

    Args:
        retention_days: Number of days to retain

    Returns:
        Cleanup status
    """
    try:
        _cleanup_old_logs(retention_days)
        return {
            "status": "success",
            "message": f"Logs cleaned up (retention: {retention_days} days)",
        }
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return {
            "status": "error",
            "message": str(e),
        }


# ============================================================================
# WEBSOCKET FOR REAL-TIME EVENT STREAMING
# ============================================================================


@router.websocket("/ws/events")
async def websocket_events(websocket: WebSocket):
    """WebSocket endpoint for real-time event streaming.

    Clients connect and receive events as they're logged.

    Example (JavaScript):
    ```
    const ws = new WebSocket("ws://localhost:8000/api/logs/ws/events");
    ws.onmessage = (event) => {
        const log_event = JSON.parse(event.data);
        console.log("New event:", log_event.event_type, log_event.severity);
    };
    ```
    """
    await event_stream_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            # Ignore client messages for now
    except WebSocketDisconnect:
        event_stream_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        event_stream_manager.disconnect(websocket)


def broadcast_event(event: Dict):
    """Broadcast an event to all connected WebSocket clients.

    Called by OperationalLogger when events are logged.

    Args:
        event: Event dictionary
    """
    import asyncio

    # Run async broadcast in a way that works from sync context
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Already in async context, schedule as task
            asyncio.create_task(event_stream_manager.broadcast(event))
        else:
            # In sync context, run until complete
            loop.run_until_complete(event_stream_manager.broadcast(event))
    except Exception as e:
        logger.debug(f"Could not broadcast event: {e}")
