"""
API routes for track metadata management.

Provides endpoints for:
- Querying metadata by track ID
- Querying metadata by class
- Searching metadata by various criteria
- Adding/updating metadata
- Managing notes, tags, alerts
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Any
from pydantic import BaseModel
import time

from services.tracker.metadata_manager import get_metadata_manager

router = APIRouter(prefix="/api/metadata", tags=["metadata"])


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================


class NoteRequest(BaseModel):
    """Request model for adding a note."""

    text: str
    author: str = "user"


class TagRequest(BaseModel):
    """Request model for managing tags."""

    tag: str


class AlertRequest(BaseModel):
    """Request model for adding an alert."""

    alert_type: str
    message: str
    severity: str = "info"


class AttributeRequest(BaseModel):
    """Request model for setting attributes."""

    key: str
    value: Any


class BehaviorRequest(BaseModel):
    """Request model for setting behavior."""

    behavior: str
    confidence: float = 1.0


class ZoneRequest(BaseModel):
    """Request model for adding a zone."""

    zone_name: str


class MetadataResponse(BaseModel):
    """Response model for metadata."""

    track_id: int
    metadata: dict
    formatted: Optional[str] = None


# ============================================================================
# QUERY ENDPOINTS
# ============================================================================


@router.get("/track/{track_id}", response_model=MetadataResponse)
async def get_track_metadata(track_id: int):
    """Get metadata for a specific track.

    Args:
        track_id: Track ID to query

    Returns:
        Track metadata with formatted summary
    """
    manager = get_metadata_manager()
    metadata = manager.get_track_metadata(track_id)

    if metadata is None:
        raise HTTPException(status_code=404, detail=f"Track {track_id} not found")

    # Format metadata for display
    formatted = _format_metadata(track_id, metadata)

    return {"track_id": track_id, "metadata": metadata, "formatted": formatted}


@router.get("/class/{class_id}")
async def get_class_metadata(class_id: int):
    """Get metadata for all tracks of a specific class.

    Args:
        class_id: Class ID to query

    Returns:
        List of tracks with metadata
    """
    manager = get_metadata_manager()
    tracks = manager.get_tracks_by_class(class_id)

    # Format each track
    formatted_tracks = []
    for track_data in tracks:
        track_id = track_data["track_id"]
        metadata = track_data["metadata"]

        formatted_tracks.append(
            {
                "track_id": track_id,
                "metadata": metadata,
                "formatted": _format_metadata(track_id, metadata),
            }
        )

    return {
        "class_id": class_id,
        "total_tracks": len(formatted_tracks),
        "tracks": formatted_tracks,
    }


@router.get("/search")
async def search_metadata(
    tag: Optional[str] = Query(None, description="Filter by tag"),
    alert_type: Optional[str] = Query(None, description="Filter by alert type"),
    zone: Optional[str] = Query(None, description="Filter by visited zone"),
    behavior: Optional[str] = Query(None, description="Filter by behavior type"),
):
    """Search metadata by various criteria.

    Args:
        tag: Filter by tag
        alert_type: Filter by alert type
        zone: Filter by visited zone
        behavior: Filter by behavior type

    Returns:
        List of matching tracks
    """
    manager = get_metadata_manager()
    results = manager.search_metadata(
        tag=tag, alert_type=alert_type, zone=zone, behavior=behavior
    )

    # Format results
    formatted_results = []
    for track_data in results:
        track_id = track_data["track_id"]
        metadata = track_data["metadata"]

        formatted_results.append(
            {
                "track_id": track_id,
                "metadata": metadata,
                "formatted": _format_metadata(track_id, metadata),
            }
        )

    return {
        "total_matches": len(formatted_results),
        "filters": {
            "tag": tag,
            "alert_type": alert_type,
            "zone": zone,
            "behavior": behavior,
        },
        "results": formatted_results,
    }


@router.get("/all")
async def get_all_metadata():
    """Get all stored metadata.

    Returns:
        All tracks with metadata
    """
    manager = get_metadata_manager()
    all_metadata = manager.get_all_metadata()

    formatted_tracks = []
    for track_id, metadata in all_metadata.items():
        formatted_tracks.append(
            {
                "track_id": track_id,
                "metadata": metadata,
                "formatted": _format_metadata(track_id, metadata),
            }
        )

    return {"total_tracks": len(formatted_tracks), "tracks": formatted_tracks}


@router.get("/stats")
async def get_metadata_stats():
    """Get metadata statistics.

    Returns:
        Statistics about stored metadata
    """
    manager = get_metadata_manager()
    stats = manager.get_stats()
    return stats


# ============================================================================
# UPDATE ENDPOINTS
# ============================================================================


@router.post("/track/{track_id}/note")
async def add_note(track_id: int, request: NoteRequest):
    """Add a note to a track.

    Args:
        track_id: Track ID
        request: Note request

    Returns:
        Success message
    """
    manager = get_metadata_manager()
    metadata = manager.get_track_metadata(track_id)

    if metadata is None:
        raise HTTPException(
            status_code=404,
            detail=f"Track {track_id} not found. Notes can only be added to active or recently seen tracks.",
        )

    # Add note to metadata
    if "notes" not in metadata:
        metadata["notes"] = []

    metadata["notes"].append(
        {"text": request.text, "author": request.author, "timestamp": time.time()}
    )
    metadata["updated_at"] = time.time()

    # Update manager (need to get class_id from somewhere - could store in metadata)
    # For now, just update the stored metadata directly
    manager._metadata[track_id] = metadata
    manager._last_seen[track_id] = time.time()

    return {
        "status": "success",
        "message": f"Note added to track {track_id}",
        "note": request.text,
    }


@router.post("/track/{track_id}/tag")
async def add_tag(track_id: int, request: TagRequest):
    """Add a tag to a track.

    Args:
        track_id: Track ID
        request: Tag request

    Returns:
        Success message
    """
    manager = get_metadata_manager()
    metadata = manager.get_track_metadata(track_id)

    if metadata is None:
        raise HTTPException(status_code=404, detail=f"Track {track_id} not found")

    if "tags" not in metadata:
        metadata["tags"] = []

    if request.tag not in metadata["tags"]:
        metadata["tags"].append(request.tag)
        metadata["updated_at"] = time.time()

        manager._metadata[track_id] = metadata
        manager._last_seen[track_id] = time.time()

    return {
        "status": "success",
        "message": f"Tag '{request.tag}' added to track {track_id}",
        "tags": metadata["tags"],
    }


@router.delete("/track/{track_id}/tag")
async def remove_tag(track_id: int, request: TagRequest):
    """Remove a tag from a track.

    Args:
        track_id: Track ID
        request: Tag request

    Returns:
        Success message
    """
    manager = get_metadata_manager()
    metadata = manager.get_track_metadata(track_id)

    if metadata is None:
        raise HTTPException(status_code=404, detail=f"Track {track_id} not found")

    if "tags" in metadata and request.tag in metadata["tags"]:
        metadata["tags"].remove(request.tag)
        metadata["updated_at"] = time.time()

        manager._metadata[track_id] = metadata
        manager._last_seen[track_id] = time.time()

    return {
        "status": "success",
        "message": f"Tag '{request.tag}' removed from track {track_id}",
        "tags": metadata.get("tags", []),
    }


@router.post("/track/{track_id}/alert")
async def add_alert(track_id: int, request: AlertRequest):
    """Add an alert to a track.

    Args:
        track_id: Track ID
        request: Alert request

    Returns:
        Success message
    """
    manager = get_metadata_manager()
    metadata = manager.get_track_metadata(track_id)

    if metadata is None:
        raise HTTPException(status_code=404, detail=f"Track {track_id} not found")

    if "alerts" not in metadata:
        metadata["alerts"] = []

    metadata["alerts"].append(
        {
            "type": request.alert_type,
            "message": request.message,
            "severity": request.severity,
            "timestamp": time.time(),
        }
    )
    metadata["updated_at"] = time.time()

    manager._metadata[track_id] = metadata
    manager._last_seen[track_id] = time.time()

    return {
        "status": "success",
        "message": f"Alert added to track {track_id}",
        "alert": {
            "type": request.alert_type,
            "message": request.message,
            "severity": request.severity,
        },
    }


@router.post("/track/{track_id}/attribute")
async def set_attribute(track_id: int, request: AttributeRequest):
    """Set a custom attribute on a track.

    Args:
        track_id: Track ID
        request: Attribute request

    Returns:
        Success message
    """
    manager = get_metadata_manager()
    metadata = manager.get_track_metadata(track_id)

    if metadata is None:
        raise HTTPException(status_code=404, detail=f"Track {track_id} not found")

    if "attributes" not in metadata:
        metadata["attributes"] = {}

    metadata["attributes"][request.key] = request.value
    metadata["updated_at"] = time.time()

    manager._metadata[track_id] = metadata
    manager._last_seen[track_id] = time.time()

    return {
        "status": "success",
        "message": f"Attribute '{request.key}' set on track {track_id}",
        "attributes": metadata["attributes"],
    }


@router.post("/track/{track_id}/behavior")
async def set_behavior(track_id: int, request: BehaviorRequest):
    """Set behavior for a track.

    Args:
        track_id: Track ID
        request: Behavior request

    Returns:
        Success message
    """
    manager = get_metadata_manager()
    metadata = manager.get_track_metadata(track_id)

    if metadata is None:
        raise HTTPException(status_code=404, detail=f"Track {track_id} not found")

    metadata["behavior"] = {
        "type": request.behavior,
        "confidence": request.confidence,
        "detected_at": time.time(),
    }
    metadata["updated_at"] = time.time()

    manager._metadata[track_id] = metadata
    manager._last_seen[track_id] = time.time()

    return {
        "status": "success",
        "message": f"Behavior '{request.behavior}' set on track {track_id}",
        "behavior": metadata["behavior"],
    }


@router.post("/track/{track_id}/zone")
async def add_zone(track_id: int, request: ZoneRequest):
    """Add a zone to track's visited zones.

    Args:
        track_id: Track ID
        request: Zone request

    Returns:
        Success message
    """
    manager = get_metadata_manager()
    metadata = manager.get_track_metadata(track_id)

    if metadata is None:
        raise HTTPException(status_code=404, detail=f"Track {track_id} not found")

    if "zones_visited" not in metadata:
        metadata["zones_visited"] = []

    # Handle set/list
    zones = metadata["zones_visited"]
    if isinstance(zones, set):
        zones = list(zones)
        metadata["zones_visited"] = zones

    if request.zone_name not in zones:
        zones.append(request.zone_name)
        metadata["updated_at"] = time.time()

    manager._metadata[track_id] = metadata
    manager._last_seen[track_id] = time.time()

    return {
        "status": "success",
        "message": f"Zone '{request.zone_name}' added to track {track_id}",
        "zones_visited": zones,
    }


# ============================================================================
# MANAGEMENT ENDPOINTS
# ============================================================================


@router.post("/cleanup")
async def cleanup_expired():
    """Clean up metadata for expired tracks.

    Returns:
        Number of tracks cleaned up
    """
    manager = get_metadata_manager()
    count = manager.cleanup_expired()

    return {
        "status": "success",
        "message": f"Cleaned up {count} expired tracks",
        "cleaned_count": count,
    }


@router.post("/save")
async def save_metadata():
    """Save metadata to persistence file.

    Returns:
        Success message
    """
    manager = get_metadata_manager()
    manager.save_metadata()

    return {
        "status": "success",
        "message": "Metadata saved to disk",
        "path": manager.persistence_path,
    }


@router.delete("/clear")
async def clear_all_metadata():
    """Clear all metadata (use with caution).

    Returns:
        Success message
    """
    manager = get_metadata_manager()
    manager.clear_all()

    return {"status": "success", "message": "All metadata cleared"}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _format_metadata(track_id: int, metadata: dict) -> str:
    """Format metadata for human-readable display.

    Args:
        track_id: Track ID
        metadata: Metadata dictionary

    Returns:
        Formatted string
    """
    lines = [f"=== Track {track_id} Metadata ==="]

    # Timestamps
    created = metadata.get("created_at")
    updated = metadata.get("updated_at")

    if created:
        created_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created))
        lines.append(f"Created: {created_str}")

    if updated:
        updated_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(updated))
        age = time.time() - updated
        lines.append(f"Updated: {updated_str} ({age:.1f}s ago)")

    # Tags
    tags = metadata.get("tags", [])
    if tags:
        lines.append(f"Tags: {', '.join(tags)}")

    # Behavior
    behavior = metadata.get("behavior")
    if behavior:
        behavior_type = behavior.get("type")
        confidence = behavior.get("confidence", 1.0)
        lines.append(f"Behavior: {behavior_type} (confidence: {confidence:.2f})")

    # Zones
    zones = metadata.get("zones_visited", [])
    if isinstance(zones, set):
        zones = list(zones)
    if zones:
        lines.append(f"Zones Visited: {', '.join(zones)}")

    # Attributes
    attributes = metadata.get("attributes", {})
    if attributes:
        lines.append("Attributes:")
        for key, value in attributes.items():
            lines.append(f"  - {key}: {value}")

    # Alerts
    alerts = metadata.get("alerts", [])
    if alerts:
        lines.append(f"Alerts ({len(alerts)}):")
        for alert in alerts[-5:]:  # Show last 5 alerts
            alert_time = time.strftime(
                "%H:%M:%S", time.localtime(alert.get("timestamp", 0))
            )
            severity = alert.get("severity", "info").upper()
            alert_type = alert.get("type", "unknown")
            message = alert.get("message", "")
            lines.append(f"  [{alert_time}] {severity} {alert_type}: {message}")

    # Notes
    notes = metadata.get("notes", [])
    if notes:
        lines.append(f"Notes ({len(notes)}):")
        for note in notes[-5:]:  # Show last 5 notes
            note_time = time.strftime(
                "%H:%M:%S", time.localtime(note.get("timestamp", 0))
            )
            author = note.get("author", "unknown")
            text = note.get("text", "")
            lines.append(f"  [{note_time}] {author}: {text}")

    # Custom metadata
    custom = metadata.get("custom", {})
    if custom:
        lines.append("Custom:")
        for key, value in custom.items():
            lines.append(f"  - {key}: {value}")

    return "\n".join(lines)
