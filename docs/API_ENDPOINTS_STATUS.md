# API Endpoints Status

## Implemented Endpoints Summary

### Metadata Endpoints (`/api/metadata`)

-   **GET /api/metadata/all** - Get all stored metadata ✅

    -   Returns all tracks with their metadata
    -   File: [api/routes/metadata.py](../api/routes/metadata.py#L186)

-   **GET /api/metadata/stats** - Get metadata statistics ✅
    -   Returns statistics about stored metadata (total_tracks, total_notes, total_alerts, tracks_by_class, recent_tracks, oldest_track_age)
    -   File: [api/routes/metadata.py](../api/routes/metadata.py#L209)

### Tracks Endpoints (`/api/tracks`)

-   **GET /api/tracks/** - List active global IDs ✅

    -   Returns list of active global IDs (GUIDs) and count
    -   File: [api/routes/tracks.py](../api/routes/tracks.py#L66)

-   **POST /api/tracks/query** - Query nearest global IDs by embedding

    -   Returns GIDs matching embeddings with distances
    -   File: [api/routes/tracks.py](../api/routes/tracks.py#L37)

-   **GET /api/tracks/{gid}** - Get metadata for a global ID

    -   Returns metadata history and status for a specific GUID
    -   File: [api/routes/tracks.py](../api/routes/tracks.py#L53)

-   **DELETE /api/tracks/{gid}** - Deactivate a global ID

    -   Marks a GUID as inactive
    -   File: [api/routes/tracks.py](../api/routes/tracks.py#L78)

-   **POST /api/tracks/decay** - Decay stale global IDs
    -   Deactivates GUIDs not seen within max_age_seconds
    -   File: [api/routes/tracks.py](../api/routes/tracks.py#L88)

## Track Persistence System

### Overview

Tracks, metadata, and GUIDs now persist even when MJPEG streams are turned off. They are retained for a configurable time period and automatically restored when cameras come back online.

### Key Features

1. **Metadata Persistence** (via MetadataManager)

    - Stores track metadata in memory with automatic JSON serialization
    - Saves to `output/track_metadata.json` on shutdown
    - Auto-loads on startup
    - TTL: 1 hour after track disappears (configurable)
    - File: [services/tracker/metadata_manager.py](../services/tracker/metadata_manager.py)

2. **GUID Persistence** (via ReIDStore)

    - GUIDs and their embeddings stored in FAISS index
    - History of sightings maintained per GUID
    - `last_seen` timestamp tracks when each GUID was last observed
    - Marked as inactive (not deleted) when not seen for >1 hour
    - Reactivated automatically if same person re-appears (embedding match)
    - File: [services/reid/reid_store.py](../services/reid/reid_store.py)

3. **Server Lifecycle Management**
    - **Startup**: Loads persisted metadata, cleans up expired tracks
    - **Shutdown**: Saves metadata, decays stale GUIDs
    - File: [api/server.py](../api/server.py#L30)

### How It Works

#### When Camera Goes Offline

1. ReIDStore keeps GUID entries with `last_seen` timestamp
2. MetadataManager keeps all track metadata
3. On server shutdown, both are persisted to disk
4. On server restart, both are loaded back into memory

#### When Camera Comes Back Online

1. New detections are processed and compared to stored embeddings
2. ReIDStore's FAISS index finds matching GUIDs using `upsert()`
3. If embedding is close enough (distance < match_threshold), same GUID is reused
4. Metadata is auto-restored from MetadataManager
5. ReID history is updated with new sighting

#### Automatic Cleanup

-   Tracks not seen for >1 hour (TTL) are marked as inactive
-   Can be cleaned up via POST `/api/tracks/decay?max_age_seconds=3600`
-   `GET /api/metadata/stats` shows age of oldest track

### Configuration

-   **Metadata TTL**: Default 3600 seconds (1 hour)
-   **Persistence file**: `output/track_metadata.json`
-   **ReID match threshold**: 0.5 (squared L2 distance)
-   **ReID history per GUID**: Last 10 sightings

### API for Track Recovery

#### Example: List all active GUIDs

```bash
curl http://localhost:8000/api/tracks/
# Returns: {"active_gids": [1, 2, 3], "count": 3}
```

#### Example: Get stats including oldest track age

```bash
curl http://localhost:8000/api/metadata/stats
# Returns:
# {
#   "total_tracks": 50,
#   "total_notes": 0,
#   "total_alerts": 12,
#   "tracks_by_class": {"0": 10, "1": 40},
#   "recent_tracks": 15,
#   "oldest_track_age": 1234.5
# }
```

#### Example: Decay stale GUIDs (1 hour)

```bash
curl -X POST http://localhost:8000/api/tracks/decay \
  -H "Content-Type: application/json" \
  -d '{"max_age_seconds": 3600}'
# Returns: {"removed": 5}
```

#### Example: Query for matching GUIDs by embedding

```bash
curl -X POST http://localhost:8000/api/tracks/query \
  -H "Content-Type: application/json" \
  -d '{
    "embedding": [0.1, 0.2, 0.3, ...],  # L2-normalized 128-dim vector
    "topk": 5
  }'
# Returns: [{"gid": 1, "distance": 0.2}, {"gid": 5, "distance": 0.3}, ...]
```

## Implementation Details

### Recent Changes

1. **analyzer.py**: Fixed unused imports and API compatibility

    - Removed `base64` and `io.BytesIO` unused imports
    - Added robust error handling for google.generativeai API

2. **server.py**: Enhanced lifespan management
    - Added metadata loading and cleanup on startup
    - Added metadata saving and ReID decay on shutdown
    - Integrated persistence file path and TTL cleanup

### Testing Track Persistence

```bash
# 1. Start server with runner
START_RUNNER=true python -m uvicorn api.server:app --reload

# 2. Let it detect some tracks
# Wait 30+ seconds for tracks to be created

# 3. Stop server (Ctrl+C)
# Metadata saved to output/track_metadata.json

# 4. Restart server
# Metadata loaded back automatically

# 5. Check that GUIDs are still present
curl http://localhost:8000/api/tracks/
```
