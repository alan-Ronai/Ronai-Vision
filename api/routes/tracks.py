from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()

# Module-level ReIDStore instance (set by application on startup)
_reid_store = None


def set_reid_store(store):
    global _reid_store
    _reid_store = store


class QueryRequest(BaseModel):
    embedding: List[float] = Field(
        ..., description="L2-normalized embedding vector", example=[0.1, 0.2, 0.3, 0.4]
    )
    topk: Optional[int] = Field(
        5, description="Number of nearest neighbors to return", example=3
    )


class QueryResult(BaseModel):
    gid: Optional[int]
    distance: float


class DecayRequest(BaseModel):
    max_age_seconds: float = Field(
        3600.0,
        description="Maximum age (seconds) before an entry is considered stale",
        example=3600,
    )


@router.post(
    "/query",
    response_model=List[QueryResult],
    summary="Query nearest global IDs",
    description="Provide an L2-normalized embedding and get nearest global IDs with distances.",
)
def query_embedding(body: QueryRequest):
    if _reid_store is None:
        raise HTTPException(status_code=500, detail="ReIDStore not configured")
    import numpy as np

    emb = np.asarray(body.embedding, dtype=np.float32)
    results = _reid_store.query(emb, topk=body.topk or 5)
    return [{"gid": gid, "distance": dist} for gid, dist in results]


@router.get("/{gid}", summary="Get metadata for a global id")
def get_gid_meta(gid: int):
    if _reid_store is None:
        raise HTTPException(status_code=500, detail="ReIDStore not configured")
    meta = _reid_store.id_to_meta.get(gid)
    if meta is None:
        raise HTTPException(status_code=404, detail="gid not found")
    return {"gid": gid, "meta": meta}


@router.get("/", summary="List active global IDs")
def list_active():
    if _reid_store is None:
        raise HTTPException(status_code=500, detail="ReIDStore not configured")
    active = [gid for gid, m in _reid_store.id_to_meta.items() if m.get("active", True)]
    return {"active_gids": active, "count": len(active)}


@router.delete("/{gid}", summary="Deactivate (remove) a global id")
def delete_gid(gid: int):
    if _reid_store is None:
        raise HTTPException(status_code=500, detail="ReIDStore not configured")
    ok = _reid_store.remove_gid(gid)
    if not ok:
        raise HTTPException(status_code=404, detail="gid not found or already inactive")
    return {"removed": True, "gid": gid}


@router.post(
    "/decay",
    summary="Decay stale global ids",
    description="Deactivate global IDs that have not been seen within the given max_age_seconds.",
)
def decay_stale(body: DecayRequest):
    if _reid_store is None:
        raise HTTPException(status_code=500, detail="ReIDStore not configured")
    removed = _reid_store.decay_stale(body.max_age_seconds)
    return {"removed": removed}
