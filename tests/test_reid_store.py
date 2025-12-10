import time
import numpy as np

from services.reid.reid_store import ReIDStore


def l2_normalize(v: np.ndarray) -> np.ndarray:
    v = np.array(v, dtype=np.float32)
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def test_upsert_and_query_and_decay():
    store = ReIDStore()

    v1 = l2_normalize([1.0, 0.0, 0.0])
    v2 = l2_normalize([0.0, 1.0, 0.0])

    # initial upsert two distinct vectors
    gids = store.upsert(
        np.vstack([v1, v2]),
        meta=[{"camera_id": 1, "track_id": 1}, {"camera_id": 1, "track_id": 2}],
    )
    assert len(gids) == 2

    gid1, gid2 = gids[0], gids[1]
    assert gid1 != gid2

    # query v1 should return gid1 as top-1
    res = store.query(v1, topk=1)
    assert len(res) == 1
    assert res[0][0] == gid1

    # upsert a duplicate of v1 -> should reuse gid1
    gids2 = store.upsert(np.vstack([v1]), meta=[{"camera_id": 2, "track_id": 5}])
    assert len(gids2) == 1
    assert gids2[0] == gid1

    # mark gid1 as old and decay
    store.id_to_meta[gid1]["last_seen"] = time.time() - 3600
    removed = store.decay_stale(max_age_seconds=60)
    assert removed >= 1
