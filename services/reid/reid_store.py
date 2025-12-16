"""FAISS-backed cross-camera ReID store.

Stores L2-normalized embeddings in a FAISS index and maps them to
global IDs (persistent mapping in memory). Provides upsert and query
functions for assigning global IDs to incoming track embeddings.
"""

import os
import time
import numpy as np
from typing import Tuple, List, Dict, Optional
import faiss


class ReIDStore:
    def __init__(self, dim: int = None):
        self.dim = dim
        self.index = None
        # id_to_meta stores metadata for each global id. Each entry is a dict
        # with keys: 'history' (list of seen meta entries), 'last_seen' (timestamp),
        # and 'active' (bool).
        self.id_to_meta: Dict[int, Dict] = {}
        # mapping from faiss internal index position -> global id
        self.index_to_gid: List[int] = []
        self.next_global_id = 1
        # distance threshold for considering two embeddings the same (squared L2)
        self.match_threshold = 0.5
        # maximum number of history entries to keep per gid
        self.max_history = 10

    def _ensure_index(self, dim: int):
        if self.index is None:
            self.dim = dim
            self.index = faiss.IndexFlatL2(dim)

    def upsert(self, embeddings: np.ndarray, meta: List[Dict]) -> List[int]:
        """Insert embeddings and return assigned global IDs.

        embeddings: (N, D) float32 L2-normalized
        meta: list of metadata dict (camera_id, track_id, timestamp)
        """
        import logging

        logger = logging.getLogger(__name__)

        if embeddings is None or len(embeddings) == 0:
            return []

        embeddings = embeddings.astype(np.float32)
        N, D = embeddings.shape
        self._ensure_index(D)

        gids: List[int] = []
        now = time.time()

        # If index is empty, add all embeddings and assign new IDs
        if self.index.ntotal == 0:
            self.index.add(embeddings)
            for i in range(N):
                gid = self.next_global_id
                self.next_global_id += 1
                # initialize meta history
                self.id_to_meta[gid] = {
                    "history": [meta[i]],
                    "last_seen": now,
                    "active": True,
                    # persist latest vector (1D float32)
                    "vector": embeddings[i].astype(np.float32).reshape(-1),
                }
                self.index_to_gid.append(gid)
                gids.append(gid)
            logger.info(
                f"ReIDStore: Initial insert of {N} embeddings, assigned GIDs: {gids}"
            )
            return gids

        # Search for nearest neighbors
        Dists, Ids = self.index.search(embeddings, 1)

        # Prepare list of embeddings that need to be added (those without a match)
        to_add_list = []
        to_add_meta = []

        for i in range(N):
            dist = float(Dists[i, 0])
            idx = int(Ids[i, 0]) if Ids.size > 0 else -1
            if idx >= 0 and dist < self.match_threshold:
                # matched existing vector -> reuse its global id
                matched_gid = self.index_to_gid[idx]
                # append to history and update last_seen
                entry = self.id_to_meta.get(matched_gid)
                if entry is None:
                    entry = {
                        "history": [meta[i]],
                        "last_seen": now,
                        "active": True,
                        "vector": embeddings[i].astype(np.float32).reshape(-1),
                    }
                else:
                    h = entry.setdefault("history", [])
                    h.append(meta[i])
                    if len(h) > self.max_history:
                        entry["history"] = h[-self.max_history :]
                    entry["last_seen"] = now
                    entry["active"] = True
                    # update persisted latest vector
                    entry["vector"] = embeddings[i].astype(np.float32).reshape(-1)
                self.id_to_meta[matched_gid] = entry
                gids.append(matched_gid)
            else:
                # no match -> schedule to add
                to_add_list.append(embeddings[i : i + 1])
                to_add_meta.append(meta[i])

        # Add only the unmatched embeddings to the index and assign new gids
        if len(to_add_list) > 0:
            to_add_all = np.vstack(to_add_list).astype(np.float32)
            base = self.index.ntotal
            self.index.add(to_add_all)
            # assign gids for newly added vectors
            for i in range(len(to_add_list)):
                gid = self.next_global_id
                self.next_global_id += 1
                self.id_to_meta[gid] = {
                    "history": [to_add_meta[i]],
                    "last_seen": now,
                    "active": True,
                    "vector": to_add_all[i].astype(np.float32).reshape(-1),
                }
                # the faiss internal index position is base + i
                self.index_to_gid.append(gid)
                gids.append(gid)

        return gids

    def query(self, embedding: np.ndarray, topk: int = 5) -> List[Tuple[int, float]]:
        """Query nearest global IDs for a single embedding.

        Returns list of (global_id, distance).
        """
        if self.index is None or self.index.ntotal == 0:
            return []
        embedding = embedding.astype(np.float32).reshape(1, -1)
        Dists, Ids = self.index.search(embedding, topk)
        results: List[Tuple[int, float]] = []
        k = min(topk, Dists.shape[1])
        for i in range(k):
            idx = int(Ids[0, i])
            gid = self.index_to_gid[idx] if 0 <= idx < len(self.index_to_gid) else None
            results.append((gid, float(Dists[0, i])))
        return results

    def remove_gid(self, gid: int) -> bool:
        """Mark a global id as inactive and rebuild the index without it.

        Returns True if gid existed and was removed, False otherwise.
        """
        if gid not in self.id_to_meta:
            return False
        # mark inactive
        self.id_to_meta[gid]["active"] = False
        # rebuild index to drop inactive entries
        self._rebuild_index(exclude_inactive=True)
        return True

    def decay_stale(self, max_age_seconds: float) -> int:
        """Remove (deactivate) global ids not seen within max_age_seconds.

        Returns number of ids removed/deactivated.
        Note: This rebuilds the FAISS index, which can be expensive.
        """
        now = time.time()
        removed = 0
        for gid, meta in list(self.id_to_meta.items()):
            last = meta.get("last_seen", 0)
            if (now - last) > max_age_seconds and meta.get("active", True):
                meta["active"] = False
                removed += 1

        if removed > 0:
            self._rebuild_index(exclude_inactive=True)

        return removed

    def _rebuild_index(self, exclude_inactive: bool = True):
        """Rebuild the FAISS index from current id_to_meta entries.

        If `exclude_inactive` is True, entries with `active=False` are skipped.
        """
        # collect embeddings in same order as new index_to_gid
        gids_to_keep = []
        vectors = []
        # We don't currently store vectors in id_to_meta; the store assumes the
        # FAISS index is authoritative. To rebuild we need to keep a copy of the
        # vectors elsewhere or rebuild from application-level track features.
        # For now we will rebuild by keeping vectors from the current index if
        # available and filtering by id_to_meta active flag.
        if self.index is None or self.index.ntotal == 0:
            # nothing to rebuild
            self.index = None
            self.index_to_gid = []
            return

        # Rebuild from stored vectors in id_to_meta (preferred);
        # fall back to reading from current FAISS index only if necessary.
        for gid, meta in self.id_to_meta.items():
            if exclude_inactive and not meta.get("active", True):
                continue
            vec = meta.get("vector")
            if vec is not None:
                arr = np.asarray(vec, dtype=np.float32).reshape(1, -1)
                vectors.append(arr)
                gids_to_keep.append(gid)

        # If we found no stored vectors, try to extract from the existing index
        if len(vectors) == 0 and self.index is not None and self.index.ntotal > 0:
            try:
                total = self.index.ntotal
                for pos in range(total):
                    gid = self.index_to_gid[pos]
                    meta = self.id_to_meta.get(gid, {})
                    if exclude_inactive and not meta.get("active", True):
                        continue
                    vec = self.index.reconstruct(pos)
                    vectors.append(np.array(vec, dtype=np.float32).reshape(1, -1))
                    gids_to_keep.append(gid)
            except Exception:
                # reconstruct unsupported or failed; clear index
                vectors = []
                gids_to_keep = []

        if len(vectors) == 0:
            # clear index
            self.index = None
            self.index_to_gid = []
            return

        stacked = np.vstack(vectors).astype(np.float32)
        # create a fresh index
        self.index = faiss.IndexFlatL2(stacked.shape[1])
        self.index.add(stacked)
        self.index_to_gid = gids_to_keep


# Global singleton instance
_reid_store_instance: Optional[ReIDStore] = None


def get_reid_store() -> ReIDStore:
    """Get or create global ReIDStore singleton.

    Returns:
        ReIDStore instance (shared across all modules)
    """
    global _reid_store_instance
    if _reid_store_instance is None:
        _reid_store_instance = ReIDStore()
    return _reid_store_instance


def reset_reid_store():
    """Reset global ReIDStore singleton (for testing)."""
    global _reid_store_instance
    _reid_store_instance = None
