"""Lightweight BoT-SORT-like tracker.

This is a simplified implementation inspired by BoT-SORT: it combines
motion (centroid distance) with appearance (cosine distance of ReID
embeddings) for assignment. It's intentionally lightweight so it works
on CPU for development. It is NOT a full BoT-SORT reproduction but
provides the same inputs/outputs for the pipeline.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np

from services.tracker.base_tracker import BaseTracker, Track


def simple_hungarian_assignment(cost_matrix):
    """
    Simplified greedy assignment (approximation of Hungarian algorithm).
    Returns list of (row_idx, col_idx) pairs for assignment.
    """
    cost = np.array(cost_matrix, dtype=np.float32)
    n_rows, n_cols = cost.shape

    if n_rows == 0 or n_cols == 0:
        return []

    assigned = []
    used_rows = set()
    used_cols = set()

    while len(used_rows) < n_rows and len(used_cols) < n_cols:
        min_cost = np.inf
        best_i, best_j = -1, -1

        for i in range(n_rows):
            if i in used_rows:
                continue
            for j in range(n_cols):
                if j in used_cols:
                    continue
                if cost[i, j] < min_cost:
                    min_cost = cost[i, j]
                    best_i, best_j = i, j

        if best_i >= 0:
            assigned.append((best_i, best_j))
            used_rows.add(best_i)
            used_cols.add(best_j)
        else:
            break

    return assigned


def _centroid(box: np.ndarray) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 1.0
    na = np.linalg.norm(a) + 1e-6
    nb = np.linalg.norm(b) + 1e-6
    return 1.0 - float(np.dot(a, b) / (na * nb))


class BoTSortTracker(BaseTracker):
    """Simple per-camera tracker that uses embeddings + centroid distance.

    Params:
        max_lost: frames to keep a track without matching
        lambda_app: weight for appearance cost (0..1), higher favors appearance
        max_cost: maximum cost allowed for a match
    """

    def __init__(
        self, max_lost: int = 30, lambda_app: float = 0.7, max_cost: float = 0.7
    ):
        self.max_lost = max_lost
        self.lambda_app = lambda_app
        self.max_cost = max_cost

        self.next_id = 1
        self.tracks: Dict[int, Track] = {}
        self.lost: Dict[int, int] = {}
        self.track_features: Dict[int, np.ndarray] = {}

    def update(
        self,
        boxes: np.ndarray,
        class_ids: np.ndarray,
        confidences: np.ndarray,
        features: Optional[np.ndarray] = None,
    ) -> List[Track]:
        N = len(boxes)
        if N == 0:
            # increment lost counters
            to_delete = []
            for tid in list(self.lost.keys()):
                self.lost[tid] += 1
                if self.lost[tid] > self.max_lost:
                    to_delete.append(tid)
            for tid in to_delete:
                del self.tracks[tid]
                del self.lost[tid]
                if tid in self.track_features:
                    del self.track_features[tid]
            return list(self.tracks.values())

        det_centroids = np.array([_centroid(b) for b in boxes])

        if len(self.tracks) == 0:
            # create new tracks
            for i in range(N):
                tid = self.next_id
                self.next_id += 1
                tr = Track(
                    track_id=tid,
                    box=boxes[i],
                    class_id=int(class_ids[i]),
                    confidence=float(confidences[i]),
                )
                self.tracks[tid] = tr
                self.lost[tid] = 0
                if features is not None:
                    self.track_features[tid] = features[i]
            return list(self.tracks.values())

        # Build cost matrix between existing tracks and detections
        track_ids = list(self.tracks.keys())
        M = len(track_ids)
        cost = np.zeros((M, N), dtype=np.float32)

        for i, tid in enumerate(track_ids):
            tb = self.tracks[tid].box
            tcent = np.array(_centroid(tb))
            tfeat = self.track_features.get(tid, None)
            for j in range(N):
                dcent = np.linalg.norm(tcent - det_centroids[j])
                # normalize centroid distance by image diagonal approx (assume 1080p)
                dcent = dcent / 1500.0
                dapp = _cosine(tfeat, features[j]) if features is not None else 1.0
                c = (1.0 - self.lambda_app) * dcent + self.lambda_app * dapp
                cost[i, j] = c

        row_ind, col_ind = (
            zip(*simple_hungarian_assignment(cost)) if cost.size > 0 else ([], [])
        )

        assigned_tracks = set()
        assigned_dets = set()

        # Apply matches
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] > self.max_cost:
                continue
            tid = track_ids[r]
            self.tracks[tid].box = boxes[c]
            self.tracks[tid].class_id = int(class_ids[c])
            self.tracks[tid].confidence = float(confidences[c])
            self.tracks[tid].hits += 1
            self.lost[tid] = 0
            if features is not None:
                self.track_features[tid] = features[c]
            assigned_tracks.add(tid)
            assigned_dets.add(c)

        # Unmatched detections -> new tracks
        for j in range(N):
            if j in assigned_dets:
                continue
            tid = self.next_id
            self.next_id += 1
            tr = Track(
                track_id=tid,
                box=boxes[j],
                class_id=int(class_ids[j]),
                confidence=float(confidences[j]),
            )
            self.tracks[tid] = tr
            self.lost[tid] = 0
            if features is not None:
                self.track_features[tid] = features[j]

        # Unmatched tracks -> increment lost
        for tid in list(track_ids):
            if tid not in assigned_tracks:
                self.lost[tid] += 1
                if self.lost[tid] > self.max_lost:
                    del self.tracks[tid]
                    del self.lost[tid]
                    if tid in self.track_features:
                        del self.track_features[tid]

        return list(self.tracks.values())

    def get_active_tracks(self) -> List[Track]:
        return list(self.tracks.values())
