"""Cross-camera ReID: global track deduplication and re-identification.

Handles upserting per-camera tracks into a global ReID store for cross-camera
matching and deduplication.
"""

import numpy as np
from typing import List, Dict, Optional

from services.reid.reid_store import ReIDStore


class CrossCameraReID:
    """Global ReID store manager.

    Maintains a FAISS-based embedding store for deduplicating and matching
    tracks across cameras.
    """

    def __init__(self):
        """Initialize global ReID store."""
        self.store = ReIDStore()
        self.global_id_map = {}  # Maps (camera_id, local_track_id) -> global_id

    def upsert_tracks(
        self,
        camera_id: str,
        tracks: List,
        features: Optional[List],
        boxes: np.ndarray,
    ) -> Dict[int, int]:
        """Upsert per-camera tracks to global store.

        Args:
            camera_id: Camera identifier
            tracks: List of Track objects from tracker
            features: List of feature arrays (can contain None for non-person)
            boxes: (N, 4) detection boxes

        Returns:
            Dict mapping local_track_id -> global_track_id
        """
        local_to_global = {}

        if features is None or len(features) == 0:
            return local_to_global

        # Extract embeddings for each track
        meta_list = []
        emb_list = []
        local_ids = []

        for track in tracks:
            track_id = track.track_id

            # Find matching detection by box similarity
            matched_idx = None
            for i, box in enumerate(boxes):
                if np.allclose(box, track.box, atol=2.0):
                    matched_idx = i
                    break

            # Only upsert if we have valid features (not None)
            if matched_idx is not None and matched_idx < len(features):
                feat = features[matched_idx]
                if feat is not None and isinstance(feat, np.ndarray):
                    # Check if feature is not all zeros
                    if not np.allclose(feat, 0):
                        emb_list.append(feat)
                        meta_list.append(
                            {
                                "camera_id": camera_id,
                                "track_id": track_id,
                            }
                        )
                        local_ids.append(track_id)

        # Upsert to global store
        if emb_list:
            emb_array = np.vstack(emb_list).astype(np.float32)
            global_ids = self.store.upsert(emb_array, meta_list)

            # Build mapping
            for local_id, global_id in zip(local_ids, global_ids):
                self.global_id_map[(camera_id, local_id)] = global_id
                local_to_global[local_id] = global_id

        return local_to_global

    def get_global_id(self, camera_id: str, local_track_id: int) -> Optional[int]:
        """Get global track ID for a local track.

        Args:
            camera_id: Camera identifier
            local_track_id: Track ID from per-camera tracker

        Returns:
            Global track ID or None if not found
        """
        return self.global_id_map.get((camera_id, local_track_id))
