"""Simple IoU-based centroid tracker for frame-to-frame tracking."""

import numpy as np
from typing import List, Dict, Tuple
from services.tracker.base_tracker import BaseTracker, Track


class CentroidTracker(BaseTracker):
    """Lightweight centroid-based tracker using IoU for association.

    Tracks objects by matching centroids between frames.
    Good for simple scenes and as a fallback tracker.
    """

    def __init__(self, max_disappeared: int = 50, max_distance: float = 100.0):
        """
        Args:
            max_disappeared: max frames to keep a track alive without detections
            max_distance: max centroid distance (pixels) for association
        """
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.next_track_id = 1
        self.tracks: Dict[int, Track] = {}
        self.disappeared: Dict[int, int] = {}

    def update(
        self,
        boxes: np.ndarray,
        class_ids: np.ndarray,
        confidences: np.ndarray,
    ) -> List[Track]:
        """Update tracker with new detections."""
        if len(boxes) == 0:
            # No detections: increment disappeared counters
            for track_id in list(self.disappeared.keys()):
                self.disappeared[track_id] += 1
                if self.disappeared[track_id] > self.max_disappeared:
                    del self.tracks[track_id]
                    del self.disappeared[track_id]
            return list(self.tracks.values())

        # Compute centroids of current detections
        centroids = self._get_centroids(boxes)

        # Compute IoU between existing tracks and detections
        if len(self.tracks) > 0:
            track_ids = list(self.tracks.keys())
            track_boxes = np.array([self.tracks[tid].box for tid in track_ids])

            # Simple centroid distance matching
            matches, unmatched_dets, unmatched_tracks = self._match_detections(
                track_boxes, boxes, centroids
            )

            # Update matched tracks
            for track_id, det_idx in matches:
                self.tracks[track_id].box = boxes[det_idx]
                self.tracks[track_id].class_id = int(class_ids[det_idx])
                self.tracks[track_id].confidence = float(confidences[det_idx])
                self.tracks[track_id].hits += 1
                self.tracks[track_id].age += 1
                self.disappeared[track_id] = 0

            # Create new tracks for unmatched detections
            for det_idx in unmatched_dets:
                track_id = self.next_track_id
                self.next_track_id += 1
                track = Track(
                    track_id=track_id,
                    box=boxes[det_idx],
                    class_id=int(class_ids[det_idx]),
                    confidence=float(confidences[det_idx]),
                )
                self.tracks[track_id] = track
                self.disappeared[track_id] = 0

            # Mark unmatched tracks as disappeared
            for track_id in unmatched_tracks:
                self.disappeared[track_id] += 1
                if self.disappeared[track_id] > self.max_disappeared:
                    del self.tracks[track_id]
                    del self.disappeared[track_id]
        else:
            # No existing tracks: create new ones for all detections
            for i, box in enumerate(boxes):
                track_id = self.next_track_id
                self.next_track_id += 1
                track = Track(
                    track_id=track_id,
                    box=box,
                    class_id=int(class_ids[i]),
                    confidence=float(confidences[i]),
                )
                self.tracks[track_id] = track
                self.disappeared[track_id] = 0

        return list(self.tracks.values())

    def get_active_tracks(self) -> List[Track]:
        """Return currently active tracks."""
        return list(self.tracks.values())

    @staticmethod
    def _get_centroids(boxes: np.ndarray) -> np.ndarray:
        """Compute centroids from bounding boxes."""
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return np.column_stack([cx, cy])

    def _match_detections(
        self,
        track_boxes: np.ndarray,
        det_boxes: np.ndarray,
        det_centroids: np.ndarray,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Match detections to tracks using centroid distance."""
        track_centroids = self._get_centroids(track_boxes)

        matches = []
        unmatched_dets = list(range(len(det_boxes)))
        unmatched_tracks = list(range(len(track_boxes)))

        # Greedy matching: closest centroid first
        while unmatched_dets and unmatched_tracks:
            min_dist = float("inf")
            best_match = None

            for track_idx in unmatched_tracks:
                for det_idx in unmatched_dets:
                    dist = np.linalg.norm(
                        track_centroids[track_idx] - det_centroids[det_idx]
                    )
                    if dist < min_dist:
                        min_dist = dist
                        best_match = (track_idx, det_idx)

            if min_dist <= self.max_distance and best_match:
                track_idx, det_idx = best_match
                track_id = list(self.tracks.keys())[track_idx]
                matches.append((track_id, det_idx))
                unmatched_tracks.remove(track_idx)
                unmatched_dets.remove(det_idx)
            else:
                break

        return matches, unmatched_dets, unmatched_tracks
