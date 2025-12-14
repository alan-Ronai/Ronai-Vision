"""Frame overlay renderer for visualization and output."""

import cv2
import numpy as np
from typing import List, Optional, Tuple
from services.tracker.base_tracker import Track


class FrameRenderer:
    """Renders bounding boxes, track IDs, and segmentation masks on frames."""

    def __init__(
        self,
        font_scale: float = 0.6,
        thickness: int = 2,
        bbox_color: Tuple[int, int, int] = (0, 255, 0),
        text_color: Tuple[int, int, int] = (255, 255, 255),
        text_bg_color: Tuple[int, int, int] = (0, 0, 0),
        show_masks: bool = False,
        show_fps: bool = True,
    ):
        """
        Args:
            font_scale: OpenCV font scale
            thickness: line thickness for boxes and text
            bbox_color: BGR color for bounding boxes
            text_color: BGR color for text
            text_bg_color: BGR color for text background
            show_fps: Whether to display FPS counter
        """
        self.font_scale = font_scale
        self.thickness = thickness
        self.bbox_color = bbox_color
        self.text_color = text_color
        self.text_bg_color = text_bg_color
        # Whether to overlay segmentation masks by default. Keep False for clean box+ID output.
        self.show_masks = show_masks
        self.show_fps = show_fps

    def render_detections(
        self,
        frame: np.ndarray,
        boxes: np.ndarray,
        class_ids: np.ndarray,
        confidences: np.ndarray,
        class_names: List[str],
    ) -> np.ndarray:
        """Draw detections on frame.

        Args:
            frame: (H, W, 3) BGR numpy array
            boxes: (N, 4) array of [x1, y1, x2, y2]
            class_ids: (N,) integer class IDs
            confidences: (N,) confidence scores
            class_names: list of class name strings

        Returns:
            Rendered frame
        """
        output = frame.copy()

        for box, class_id, conf in zip(boxes, class_ids, confidences):
            x1, y1, x2, y2 = box.astype(np.int32)
            class_name = (
                class_names[int(class_id)] if class_id < len(class_names) else "unknown"
            )
            label = f"{class_name} {conf:.2f}"

            # Draw box
            cv2.rectangle(output, (x1, y1), (x2, y2), self.bbox_color, self.thickness)

            # Draw label with background
            self._draw_label(output, label, (x1, y1 - 5))

        return output

    def render_tracks(
        self,
        frame: np.ndarray,
        tracks: List[Track],
        class_names: List[str],
    ) -> np.ndarray:
        """Draw tracks (with IDs) on frame.

        Args:
            frame: (H, W, 3) BGR numpy array
            tracks: list of Track objects
            class_names: list of class name strings

        Returns:
            Rendered frame
        """
        output = frame.copy()

        for track in tracks:
            x1, y1, x2, y2 = track.box.astype(np.int32)
            class_name = (
                class_names[track.class_id]
                if track.class_id < len(class_names)
                else "unknown"
            )

            # Show global ID if available, otherwise show local track ID
            if hasattr(track, "global_id") and track.global_id is not None:
                label = f"GID:{track.global_id} {class_name} {track.confidence:.2f}"
            else:
                label = f"ID:{track.track_id} {class_name} {track.confidence:.2f}"

            # Draw box
            cv2.rectangle(output, (x1, y1), (x2, y2), self.bbox_color, self.thickness)

            # Draw label with background
            self._draw_label(output, label, (x1, y1 - 5))

            # Draw centroid
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            cv2.circle(output, (cx, cy), 4, self.bbox_color, -1)

        return output

    def render_masks(
        self,
        frame: np.ndarray,
        masks: np.ndarray,
        alpha: float = 0.3,
        color: Tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        """Overlay segmentation masks on frame.

        Args:
            frame: (H, W, 3) BGR numpy array
            masks: (N, H, W) binary or float masks
            alpha: blend factor [0, 1]
            color: BGR color for mask overlay

        Returns:
            Rendered frame with mask overlay
        """
        # If masks overlaying is disabled, return the original frame unchanged.
        if not self.show_masks:
            return frame

        output = frame.copy()

        for mask in masks:
            # Normalize mask to uint8 {0,255}
            if mask.max() <= 1.0:
                mask_uint8 = (mask * 255).astype(np.uint8)
            else:
                mask_uint8 = mask.astype(np.uint8)

            mask_bool = mask_uint8 > 0

            if not np.any(mask_bool):
                continue

            # Create overlay: same as output but colored only at mask pixels
            overlay = output.copy()
            overlay[mask_bool] = color

            # Blend overlay into output; since overlay only differs on mask,
            # only masked pixels will change.
            cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

        return output

    def _draw_label(
        self,
        frame: np.ndarray,
        label: str,
        pos: Tuple[int, int],
    ):
        """Draw text label with background."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        (w, h), baseline = cv2.getTextSize(label, font, self.font_scale, self.thickness)
        x, y = pos

        # Draw background rectangle
        cv2.rectangle(
            frame,
            (x, y - h - baseline),
            (x + w, y),
            self.text_bg_color,
            -1,
        )

        # Draw text
        cv2.putText(
            frame,
            label,
            (x, y - baseline),
            font,
            self.font_scale,
            self.text_color,
            self.thickness,
        )

    def render_fps(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """Draw FPS counter on frame.

        Args:
            frame: (H, W, 3) BGR numpy array
            fps: Current FPS value

        Returns:
            Frame with FPS overlay
        """
        if not self.show_fps or fps is None:
            return frame

        output = frame.copy()

        # FPS text in top-right corner
        fps_text = f"FPS: {fps:.1f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2

        # Get text size
        (w, h), baseline = cv2.getTextSize(fps_text, font, font_scale, thickness)

        # Position in top-right with padding
        h_img, w_img = frame.shape[:2]
        x = w_img - w - 15
        y = h + 15

        # Draw semi-transparent background
        overlay = output.copy()
        cv2.rectangle(
            overlay,
            (x - 5, y - h - baseline - 5),
            (x + w + 5, y + baseline + 5),
            (0, 0, 0),
            -1,
        )
        cv2.addWeighted(overlay, 0.6, output, 0.4, 0, output)

        # Draw FPS text in bright green
        cv2.putText(
            output,
            fps_text,
            (x, y),
            font,
            font_scale,
            (0, 255, 0),  # Bright green
            thickness,
        )

        return output
