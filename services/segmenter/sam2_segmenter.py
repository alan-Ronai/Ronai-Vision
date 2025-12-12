"""SAM2 segmenter that requires a local SAM2 checkpoint."""

"""SAM2 segmenter that requires a local SAM2 checkpoint."""

import os
import numpy as np
from typing import Optional
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from services.segmenter.base_segmenter import BaseSegmenter, SegmentationResult


class SAM2Segmenter(BaseSegmenter):
    """Wraps Meta's SAM2 model. Requires a local checkpoint `models/sam2_{type}.pt`.

    This implementation no longer provides a dummy fallback — initialization
    will raise if SAM2 or the checkpoint is not available.
    """

    def __init__(
        self,
        model_type: str = "tiny",
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
    ):
        """
        Initialize the SAM2 predictor.
        """

        self.model_type = model_type
        self.device = device
        self.predictor = None

        try:
            # Resolve device: if explicitly "cpu", use it; otherwise try the requested device
            if device == "cpu":
                self.device = "cpu"
            else:
                # Try the requested device (e.g., "cuda", "mps")
                # Validate it's available before using
                if device == "cuda" and torch.cuda.is_available():
                    self.device = "cuda"
                elif device == "mps":
                    # MPS (Metal Performance Shaders on macOS)
                    try:
                        if torch.backends.mps.is_available():
                            self.device = "mps"
                        else:
                            self.device = "cpu"
                    except Exception:
                        self.device = "cpu"
                else:
                    # Fallback to CPU if device not available
                    self.device = "cpu"

            model_cfg = {
                "tiny": "sam2_hiera_t.yaml",
                "small": "sam2_hiera_s.yaml",
                "base": "sam2_hiera_b+.yaml",
                "large": "sam2_hiera_l.yaml",
            }
            config_name = model_cfg.get(model_type, "sam2_hiera_t.yaml")

            if checkpoint_path is None:
                checkpoint_path = self._find_model(model_type)

            sam2_model = build_sam2(config_name, checkpoint_path, device=self.device)
            self.predictor = SAM2ImagePredictor(sam2_model)
            print(f"✓ SAM2 {model_type} loaded on {self.device}")
        except FileNotFoundError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to initialize SAM2: {e}") from e

    @staticmethod
    def _find_model(model_type: str) -> str:
        """Find model locally or raise error."""
        # Resolve the repository-root `models` directory relative to this file
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        models_dir = os.path.join(base, "models")
        checkpoint_path = os.path.join(models_dir, f"sam2_{model_type}.pt")

        if os.path.exists(checkpoint_path):
            return checkpoint_path

        raise FileNotFoundError(
            f"SAM2 {model_type} checkpoint not found at {checkpoint_path}."
            " Please download it and place it at this location."
        )

    def segment(
        self,
        frame: np.ndarray,
        boxes: Optional[np.ndarray] = None,
        points: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
    ) -> SegmentationResult:
        """Segment frame using prompts.

        Args:
            frame: BGR uint8 frame (as produced by cv2).
            boxes: array of [x1,y1,x2,y2] in image coordinates.
            points: array of point coordinates if available.
            labels: point labels (0 background, 1 foreground) if points provided.

        Returns:
            SegmentationResult with binary masks and scores.

        Raises:
            RuntimeError: if prediction fails.
        """

        def _to_numpy(x):
            # Convert either torch.Tensor or numpy-like outputs to numpy.ndarray
            if isinstance(x, np.ndarray):
                return x
            try:
                import torch as _torch

                if isinstance(x, _torch.Tensor):
                    return x.cpu().numpy()
            except Exception:
                pass
            # Fallback: attempt to convert
            return np.asarray(x)

        try:
            rgb_frame = frame[:, :, ::-1].copy()
            self.predictor.set_image(rgb_frame)

            masks_list = []
            scores_list = []

            # Clean up GPU memory BEFORE prediction to prevent cache accumulation.
            # SAM2's internal image cache can grow over many frames without clearing,
            # causing exponential slowdown. Clearing before set_image helps significantly.
            try:
                import torch as _torch

                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
                # Also try MPS (macOS GPU)
                try:
                    if hasattr(_torch, "mps") and _torch.mps.is_available():
                        _torch.mps.empty_cache()
                except Exception:
                    pass
            except Exception:
                pass

            if boxes is not None and len(boxes) > 0:
                boxes_tensor = torch.from_numpy(boxes).float().to(self.device)
                with torch.no_grad():
                    masks, iou_preds, _ = self.predictor.predict(
                        box=boxes_tensor,
                        multimask_output=False,
                    )
                masks_list.append(_to_numpy(masks))
                scores_list.append(_to_numpy(iou_preds))

            if points is not None and len(points) > 0:
                points_tensor = torch.from_numpy(points).float().to(self.device)
                labels_tensor = (
                    torch.from_numpy(labels).long().to(self.device)
                    if labels is not None
                    else torch.ones(len(points), dtype=torch.long, device=self.device)
                )
                with torch.no_grad():
                    masks, iou_preds, _ = self.predictor.predict(
                        point_coords=points_tensor,
                        point_labels=labels_tensor,
                        multimask_output=False,
                    )
                masks_list.append(_to_numpy(masks))
                scores_list.append(_to_numpy(iou_preds))

            if masks_list:
                all_masks = np.concatenate(masks_list, axis=0)
                all_scores = np.concatenate(scores_list, axis=0)
                all_masks = (all_masks > 0.5).astype(np.uint8)
            else:
                h, w = frame.shape[:2]
                all_masks = np.zeros((0, h, w), dtype=np.uint8)
                all_scores = np.zeros(0, dtype=np.float32)

            return SegmentationResult(
                masks=all_masks, scores=all_scores.astype(np.float32)
            )
        except Exception as e:
            raise RuntimeError(f"SAM2 prediction failed: {e}") from e
        finally:
            # Clean up after prediction to prevent memory fragmentation.
            try:
                import torch as _torch

                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
                try:
                    if hasattr(_torch, "mps") and _torch.mps.is_available():
                        _torch.mps.empty_cache()
                except Exception:
                    pass
            except Exception:
                pass

    def segment_from_detections(
        self,
        frame: np.ndarray,
        boxes: Optional[np.ndarray],
        class_ids: Optional[np.ndarray],
        class_names: Optional[list] = None,
        allowed_class_names: Optional[list] = None,
    ) -> SegmentationResult:
        """Convenience: filter detector boxes by class name and run SAM on those boxes.

        This is the recommended workflow for extracting person masks: run your
        object detector (YOLO) to find candidate boxes, then call this method
        to only pass boxes for classes you care about (e.g. `person`) into SAM.

        Args:
            frame: BGR frame
            boxes: (N,4) array of boxes from detector
            class_ids: (N,) array of integer class ids from detector
            class_names: optional list mapping class_id -> name (if available)
            allowed_class_names: list of class names to keep (default ['person'])

        Returns:
            SegmentationResult for the filtered boxes.
        """

        if boxes is None or len(boxes) == 0:
            h, w = frame.shape[:2]
            return SegmentationResult(
                masks=np.zeros((0, h, w), dtype=np.uint8), scores=np.zeros(0)
            )

        if allowed_class_names is None:
            allowed_class_names = ["person"]

        # If class names are provided, filter by them; otherwise, keep all boxes.
        keep_idx = []
        if class_ids is not None and class_names is not None:
            for i, cid in enumerate(class_ids):
                name = class_names[int(cid)] if int(cid) < len(class_names) else None
                if name in allowed_class_names:
                    keep_idx.append(i)
        else:
            # no class info — keep all
            keep_idx = list(range(len(boxes)))

        if not keep_idx:
            h, w = frame.shape[:2]
            return SegmentationResult(
                masks=np.zeros((0, h, w), dtype=np.uint8), scores=np.zeros(0)
            )

        filtered_boxes = boxes[keep_idx]
        return self.segment(frame, boxes=filtered_boxes)
