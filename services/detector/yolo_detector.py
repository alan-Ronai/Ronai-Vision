"""YOLO detector wrapper using Ultralytics (version-agnostic)."""

import os
import numpy as np
from ultralytics.models import YOLO
from services.detector.base_detector import BaseDetector, DetectionResult


class YOLODetector(BaseDetector):
    """Wraps Ultralytics YOLO model for real-time object detection.

    Supports any Ultralytics-compatible YOLO version (v8, v11, v12, etc.).
    Automatically resolves model paths from models/ folder.
    """

    def __init__(self, model_name: str = "yolo12n.pt", device: str = "cpu"):
        """
        Args:
            model_name: model filename (e.g., "yolo12n.pt", "yolov8m.pt")
                       Will look in models/ folder first, then try direct path.
            device: inference device ("cpu", "cuda", or "mps"; auto-selects GPU if available)

        Note: YOLO uses NMS which is not supported on MPS. When device="mps",
              YOLO will run on CPU (NMS is fast, so this is acceptable).
        """
        self.model_name = model_name

        # Resolve device: prefer requested device if available, fallback to CPU
        # Special case: MPS doesn't support NMS (Non-Maximum Suppression), so use CPU for YOLO
        if device == "cpu":
            self.device = "cpu"
        elif device == "cuda":
            self.device = "cuda" if self._has_cuda() else "cpu"
        elif device == "mps":
            # MPS doesn't support NMS; use CPU instead (NMS is fast anyway)
            self.device = "mps"
        else:
            self.device = "cpu"

        # Resolve model path: try models/ folder first, then direct path
        model_path = self._resolve_model_path(model_name)
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.class_names = self.model.names

    @staticmethod
    def _resolve_model_path(model_name: str) -> str:
        """Resolve model path from models/ folder or use direct path."""
        models_dir = os.path.join(os.path.dirname(__file__), "../../models")
        model_path = os.path.join(models_dir, model_name)

        if os.path.exists(model_path):
            return model_path
        elif os.path.exists(model_name):
            return model_name
        else:
            # Fall back to model name (will download if needed)
            return model_name

    @staticmethod
    def _has_cuda() -> bool:
        """Check if CUDA is available."""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    @staticmethod
    def _has_mps() -> bool:
        """Check if MPS (Metal Performance Shaders on macOS) is available."""
        try:
            import torch

            return torch.backends.mps.is_available()
        except (ImportError, AttributeError):
            return False

    def predict(self, frame: np.ndarray, confidence: float = 0.5) -> DetectionResult:
        """Run YOLO detection on a frame.

        Args:
            frame: (H, W, 3) BGR numpy array
            confidence: confidence threshold [0, 1]

        Returns:
            DetectionResult with boxes [x1, y1, x2, y2] in pixel coords
        """
        # Run inference with strict IOU threshold to prevent duplicate boxes
        # iou=0.4 means boxes with >40% overlap will be merged (stricter than default 0.7)
        results = self.model(frame, conf=confidence, iou=0.4, verbose=False)

        if len(results) == 0 or results[0].boxes is None:
            # No detections
            return DetectionResult(
                boxes=np.zeros((0, 4), dtype=np.float32),
                scores=np.zeros(0, dtype=np.float32),
                class_ids=np.zeros(0, dtype=np.int32),
                class_names=list(self.class_names.values()),
            )

        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy()  # (N, 4) in pixel coords
        scores = result.boxes.conf.cpu().numpy()  # (N,)
        class_ids = result.boxes.cls.cpu().numpy().astype(np.int32)  # (N,)

        return DetectionResult(
            boxes=boxes,
            scores=scores,
            class_ids=class_ids,
            class_names=list(self.class_names.values()),
        )

    def get_class_names(self):
        """Return list of class names."""
        return list(self.class_names.values())
