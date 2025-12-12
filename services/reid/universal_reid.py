"""Universal/generic ReID encoder for non-specialized classes using CLIP.

This module provides a lightweight fallback ReID encoder using CLIP for object classes
that don't have specialized models (animals, tools, equipment, etc.).

Uses OpenAI's CLIP Vision encoder to extract appearance embeddings.
"""

import os
import numpy as np
from typing import Optional
import torch
from services.reid.base_reid import BaseReID


class UniversalReID(BaseReID):
    """CLIP-based universal appearance encoder.

    Provides embeddings for arbitrary object classes using CLIP's vision encoder.
    CLIP is trained on diverse image-text pairs and works well for general visual understanding.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        processor_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """Initialize universal ReID model using CLIP.

        Args:
            model_name: CLIP model name (HuggingFace hub ID)
            processor_path: Path to processor directory (default: models/clip-vit-base-patch32-processor)
            device: inference device ("cpu", "cuda", "mps")
        """
        # Resolve device
        env_device = os.environ.get("DEVICE", "cpu")
        if device is None:
            device = env_device

        # Resolve device: prefer requested device if available, fallback to CPU
        if device == "cpu":
            self.device = "cpu"
        elif device == "cuda":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "mps":
            try:
                self.device = "mps" if torch.backends.mps.is_available() else "cpu"
            except Exception:
                self.device = "cpu"
        else:
            self.device = "cpu"

        try:
            from transformers import CLIPProcessor, CLIPVisionModel

            # Resolve processor path
            if processor_path is None:
                processor_path = os.path.join(
                    os.path.dirname(__file__),
                    "../../models/clip-vit-base-patch32-processor",
                )

            # Require local processor directory (no HuggingFace download fallback)
            if not os.path.exists(processor_path):
                raise FileNotFoundError(
                    f"CLIP processor not found at {processor_path}. "
                    f"Download it first and place in models/clip-vit-base-patch32-processor/"
                )

            # Load from local directory with use_fast=True
            self.processor = CLIPProcessor.from_pretrained(
                processor_path, use_fast=True
            )

            # Load CLIP model from HuggingFace (model is lightweight, processor is the heavy part)
            self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")

            # Move model to device
            self.model = self.model.to(self.device)
            self.model.eval()

            # CLIP vision output dimension
            self.feature_dim = 768

            print(f"✓ Universal ReID (CLIP) loaded on {self.device}")
        except ImportError as e:
            raise ImportError(
                f"CLIP is not installed. Install transformers: pip install transformers"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to initialize CLIP ReID: {e}") from e

    def extract_features(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Extract embeddings for generic object crops using CLIP.

        Args:
            frame: (H, W, 3) BGR uint8 image
            boxes: (N, 4) array of [x1, y1, x2, y2] in pixel coords

        Returns:
            (N, D) numpy array of float32 features (L2-normalized)
        """
        if boxes is None or len(boxes) == 0:
            return np.zeros((0, self.feature_dim), dtype=np.float32)

        crops = []
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                crop = np.zeros((128, 128, 3), dtype=np.uint8)
            else:
                crop = frame[y1:y2, x1:x2]
            crops.append(crop)

        try:
            from PIL import Image

            # Convert crops to RGB (CLIP expects RGB)
            pil_images = []
            for crop in crops:
                # Convert BGR to RGB
                crop_rgb = crop[:, :, ::-1] if crop.ndim == 3 else crop
                pil_images.append(Image.fromarray(crop_rgb))

            # Process images with CLIP processor
            inputs = self.processor(images=pil_images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            feats = outputs.image_embeds  # (N, 768)

            # Convert to numpy
            if isinstance(feats, torch.Tensor):
                feats = feats.detach().cpu().numpy()

            feats = feats.astype(np.float32)

            # L2-normalize
            norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-6
            feats = feats / norms.astype(np.float32)

            return feats
        except Exception as e:
            raise RuntimeError(f"CLIP feature extraction failed: {e}") from e
