"""Universal/generic ReID encoder for non-specialized classes using CLIP.

This module provides a lightweight fallback ReID encoder using CLIP for object classes
that don't have specialized models (animals, tools, equipment, etc.).

Uses OpenAI's CLIP Vision encoder to extract appearance embeddings.
"""

import os
import numpy as np
from typing import Optional
import torch
import logging
from .base_reid import BaseReID

logger = logging.getLogger(__name__)


class UniversalReID(BaseReID):
    """CLIP-based universal appearance encoder.

    Provides embeddings for arbitrary object classes using CLIP's vision encoder.
    CLIP is trained on diverse image-text pairs and works well for general visual understanding.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        model_path: Optional[str] = None,
        processor_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """Initialize universal ReID model using CLIP.

        Args:
            model_name: CLIP model name (HuggingFace hub ID)
            model_path: Path to local CLIP model file (default: models/clip-vit-base-patch32-full.pt)
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

            # Find models directory - primary location is hamal-ai/ai-service/models/
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up from services/reid/ to ai-service/
            ai_service_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
            models_dir = os.path.join(ai_service_dir, "models")

            # Resolve processor path
            if processor_path is None or not os.path.exists(processor_path):
                # Search in models directories
                search_name = processor_path or "clip-vit-base-patch32-processor"
                processor_path = None
                for candidate in [os.path.join(models_dir, search_name), search_name]:
                    if os.path.exists(candidate):
                        processor_path = candidate
                        break

            if processor_path is None or not os.path.exists(processor_path):
                raise FileNotFoundError(
                    f"CLIP processor not found. Expected at: {os.path.join(models_dir, 'clip-vit-base-patch32-processor')}"
                )

            # Load processor from local directory with use_fast=True
            self.processor = CLIPProcessor.from_pretrained(
                processor_path, use_fast=True, local_files_only=True
            )
            logger.info(f"✅ CLIP processor loaded from {processor_path}")

            # Try to load model from local file first
            if model_path is None or not os.path.exists(model_path):
                # Search in models directories
                search_name = model_path or "clip-vit-base-patch32-full.pt"
                model_path = None
                for candidate in [os.path.join(models_dir, search_name), search_name]:
                    if os.path.exists(candidate):
                        model_path = candidate
                        break

            if model_path and os.path.exists(model_path):
                # Load from local .pt file
                logger.info(f"Loading CLIP model from {model_path}")
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

                # Check if checkpoint is already a loaded model instance
                if isinstance(checkpoint, torch.nn.Module):
                    # Checkpoint is a full model object - check if it's a CLIP model
                    try:
                        from transformers.models.clip.modeling_clip import CLIPModel

                        if isinstance(checkpoint, CLIPModel):
                            # Extract vision model from full CLIP model
                            self.model = checkpoint.vision_model
                            logger.info("✅ CLIP model loaded from full model checkpoint (extracted vision_model)")
                        elif hasattr(checkpoint, 'vision_model'):
                            # Has vision_model attribute - use it
                            self.model = checkpoint.vision_model
                            logger.info("✅ CLIP model loaded (extracted vision_model from checkpoint)")
                        else:
                            # Already a vision model
                            self.model = checkpoint
                            logger.info("✅ CLIP vision model loaded from checkpoint")
                    except ImportError:
                        # Can't check type, just assume it's the right model
                        self.model = checkpoint
                        logger.info("✅ CLIP model loaded from checkpoint (assumed correct type)")

                elif isinstance(checkpoint, dict):
                    # Checkpoint is a state dict - load it the traditional way
                    logger.info("Loading CLIP from state dict")

                    # Initialize model architecture
                    self.model = CLIPVisionModel.from_pretrained(
                        model_name,
                        local_files_only=False  # Allow downloading config if needed
                    )

                    # Extract state dict
                    if "model" in checkpoint:
                        state_dict = checkpoint["model"]
                    elif "state_dict" in checkpoint:
                        state_dict = checkpoint["state_dict"]
                    else:
                        state_dict = checkpoint

                    # Load state dict
                    self.model.load_state_dict(state_dict, strict=False)
                    logger.info("✅ CLIP model loaded from state dict")
                else:
                    raise RuntimeError(f"Unexpected checkpoint type: {type(checkpoint)}")
            else:
                # Fallback to HuggingFace download
                logger.warning("Local CLIP model not found, downloading from HuggingFace...")
                self.model = CLIPVisionModel.from_pretrained(model_name)

            # Move model to device
            self.model = self.model.to(self.device)
            self.model.eval()

            # CLIP vision output dimension (ViT-B/32 uses 768)
            self.feature_dim = 768

            logger.info(f"✅ Universal ReID (CLIP) loaded on {self.device}")
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
            (N, 768) numpy array of float32 features (L2-normalized)
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

            feats = outputs.pooler_output  # (N, 768) - use pooler output for CLIP vision

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
