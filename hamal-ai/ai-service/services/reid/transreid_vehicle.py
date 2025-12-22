"""TransReID-based Vehicle ReID wrapper for car/truck/motorcycle tracking.

TransReID is a transformer-based ReID model that works well for vehicles.
Expects a checkpoint named `deit_transreid_vehicleID.pth` in models/ directory.

This module loads the checkpoint and provides extract_features() for vehicle crops.
"""

import os
import numpy as np
from typing import Optional
import torch
import logging
from .base_reid import BaseReID

logger = logging.getLogger(__name__)


class TransReIDVehicle(BaseReID):
    """TransReID Vehicle ReID feature extractor.

    Loads a local checkpoint named `deit_transreid_vehicleID.pth` from the models/ directory.
    Optimized for vehicle appearance: color, shape, headlights, roof patterns.
    """

    def __init__(
        self,
        model_name: str = "deit_transreid_vehicleID.pth",
        device: Optional[str] = None,
    ):
        """Initialize TransReID Vehicle model.

        Args:
            model_name: checkpoint filename (default "deit_transreid_vehicleID.pth")
            device: inference device ("cpu", "cuda", "mps")

        Raises:
            FileNotFoundError: if checkpoint not found
            RuntimeError: if model initialization fails
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

        # Find models directory - primary location is hamal-ai/ai-service/models/
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up from services/reid/ to ai-service/
        ai_service_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
        models_dir = os.path.join(ai_service_dir, "models")

        # Search locations (prioritize ai-service/models)
        search_locations = [
            os.path.join(models_dir, model_name),  # Primary: ai-service/models/
            model_name,                             # Direct path if absolute
        ]

        checkpoint_path = None
        for candidate in search_locations:
            if os.path.exists(candidate):
                checkpoint_path = candidate
                break

        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"TransReID Vehicle checkpoint '{model_name}' not found. "
                f"Expected at: {os.path.join(models_dir, model_name)}"
            )

        try:
            # Load checkpoint
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )

            # Extract state_dict from checkpoint
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            # Build TransReID model using timm's DeiT architecture
            # The downloaded model is DeiT-base-based TransReID (768 dim)
            import timm

            # Create DeiT-base backbone (TransReID is built on DeiT-base)
            self.model = timm.create_model(
                "deit_base_patch16_224",  # base model (768 dim), not small (384 dim)
                pretrained=False,
                num_classes=0,  # Remove classification head, keep features only
            )

            # Load the TransReID weights
            # TransReID wraps DeiT in a 'base' module: keys are 'module.base.xxx'
            # Remove 'module.base.' prefix to match timm's DeiT architecture
            cleaned_state_dict = {}
            for k, v in state_dict.items():
                # Remove 'module.base.' prefix (from TransReID wrapper + DataParallel)
                if k.startswith("module.base."):
                    new_key = k[12:]  # Remove 'module.base.'
                    cleaned_state_dict[new_key] = v
                # Also handle plain 'module.' prefix
                elif k.startswith("module."):
                    new_key = k[7:]  # Remove 'module.'
                    cleaned_state_dict[new_key] = v
                # Handle 'base.' prefix
                elif k.startswith("base."):
                    new_key = k[5:]  # Remove 'base.'
                    cleaned_state_dict[new_key] = v
                else:
                    cleaned_state_dict[k] = v

            # Handle position embedding size mismatch
            # TransReID uses 442 tokens (196 patches + 1 CLS + camera/SIE tokens)
            # Standard DeiT uses 197 tokens (196 patches + 1 CLS)
            # We need to interpolate/trim the position embeddings
            if "pos_embed" in cleaned_state_dict:
                pretrained_pos_embed = cleaned_state_dict["pos_embed"]  # [1, 442, 768]
                model_pos_embed = self.model.pos_embed  # [1, 197, 768]

                # If sizes don't match, use only the relevant tokens
                if pretrained_pos_embed.shape[1] != model_pos_embed.shape[1]:
                    # Take CLS token + first 196 patch tokens (ignore camera/SIE tokens)
                    cleaned_state_dict["pos_embed"] = pretrained_pos_embed[:, :197, :]
                    logger.info(
                        f"Resized pos_embed from {pretrained_pos_embed.shape} to {cleaned_state_dict['pos_embed'].shape}"
                    )

            # Remove SIE embeddings and other TransReID-specific layers not in standard DeiT
            keys_to_remove = [
                k
                for k in cleaned_state_dict.keys()
                if "sie" in k.lower() or "camera" in k.lower()
            ]
            for k in keys_to_remove:
                del cleaned_state_dict[k]

            # Load state dict (strict=False to ignore classifier layers)
            missing_keys, unexpected_keys = self.model.load_state_dict(
                cleaned_state_dict, strict=False
            )

            # Check if at least some keys matched
            total_keys = len(self.model.state_dict())
            matched_keys = total_keys - len(missing_keys)
            match_ratio = matched_keys / total_keys if total_keys > 0 else 0

            if match_ratio < 0.5:  # Less than 50% keys matched
                raise RuntimeError(
                    f"Failed to load TransReID checkpoint: Only {matched_keys}/{total_keys} keys matched ({match_ratio:.1%}). "
                    f"Ensure deit_transreid_vehicleID.pth is the correct TransReID model."
                )

            logger.info(
                f"TransReID loaded: {matched_keys}/{total_keys} keys matched ({match_ratio:.1%})"
            )

            self.feature_dim = 768  # DeiT-base feature dimension

            self.model = self.model.to(self.device)
            self.model.eval()

            logger.info(f"✅ TransReID Vehicle (DeiT-based) loaded: {checkpoint_path} on {self.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize TransReID Vehicle: {e}") from e

    def extract_features(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Extract L2-normalized embeddings for vehicle crops.

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
                crop = np.zeros((128, 256, 3), dtype=np.uint8)  # Vehicle aspect ratio
            else:
                crop = frame[y1:y2, x1:x2]
            crops.append(crop)

        # Convert BGR to RGB and prepare for model
        crops_rgb = [c[:, :, ::-1] if c.ndim == 3 else c for c in crops]

        try:
            import torchvision.transforms as transforms

            transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

            from PIL import Image

            tensors = []
            for crop_rgb in crops_rgb:
                if isinstance(crop_rgb, np.ndarray):
                    img = Image.fromarray(crop_rgb)
                else:
                    img = crop_rgb
                tensor = transform(img)
                tensors.append(tensor)

            inputs = torch.stack(tensors).to(self.device)

            with torch.no_grad():
                feats = self.model(inputs)

            # Convert to numpy
            if isinstance(feats, torch.Tensor):
                feats = feats.detach().cpu().numpy()

            feats = feats.astype(np.float32)

            # L2-normalize
            norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-6
            feats = feats / norms.astype(np.float32)

            return feats
        except Exception as e:
            raise RuntimeError(f"TransReID Vehicle extraction failed: {e}") from e
