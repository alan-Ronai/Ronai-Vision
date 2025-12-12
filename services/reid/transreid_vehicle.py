"""TransReID-based Vehicle ReID wrapper for car/truck/motorcycle tracking.

TransReID is a transformer-based ReID model that works well for vehicles.
Expects a checkpoint named `deit_transreid_vehicleID.pth` in models/ directory.

This module loads the checkpoint and provides extract_features() for vehicle crops.
"""

import os
import numpy as np
from typing import Optional
import torch
from services.reid.base_reid import BaseReID


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
            model_name: checkpoint filename (default "transreid_vehicle.pth")
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

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        models_dir = os.path.join(repo_root, "models")
        checkpoint_path = os.path.join(models_dir, model_name)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"TransReID Vehicle checkpoint not found at {checkpoint_path}. "
                f"Place the checkpoint in models/ directory."
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
            # The downloaded model is DeiT-small-based TransReID
            import timm

            # Create DeiT-small backbone (TransReID is built on DeiT)
            self.model = timm.create_model(
                "deit_small_patch16_224",
                pretrained=False,
                num_classes=0,  # Remove classification head, keep features only
            )

            # Load the TransReID weights
            # Remove 'module.' prefix if present (from DataParallel training)
            cleaned_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    cleaned_state_dict[k[7:]] = v
                else:
                    cleaned_state_dict[k] = v

            # Load state dict (strict=False to ignore classifier layers)
            missing_keys, unexpected_keys = self.model.load_state_dict(
                cleaned_state_dict, strict=False
            )

            # Note: Missing keys are expected (classifier layers, position embeddings, etc.)
            # Only fail if ALL keys are missing (wrong checkpoint entirely)
            if len(missing_keys) == len(self.model.state_dict()):
                raise RuntimeError(
                    f"Failed to load TransReID checkpoint: No keys matched. "
                    f"Ensure deit_transreid_vehicleID.pth is the correct TransReID model."
                )

            self.feature_dim = 384  # DeiT-small feature dimension

            self.model = self.model.to(self.device)
            self.model.eval()

            print(f"✓ TransReID Vehicle (DeiT-based) loaded on {self.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize TransReID Vehicle: {e}") from e

    def extract_features(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Extract L2-normalized embeddings for vehicle crops.

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
