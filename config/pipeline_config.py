"""Global pipeline configuration as class-level attributes.

This module defines PipelineConfig as a singleton-like configuration class
that holds all global settings for the pipeline. Environment variables are
read once at startup and stored here, making them accessible throughout
the codebase without repeating env lookups.

Usage:
    from config.pipeline_config import PipelineConfig

    # Access settings
    confidence = PipelineConfig.YOLO_CONFIDENCE
    device = PipelineConfig.DEVICE

    # Override if needed (before pipeline init)
    PipelineConfig.YOLO_CONFIDENCE = 0.5
"""

import os
from typing import Optional, List


class PipelineConfig:
    """Global pipeline configuration class.

    All settings are class attributes initialized from environment variables
    at module load time. This allows single-point configuration access
    throughout the codebase without repeated env lookups.
    """

    # ========================================================================
    # DETECTION & SEGMENTATION CONFIGURATION
    # ========================================================================

    # YOLO confidence threshold (0.0-1.0, lower = more sensitive)
    # Override via env: YOLO_CONFIDENCE=0.5
    YOLO_CONFIDENCE: float = float(os.getenv("YOLO_CONFIDENCE", "0.4"))

    # Configure which object classes to detect and segment
    # Default: all classes (None). Override via env: ALLOWED_CLASSES=person,car,dog
    # Leave empty/unset to process all 80 COCO classes
    _ALLOWED_CLASSES_STR = os.getenv("ALLOWED_CLASSES", "")
    ALLOWED_CLASSES: Optional[List[str]] = (
        _ALLOWED_CLASSES_STR.split(",") if _ALLOWED_CLASSES_STR else None
    )

    # Enable/disable SAM2 segmentation (disabled by default for performance)
    # Override via env: USE_SEGMENTATION=true
    USE_SEGMENTATION: bool = os.getenv("USE_SEGMENTATION", "false").lower() in (
        "1",
        "true",
        "yes",
    )

    # ========================================================================
    # DEVICE CONFIGURATION
    # ========================================================================

    # Device selection: 'cpu', 'cuda', or 'mps' (auto-detected if 'auto')
    # Override via env: DEVICE=mps
    DEVICE: str = os.getenv("DEVICE", "auto")

    # ========================================================================
    # EXECUTION MODE CONFIGURATION
    # ========================================================================

    # Parallel mode toggle: set PARALLEL=true to enable per-camera workers
    PARALLEL: bool = os.getenv("PARALLEL", "false").lower() in ("1", "true", "yes")

    # Maximum frames to process (None = unlimited)
    # Override via env: MAX_FRAMES=100
    _MAX_FRAMES_STR = os.getenv("MAX_FRAMES", "")
    MAX_FRAMES: Optional[int] = int(_MAX_FRAMES_STR) if _MAX_FRAMES_STR else None

    # ========================================================================
    # OUTPUT CONFIGURATION
    # ========================================================================

    # Save rendered frames and metadata locally
    SAVE_FRAMES: bool = os.getenv("SAVE_FRAMES", "false").lower() in (
        "1",
        "true",
        "yes",
    )

    # Stream server URL for HTTP publishing (optional)
    # Override via env: STREAM_SERVER_URL=http://localhost:9000
    STREAM_SERVER_URL: Optional[str] = os.getenv("STREAM_SERVER_URL", None)

    # Stream publish token for authentication
    STREAM_PUBLISH_TOKEN: Optional[str] = os.getenv("STREAM_PUBLISH_TOKEN", None)

    # ========================================================================
    # CAMERA CONFIGURATION
    # ========================================================================

    # Path to camera configuration JSON file
    # Override via env: CAMERA_CONFIG=/path/to/cameras.json
    CAMERA_CONFIG: str = os.getenv("CAMERA_CONFIG", "config/camera_settings.json")

    # ========================================================================
    # MODEL CONFIGURATION
    # ========================================================================

    # YOLO model name
    YOLO_MODEL: str = os.getenv("YOLO_MODEL", "yolo12n.pt")

    # SAM2 model type: 'tiny' or 'small'
    SAM2_MODEL: str = os.getenv("SAM2_MODEL", "tiny")

    # ========================================================================
    # REID CONFIGURATION
    # ========================================================================

    # Only extract ReID features for person class
    # (other classes use motion-based tracking only)
    REID_PERSON_ONLY: bool = True

    # Enable ReID-based detection recovery for tracked objects
    # Recovers objects with confidence between RECOVERY_CONFIDENCE and YOLO_CONFIDENCE
    # Override via env: REID_RECOVERY=false
    REID_RECOVERY: bool = os.getenv("REID_RECOVERY", "true").lower() in (
        "1",
        "true",
        "yes",
    )

    # Lower confidence threshold for recovery search
    # Override via env: RECOVERY_CONFIDENCE=0.15
    RECOVERY_CONFIDENCE: float = float(os.getenv("RECOVERY_CONFIDENCE", "0.15"))

    # IoU threshold for spatial matching with Kalman predictions
    # Override via env: RECOVERY_IOU_THRESH=0.5
    RECOVERY_IOU_THRESH: float = float(os.getenv("RECOVERY_IOU_THRESH", "0.5"))

    # ReID similarity threshold for appearance matching
    # Override via env: RECOVERY_REID_THRESH=0.7
    RECOVERY_REID_THRESH: float = float(os.getenv("RECOVERY_REID_THRESH", "0.7"))

    # Minimum track average confidence for recovery eligibility
    # Prevents false positives from being sustained by recovery
    # Override via env: RECOVERY_MIN_TRACK_CONFIDENCE=0.28
    RECOVERY_MIN_TRACK_CONFIDENCE: float = float(
        os.getenv("RECOVERY_MIN_TRACK_CONFIDENCE", "0.28")
    )

    # Minimum track confidence history average (prevent false positives)
    # Tracks with avg confidence below this for >30 frames get deleted
    # Override via env: MIN_TRACK_CONFIDENCE=0.25
    MIN_TRACK_CONFIDENCE: float = float(os.getenv("MIN_TRACK_CONFIDENCE", "0.25"))

    # ========================================================================
    # PERFORMANCE OPTIMIZATION
    # ========================================================================

    # Run detection every Nth frame (1 = every frame, 2 = every other frame)
    # Between detections, tracker uses Kalman prediction for smooth output
    # Higher values improve FPS but may reduce tracking accuracy
    # Override via env: DETECTION_SKIP=1
    DETECTION_SKIP: int = int(os.getenv("DETECTION_SKIP", "1"))

    # ========================================================================
    # VISUALIZATION
    # ========================================================================

    # Show FPS counter on output video
    # Override via env: SHOW_FPS=true
    SHOW_FPS: bool = bool(os.getenv("SHOW_FPS", "true"))

    # ========================================================================
    # CLASS FOR: Methods
    # ========================================================================

    @classmethod
    def get_summary(cls) -> dict:
        """Return a dictionary of all configuration settings.

        Useful for debugging and logging configuration state.
        """
        return {
            "YOLO_CONFIDENCE": cls.YOLO_CONFIDENCE,
            "ALLOWED_CLASSES": cls.ALLOWED_CLASSES,
            "USE_SEGMENTATION": cls.USE_SEGMENTATION,
            "DEVICE": cls.DEVICE,
            "PARALLEL": cls.PARALLEL,
            "MAX_FRAMES": cls.MAX_FRAMES,
            "SAVE_FRAMES": cls.SAVE_FRAMES,
            "STREAM_SERVER_URL": cls.STREAM_SERVER_URL,
            "CAMERA_CONFIG": cls.CAMERA_CONFIG,
            "YOLO_MODEL": cls.YOLO_MODEL,
            "SAM2_MODEL": cls.SAM2_MODEL,
            "REID_PERSON_ONLY": cls.REID_PERSON_ONLY,
            "REID_RECOVERY": cls.REID_RECOVERY,
            "RECOVERY_CONFIDENCE": cls.RECOVERY_CONFIDENCE,
            "RECOVERY_IOU_THRESH": cls.RECOVERY_IOU_THRESH,
            "RECOVERY_REID_THRESH": cls.RECOVERY_REID_THRESH,
        }

    @classmethod
    def print_summary(cls) -> None:
        """Print configuration summary to console."""
        print("\n" + "=" * 60)
        print("PIPELINE CONFIGURATION")
        print("=" * 60)
        for key, value in cls.get_summary().items():
            print(f"  {key:25s} = {value}")
        print("=" * 60 + "\n")
