#!/bin/bash
# Pipeline Configuration Cheat Sheet
# Source this or use individual exports before running the pipeline

# ============================================================================
# DETECTION CONFIDENCE (0.0-1.0, lower = more detections)
# ============================================================================
export YOLO_CONFIDENCE=0.25          # Default: 0.25 (sensitive)
# export YOLO_CONFIDENCE=0.5         # More conservative
# export YOLO_CONFIDENCE=0.75        # Very strict

# ============================================================================
# CLASS FILTERING (comma-separated, or unset for all classes)
# ============================================================================
# export ALLOWED_CLASSES=person       # Only track people
# export ALLOWED_CLASSES=person,car   # Track people and cars
# export ALLOWED_CLASSES=car,truck    # Track vehicles only
# (unset ALLOWED_CLASSES to track all 80 COCO classes)

# ============================================================================
# DEVICE SELECTION (cpu, cuda, mps, or auto)
# ============================================================================
# export DEVICE=auto                 # Auto-detect (default)
export DEVICE=mps                     # Apple Silicon (fast)
# export DEVICE=cpu                   # CPU fallback
# export DEVICE=cuda                  # NVIDIA GPU

# ============================================================================
# EXECUTION MODE (single loop or per-camera workers)
# ============================================================================
# export PARALLEL=false              # Single loop (default, simpler)
export PARALLEL=true                  # Per-camera workers (faster for 3+ cams)

# ============================================================================
# MAX FRAMES (None = unlimited)
# ============================================================================
# export MAX_FRAMES=100              # Stop after 100 iterations
# export MAX_FRAMES=300              # Stop after 300 iterations
# (unset MAX_FRAMES for unlimited)

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================
# export SAVE_FRAMES=false           # Don't save frames locally (default)
export SAVE_FRAMES=true               # Save frames+metadata to output/multi/
# export STREAM_SERVER_URL=http://localhost:9000   # HTTP publish endpoint
# export STREAM_PUBLISH_TOKEN=secret-token         # Auth token for publishing

# ============================================================================
# CAMERA CONFIGURATION
# ============================================================================
export CAMERA_CONFIG=config/camera_settings.json     # Path to camera config
# export CAMERA_CONFIG=config/dev_cameras.json       # Alternative config

# ============================================================================
# MODEL SELECTION
# ============================================================================
# export YOLO_MODEL=yolo12n.pt       # Default (nano, fastest)
# export YOLO_MODEL=yolo12s.pt       # Small (more accurate, slower)
# export YOLO_MODEL=yolo12m.pt       # Medium (very accurate, slowest)

# export SAM2_MODEL=tiny              # Default (fastest segmentation)
# export SAM2_MODEL=small             # More accurate segmentation

# ============================================================================
# QUICK PRESETS
# ============================================================================

# PRESET: Development (fast iteration)
dev_config() {
    export YOLO_CONFIDENCE=0.25
    export DEVICE=mps
    export PARALLEL=false
    export SAVE_FRAMES=false
    unset ALLOWED_CLASSES
    echo "✓ Dev config loaded"
    python -c "from config.pipeline_config import PipelineConfig; PipelineConfig.print_summary()"
}

# PRESET: Production (high accuracy)
prod_config() {
    export YOLO_CONFIDENCE=0.5
    export DEVICE=cuda
    export PARALLEL=true
    export SAVE_FRAMES=true
    export YOLO_MODEL=yolo12m.pt
    export SAM2_MODEL=small
    unset ALLOWED_CLASSES
    echo "✓ Prod config loaded"
    python -c "from config.pipeline_config import PipelineConfig; PipelineConfig.print_summary()"
}

# PRESET: Person tracking only
person_only_config() {
    export YOLO_CONFIDENCE=0.3
    export ALLOWED_CLASSES=person
    export DEVICE=mps
    export PARALLEL=true
    echo "✓ Person-only config loaded"
    python -c "from config.pipeline_config import PipelineConfig; PipelineConfig.print_summary()"
}

# PRESET: Vehicle tracking only
vehicle_only_config() {
    export YOLO_CONFIDENCE=0.50
    export ALLOWED_CLASSES=car,truck,motorcycle,bus,van
    export DEVICE=mps
    export PARALLEL=true
    echo "✓ Vehicle-only config loaded"
    python -c "from config.pipeline_config import PipelineConfig; PipelineConfig.print_summary()"
}

# ============================================================================
# USAGE
# ============================================================================
# 
# Option 1: Source this file and use presets
#   source config_cheatsheet.sh
#   dev_config
#   python scripts/run_multi_camera.py
#
# Option 2: Set individual variables
#   export YOLO_CONFIDENCE=0.35
#   export DEVICE=mps
#   python scripts/run_multi_camera.py
#
# Option 3: Inline
#   DEVICE=mps PARALLEL=true python scripts/run_multi_camera.py
#
# ============================================================================
