"""Detection Loop - Connects RTSP cameras to AI detection pipeline.

Flow: RTSP → FFmpeg → Frames → YOLO → BoT-SORT → ReID Recovery → Gemini → Draw BBoxes → Events

Uses BoT-SORT tracker with:
- Full 8-state Kalman filter for motion prediction
- Hungarian assignment for optimal matching
- Combined cost matrix (motion + appearance via ReID)
- ReID-based detection recovery for lost tracks
- Confidence history tracking
- Track state management (tentative → confirmed → lost)
"""

import asyncio
import logging
import time
import cv2
import numpy as np
import httpx
import base64
import os
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from queue import Queue, Empty
import threading
from collections import OrderedDict

from .detection import get_bot_sort_tracker, Detection, Track
from .reid import OSNetReID, TransReIDVehicle, UniversalReID
from . import backend_sync
from .rules import RuleEngine, RuleContext, get_rule_engine
from .recording import get_frame_buffer, get_recording_manager
from .frame_selection import get_analysis_buffer, AnalysisBuffer, get_image_enhancer, ImageEnhancer
from .scenario import get_scenario_hooks, VehicleData, PersonData
from .parallel_detection import ParallelDetectionIntegration

logger = logging.getLogger(__name__)

# Detection Recovery Configuration (configurable via env vars)
RECOVERY_CONFIDENCE = (
    0.20  # Low threshold for recovery pass (raised from 0.15 to reduce noise)
)
RECOVERY_IOU_THRESH = (
    0.3  # Minimum IoU with Kalman prediction (lowered for better recovery)
)
RECOVERY_REID_THRESH = (
    0.5  # Minimum ReID similarity (for persons) (lowered for better recovery)
)
RECOVERY_MIN_TRACK_CONFIDENCE = 0.25  # Only recover tracks with decent history
# Configurable via RECOVERY_EVERY_N_FRAMES env var (default: 3)
RECOVERY_EVERY_N_FRAMES = int(os.environ.get("RECOVERY_EVERY_N_FRAMES", "3"))
# Recovery YOLO input size - smaller = faster (default: 480, options: 320, 480, 640)
RECOVERY_YOLO_IMGSZ = int(os.environ.get("RECOVERY_YOLO_IMGSZ", "480"))


class ReIDFeatureCache:
    """LRU cache for ReID features to avoid redundant extractions.

    Caches features by track_id with TTL to handle re-appearances.
    """

    def __init__(self, max_size: int = 200, ttl_seconds: float = 30.0):
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: Dict[int, float] = {}
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._hits = 0
        self._misses = 0

    def get(self, track_id: int) -> Optional[np.ndarray]:
        """Get cached feature for track, returns None if not found or expired."""
        if track_id not in self._cache:
            self._misses += 1
            return None

        # Check TTL
        if time.time() - self._timestamps[track_id] > self._ttl:
            self._evict(track_id)
            self._misses += 1
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(track_id)
        self._hits += 1
        return self._cache[track_id]

    def put(self, track_id: int, feature: np.ndarray):
        """Cache a feature for a track."""
        if track_id in self._cache:
            self._cache.move_to_end(track_id)
        else:
            # Evict oldest if at capacity
            while len(self._cache) >= self._max_size:
                oldest = next(iter(self._cache))
                self._evict(oldest)

        self._cache[track_id] = feature
        self._timestamps[track_id] = time.time()

    def _evict(self, track_id: int):
        """Remove a track from cache."""
        if track_id in self._cache:
            del self._cache[track_id]
        if track_id in self._timestamps:
            del self._timestamps[track_id]

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{hit_rate:.1f}%",
        }

    def clear(self):
        """Clear the cache."""
        self._cache.clear()
        self._timestamps.clear()

# Class Filtering Configuration
ALLOWED_CLASSES = {"person", "car", "truck", "bus", "motorcycle", "bicycle"}
MIN_BOX_AREA = 500  # Minimum pixels squared to filter noise


@dataclass
class DetectionResult:
    """Result from detection pipeline."""

    camera_id: str
    timestamp: float
    frame: np.ndarray
    tracked_vehicles: List[Dict] = field(default_factory=list)
    tracked_persons: List[Dict] = field(default_factory=list)
    new_vehicles: List[Dict] = field(default_factory=list)
    new_persons: List[Dict] = field(default_factory=list)
    armed_persons: List[Dict] = field(default_factory=list)
    annotated_frame: Optional[np.ndarray] = None


class BBoxDrawer:
    """Draws bounding boxes and labels on frames."""

    # Colors (BGR format)
    COLORS = {
        "person": (0, 255, 0),  # Green
        "person_armed": (0, 0, 255),  # Red
        "car": (255, 165, 0),  # Orange
        "truck": (255, 100, 0),  # Dark Orange
        "motorcycle": (255, 255, 0),  # Cyan
        "bus": (128, 0, 128),  # Purple
        "bicycle": (0, 255, 255),  # Yellow
        "vehicle": (255, 165, 0),  # Orange for generic vehicle
        "predicted": (128, 128, 128),  # Gray for predicted positions
        "default": (200, 200, 200),  # Gray
    }

    @staticmethod
    def draw_detections(
        frame: np.ndarray, detections: List[Dict], detection_type: str = "object"
    ) -> np.ndarray:
        """Draw bounding boxes and labels on frame."""
        annotated = frame.copy()

        for det in detections:
            bbox = det.get("bbox", det.get("box", []))
            if not bbox or len(bbox) < 4:
                continue

            x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
            track_id = det.get("track_id", det.get("id", "?"))
            label = det.get("class", det.get("label", detection_type))
            confidence = det.get("confidence", det.get("conf", 0))
            metadata = det.get("metadata", {})

            # Determine if armed
            is_armed = False
            if metadata:
                analysis = metadata.get("analysis", metadata)
                is_armed = analysis.get("armed", False) or analysis.get("חמוש", False)

            # Check if this is a predicted (not detected) position
            is_predicted = (
                det.get("is_predicted", False) or det.get("consecutive_misses", 0) > 0
            )

            # Select color
            if is_armed:
                color = BBoxDrawer.COLORS["person_armed"]
            elif is_predicted:
                color = BBoxDrawer.COLORS["predicted"]
            else:
                color = BBoxDrawer.COLORS.get(label, BBoxDrawer.COLORS["default"])

            # Draw bounding box
            thickness = 3 if is_armed else 2
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            # Build label text
            label_parts = []
            if track_id:
                label_parts.append(f"ID:{track_id}")
            label_parts.append(label)
            if confidence > 0:
                label_parts.append(f"{confidence:.0%}")

            # Add armed warning
            if is_armed:
                weapon_type = analysis.get("weaponType", analysis.get("סוג_נשק", ""))
                label_parts.append(f"ARMED")
                # Map Hebrew weapon types to English for cv2 rendering (cv2 can't render Hebrew)
                if weapon_type and weapon_type != "לא רלוונטי":
                    weapon_type_en = {
                        "רובה": "Rifle",
                        "אקדח": "Handgun",
                        "רובה קצר": "SMG",
                        "רובה צלפים": "Sniper",
                        "סכין": "Knife",
                        "נשק קר": "Cold Weapon",
                        "לא ידוע": "Unknown",
                    }.get(weapon_type, weapon_type)
                    # Only add if it's ASCII (to avoid ?????? on screen)
                    if weapon_type_en.isascii():
                        label_parts.append(weapon_type_en)

            label_text = " | ".join(label_parts)

            # Draw label background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, font, font_scale, font_thickness
            )

            # Label position (above bbox)
            label_y = max(y1 - 10, text_height + 10)
            label_x = x1

            # Background rectangle
            cv2.rectangle(
                annotated,
                (label_x, label_y - text_height - 5),
                (label_x + text_width + 10, label_y + 5),
                color,
                -1,  # Filled
            )

            # Text
            cv2.putText(
                annotated,
                label_text,
                (label_x + 5, label_y),
                font,
                font_scale,
                (255, 255, 255),  # White text
                font_thickness,
            )

            # Draw armed indicator
            if is_armed:
                # Flashing border effect
                cv2.rectangle(
                    annotated, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 0, 255), 4
                )

        return annotated

    @staticmethod
    def draw_status_overlay(
        frame: np.ndarray,
        camera_id: str,
        vehicle_count: int,
        person_count: int,
        armed_count: int,
        fps: float = 0,
    ) -> np.ndarray:
        """Draw status overlay on frame."""
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # Status bar at top
        cv2.rectangle(annotated, (0, 0), (w, 40), (0, 0, 0), -1)

        # Camera ID
        cv2.putText(
            annotated,
            f"Camera: {camera_id}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Detection counts
        status_text = f"Vehicles: {vehicle_count} | Persons: {person_count}"
        cv2.putText(
            annotated,
            status_text,
            (w - 350, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Armed warning
        if armed_count > 0:
            cv2.putText(
                annotated,
                f"ARMED: {armed_count}",
                (w // 2 - 60, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
            # Red border
            cv2.rectangle(annotated, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)

        # Timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(
            annotated,
            timestamp,
            (w - 100, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        return annotated


@dataclass
class LoopConfig:
    """Detection loop configuration."""

    backend_url: str = field(default_factory=lambda: os.environ.get("BACKEND_URL", "http://localhost:3000"))

    # FPS Settings - Lower values = slower processing = longer video duration
    # For demos, use lower FPS to extend video playback time
    detection_fps: int = 10  # How often to run YOLO detection (affects CPU/GPU usage) - lower = longer video
    stream_fps: int = 15  # FPS for streaming annotated video to frontend
    reader_fps: int = 15  # FPS for reading from RTSP camera (affects network/decoding) - lower = slower video consumption
    recording_fps: int = 15  # FPS for saved recordings

    alert_cooldown: float = 30.0
    gemini_cooldown: float = 5.0  # Don't analyze same track more than once per 5 sec
    event_cooldown: float = 5.0  # Minimum seconds between events per camera
    draw_bboxes: bool = True
    draw_weapon_bboxes: bool = False  # Disable weapon bbox visualization by default
    send_events: bool = True
    use_bot_sort: bool = True  # Use BoT-SORT tracker with full Kalman filter
    use_reid_recovery: bool = (
        True  # Enable ReID-based detection recovery (DISABLED for now)
    )
    weapon_detector: Optional[Any] = None  # Weapon detection YOLO model

    # Confidence thresholds
    yolo_confidence: float = 0.45  # Main YOLO detection confidence
    weapon_confidence: float = 0.75  # Weapon detection confidence
    recovery_confidence: float = 0.20  # Low-confidence recovery pass


class DetectionLoop:
    """Main detection loop - connects cameras to detection pipeline.

    Uses BoT-SORT tracker with full Kalman filter:
    - 8-state Kalman filter for motion prediction
    - Hungarian assignment for optimal matching
    - Combined cost matrix (motion + appearance)
    - ReID-based detection recovery for lost tracks
    - Objects must be seen for 3 frames before reported as "new"
    - Objects must be missing for 30 frames before removed
    """

    def __init__(
        self,
        yolo_model,
        reid_tracker,
        gemini_analyzer,
        config: Optional[LoopConfig] = None,
    ):
        self.yolo = yolo_model
        self.reid_tracker = reid_tracker  # Kept for metadata storage
        self.gemini = gemini_analyzer
        self.config = config or LoopConfig()
        self.drawer = BBoxDrawer()

        # CRITICAL: Initialize Multi-Class ReID Encoders
        # - OSNet for persons (512-dim)
        # - TransReID for vehicles (768-dim)
        # - CLIP for universal fallback (768-dim)
        self.osnet = None
        self.vehicle_reid = None
        self.universal_reid = None

        # Initialize OSNet for person ReID
        try:
            osnet_model = os.environ.get("OSNET_MODEL", "osnet_x0_5_imagenet.pth")
            self.osnet = OSNetReID(model_name=osnet_model)
            logger.info(f"✅ OSNet ReID (person) initialized: 512-dim features ({osnet_model})")
        except Exception as e:
            logger.warning(f"⚠️ OSNet initialization failed: {e} - Person ReID disabled")

        # Check if we should skip heavy models (for faster startup on CPU)
        skip_heavy_models = os.environ.get("SKIP_HEAVY_REID_MODELS", "").lower() in ("1", "true", "yes")

        if skip_heavy_models:
            logger.info("⏩ Skipping TransReID and CLIP models (SKIP_HEAVY_REID_MODELS=1)")
            self.vehicle_reid = None
            self.universal_reid = None
        else:
            # Initialize TransReID for vehicle ReID
            try:
                transreid_model = os.environ.get("TRANSREID_MODEL", "deit_transreid_vehicleID.pth")
                logger.info(f"Loading TransReID for vehicle ReID ({transreid_model})...")
                self.vehicle_reid = TransReIDVehicle(model_name=transreid_model)
                logger.info(f"✅ TransReID (vehicle) initialized: 768-dim features ({transreid_model})")
            except Exception as e:
                logger.warning(
                    f"⚠️ TransReID initialization failed: {e} - Vehicle ReID disabled"
                )

            # Initialize CLIP for universal ReID (fallback for other classes)
            try:
                clip_model = os.environ.get("CLIP_MODEL", "clip-vit-base-patch32-full.pt")
                clip_processor = os.environ.get("CLIP_PROCESSOR", "clip-vit-base-patch32-processor")
                logger.info(f"Loading CLIP for universal ReID ({clip_model})...")
                self.universal_reid = UniversalReID(model_path=clip_model, processor_path=clip_processor)
                logger.info(f"✅ Universal ReID (CLIP) initialized: 768-dim features ({clip_model})")
            except Exception as e:
                logger.warning(
                    f"⚠️ Universal ReID initialization failed: {e} - Universal ReID disabled"
                )

        # CRITICAL: Per-Class Confidence Thresholds
        # Different classes can have different minimum confidence scores
        # Higher threshold = fewer false positives for that class
        self.class_confidence = {
            "person": float(os.environ.get("CONF_PERSON", "0.35")),
            "car": float(os.environ.get("CONF_CAR", "0.40")),
            "truck": float(os.environ.get("CONF_TRUCK", "0.50")),
            "bus": float(os.environ.get("CONF_BUS", "0.50")),
            "motorcycle": float(os.environ.get("CONF_MOTORCYCLE", "0.40")),
            "bicycle": float(os.environ.get("CONF_BICYCLE", "0.40")),
        }

        # Parse CLASS_CONFIDENCE env var if provided (format: "person:0.35,car:0.40,truck:0.50")
        class_conf_str = os.environ.get("CLASS_CONFIDENCE", "")
        if class_conf_str:
            for item in class_conf_str.split(","):
                if ":" in item:
                    class_name, threshold = item.split(":")
                    self.class_confidence[class_name.strip()] = float(threshold.strip())

        logger.info("=== Per-Class Confidence Thresholds ===")
        for cls, thresh in sorted(self.class_confidence.items()):
            logger.info(f"  {cls}: {thresh:.2f}")
        logger.info("========================================")

        # ReID Feature Extraction Settings
        # Extracting features for every detection is SLOW on CPU but improves accuracy
        # Set USE_REID_FEATURES=false to skip ReID and get better FPS (but more ghost tracks)
        self.use_reid_features = os.environ.get("USE_REID_FEATURES", "true").lower() == "true"
        self.max_reid_per_frame = int(os.environ.get("MAX_REID_PER_FRAME", "0"))  # 0 = unlimited

        if not self.use_reid_features:
            logger.info("⚠️ ReID feature extraction DISABLED for better FPS")
        else:
            logger.info("✅ ReID feature extraction ENABLED for all detections (best accuracy)")
            if self.max_reid_per_frame > 0:
                logger.info(f"⚠️ ReID limited to {self.max_reid_per_frame} extractions per frame")

        # Use BoT-SORT tracker for advanced tracking
        self.bot_sort = get_bot_sort_tracker() if self.config.use_bot_sort else None

        # CRITICAL: Reset tracker on startup to clear ghost tracks from previous runs
        if self.bot_sort:
            self.bot_sort.reset()
            logger.info("✅ BoT-SORT tracker reset on startup - cleared ghost tracks")

        self._running = False

        # CRITICAL: Latest-frame-only storage (eliminates 2-3 second queue delay!)
        # Instead of FIFO queue that builds up frames, we only keep the LATEST frame per camera
        # This ensures we always process the most recent frame, not stale frames from 3 seconds ago
        self._latest_frames: Dict[str, Tuple[np.ndarray, float]] = {}  # {camera_id: (frame, timestamp)}
        self._latest_frames_lock = threading.Lock()

        self._result_queue: Queue = Queue(maxsize=20)
        self._last_alert: Dict[str, float] = {}
        self._last_gemini: Dict[str, float] = {}
        self._last_event: Dict[str, float] = {}  # Rate limiting for events
        self._http_client: Optional[httpx.AsyncClient] = None
        self._process_thread: Optional[threading.Thread] = None

        # Latest annotated frames for streaming
        self._annotated_frames: Dict[str, np.ndarray] = {}
        self._frame_lock = threading.Lock()

        # Stats
        self._stats = {
            "frames_processed": 0,
            "detections": 0,
            "alerts_sent": 0,
            "events_sent": 0,
            "events_rate_limited": 0,
            "frames_dropped_stale": 0,  # Count of stale frames dropped
            "reid_extractions": 0,  # ReID feature extractions
            "reid_recoveries": 0,  # Successful ReID recoveries
            # Note: gemini_calls is tracked in GeminiAnalyzer.get_call_count()
        }

        # Timing stats (rolling averages in milliseconds)
        # Comprehensive timing breakdown for bottleneck analysis
        self._timing_stats = {
            # Main pipeline stages
            "yolo_ms": 0.0,              # YOLO inference time
            "yolo_postprocess_ms": 0.0,  # YOLO result filtering/NMS
            "reid_extract_ms": 0.0,      # ReID feature extraction (per-frame total)
            "reid_single_ms": 0.0,       # ReID single extraction average
            "tracker_ms": 0.0,           # BoT-SORT tracking
            "recovery_ms": 0.0,          # Detection recovery pass
            "weapon_ms": 0.0,            # Weapon detection
            "drawing_ms": 0.0,           # Annotation drawing
            "total_frame_ms": 0.0,       # Total frame processing time

            # Sub-component breakdowns
            "kalman_predict_ms": 0.0,    # Kalman filter prediction
            "hungarian_ms": 0.0,         # Hungarian assignment
            "cost_matrix_ms": 0.0,       # Cost matrix computation

            # Frame selection & Gemini
            "frame_quality_ms": 0.0,     # Frame quality scoring
            "image_enhance_ms": 0.0,     # Image enhancement
            "gemini_vehicle_ms": 0.0,    # Gemini vehicle analysis
            "gemini_person_ms": 0.0,     # Gemini person analysis
            "cutout_gen_ms": 0.0,        # Cutout image generation

            # I/O and sync
            "backend_sync_ms": 0.0,      # Backend HTTP sync
            "frame_copy_ms": 0.0,        # Frame memory copy
            "queue_wait_ms": 0.0,        # Time waiting for queue

            # Per-detection averages
            "reid_per_detection_ms": 0.0,  # Average ReID per detection
        }
        # Per-key sample counts for accurate averaging
        self._timing_samples = {key: 0 for key in self._timing_stats.keys()}
        self._timing_alpha = 0.1  # Exponential moving average factor

        # Additional counters for rate calculations
        self._timing_counts = {
            "reid_extractions_this_frame": 0,
            "detections_this_frame": 0,
            "tracks_updated_this_frame": 0,
        }

        # Recovery throttling
        self._recovery_counter = 0

        # ReID Feature Cache for faster lookups
        self._reid_cache = ReIDFeatureCache(max_size=200, ttl_seconds=30.0)
        logger.info(f"ReID feature cache initialized (max_size=200, ttl=30s)")

        # Stale frame threshold (configurable)
        self._stale_frame_threshold = float(os.environ.get("STALE_FRAME_THRESHOLD_MS", "300")) / 1000.0

        # Rule Engine for configurable event handling
        self.rule_engine = get_rule_engine()
        logger.info("Rule Engine initialized")
        logger.info(f"Recovery config: every {RECOVERY_EVERY_N_FRAMES} frames, YOLO imgsz={RECOVERY_YOLO_IMGSZ}")

        # Frame Selection Buffer for optimal Gemini analysis timing
        # Instead of analyzing the first frame, we buffer frames and select the best one
        self.analysis_buffer = get_analysis_buffer()
        self.image_enhancer = get_image_enhancer()
        self._use_frame_selection = os.environ.get("USE_FRAME_SELECTION", "true").lower() == "true"
        self._use_image_enhancement = os.environ.get("USE_IMAGE_ENHANCEMENT", "true").lower() == "true"
        if self._use_frame_selection:
            logger.info("✅ Frame selection ENABLED - will buffer frames for optimal Gemini analysis")
        else:
            logger.info("⚠️ Frame selection DISABLED - using first frame for analysis")
        if self._use_image_enhancement:
            logger.info("✅ Image enhancement ENABLED - will enhance frames before Gemini analysis")
        else:
            logger.info("⚠️ Image enhancement DISABLED")

        # Scenario hooks for Armed Attack demo integration
        self.scenario_hooks = get_scenario_hooks()
        logger.info("✅ Scenario hooks initialized for Armed Attack demo")

        # Parallel Detection Pipeline (DISABLED - causes track fragmentation)
        # TODO: Fix parallel detection to properly handle centralized tracking
        # When enabled, uses multiple YOLO workers with dynamic scaling
        # TEMPORARILY FORCE DISABLED until track fragmentation is fixed
        self._use_parallel_detection = False  # os.environ.get("USE_PARALLEL_DETECTION", "false").lower() == "true"
        self.parallel_integration: Optional[ParallelDetectionIntegration] = None

        if self._use_parallel_detection:
            max_workers = int(os.environ.get("PARALLEL_NUM_WORKERS", "3"))
            bootstrap_frames = int(os.environ.get("PARALLEL_BOOTSTRAP_FRAMES", "30"))
            scale_up_after = int(os.environ.get("PARALLEL_SCALE_UP_FRAMES", "50"))
            scale_interval = int(os.environ.get("PARALLEL_SCALE_INTERVAL", "30"))

            self.parallel_integration = ParallelDetectionIntegration(
                detection_loop=self,
                max_workers=max_workers,
                bootstrap_frames=bootstrap_frames,
                scale_up_after=scale_up_after,
                scale_interval=scale_interval,
            )
            logger.info(
                f"✅ Parallel detection ENABLED: max_workers={max_workers}, "
                f"bootstrap={bootstrap_frames}, scale_up_after={scale_up_after}"
            )
        else:
            logger.info("⚠️ Parallel detection DISABLED (set USE_PARALLEL_DETECTION=true to enable)")

    def on_frame(self, camera_id: str, frame: np.ndarray):
        """Callback when frame received from RTSP reader.

        CRITICAL: Stores only the LATEST frame per camera (overwrites previous).
        This prevents frame queue buildup and ensures we process fresh frames only.

        Also feeds frames to:
        - Frame buffer (for pre-recording buffer)
        - Active recordings (if any)
        """
        timestamp = time.time()

        with self._latest_frames_lock:
            self._latest_frames[camera_id] = (frame.copy(), timestamp)

        # Add to frame buffer for pre-recording support
        try:
            frame_buffer = get_frame_buffer()
            if frame_buffer:
                frame_buffer.add_frame(camera_id, frame, timestamp)
        except Exception as e:
            logger.debug(f"Frame buffer error: {e}")

        # Feed to active recordings
        try:
            recording_manager = get_recording_manager()
            if recording_manager and recording_manager.is_recording(camera_id):
                recording_manager.add_frame(camera_id, frame)
        except Exception as e:
            logger.debug(f"Recording manager error: {e}")

    def get_annotated_frame(self, camera_id: str) -> Optional[np.ndarray]:
        """Get latest annotated frame for streaming."""
        with self._frame_lock:
            return self._annotated_frames.get(camera_id)

    async def start(self):
        """Start the detection loop."""
        if self._running:
            return

        self._running = True
        self._start_time = time.time()
        self._http_client = httpx.AsyncClient(timeout=30.0)

        # Initialize parallel detection if enabled
        if self._use_parallel_detection and self.parallel_integration:
            self.parallel_integration.initialize()
            logger.info("✅ Parallel detection pipeline initialized")

        # Start processing thread
        self._process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._process_thread.start()

        # Start async result handler
        asyncio.create_task(self._handle_results())

        # Start the rule engine periodic timer for time-based rules
        await self.rule_engine.start_periodic_timer()

        logger.info("Detection loop started")

    async def stop(self):
        """Stop the detection loop."""
        self._running = False

        # Stop parallel detection if enabled
        if self._use_parallel_detection and self.parallel_integration:
            self.parallel_integration.shutdown()
            logger.info("Parallel detection pipeline shutdown")

        # Stop the rule engine periodic timer
        await self.rule_engine.stop_periodic_timer()

        if self._http_client:
            await self._http_client.aclose()

        # Close backend sync client
        await backend_sync.close_http_client()

        logger.info("Detection loop stopped")

    def _process_loop(self):
        """Process frames in background thread (blocking operations).

        CRITICAL: Uses latest-frame-only approach to minimize latency.
        Always processes the most recent frame available, skipping stale frames.

        When parallel detection is enabled, frames are submitted to the parallel
        pipeline instead of being processed synchronously.
        """
        logger.info("Detection processing thread started")

        frame_interval = 1.0 / self.config.detection_fps
        last_process_time: Dict[str, float] = {}

        while self._running:
            try:
                # Get list of cameras that have frames waiting
                with self._latest_frames_lock:
                    cameras_with_frames = list(self._latest_frames.keys())

                if not cameras_with_frames:
                    time.sleep(0.01)  # Small sleep if no frames
                    continue

                # Process each camera's latest frame
                for camera_id in cameras_with_frames:
                    # Get and remove the latest frame
                    with self._latest_frames_lock:
                        frame_data = self._latest_frames.pop(camera_id, None)

                    if frame_data is None:
                        continue

                    frame, timestamp = frame_data

                    # CRITICAL: Skip stale frames (older than threshold)
                    # This prevents processing old frames that are no longer relevant
                    frame_age = time.time() - timestamp
                    if frame_age > self._stale_frame_threshold:
                        logger.debug(
                            f"Dropped stale frame from {camera_id} (age: {frame_age:.3f}s)"
                        )
                        self._stats["frames_dropped_stale"] += 1
                        continue

                    # Rate limit per camera
                    last_time = last_process_time.get(camera_id, 0)
                    if time.time() - last_time < frame_interval:
                        # Still store frame for streaming (with minimal annotation)
                        if self.config.draw_bboxes:
                            annotated = self.drawer.draw_status_overlay(
                                frame, camera_id, 0, 0, 0
                            )
                            with self._frame_lock:
                                self._annotated_frames[camera_id] = annotated
                        continue

                    last_process_time[camera_id] = time.time()

                    # Use parallel detection if enabled, otherwise standard detection
                    if self._use_parallel_detection and self.parallel_integration:
                        # Submit to parallel pipeline (async processing)
                        # Results are handled by ParallelDetectionIntegration._emit_result()
                        self.parallel_integration.submit_frame(camera_id, frame, timestamp)
                        self._stats["frames_processed"] += 1
                        # Don't process result here - parallel integration handles it
                        continue

                    # Standard sequential detection
                    result = self._detect_frame(camera_id, frame)

                    if result:
                        self._stats["frames_processed"] += 1

                        # Store annotated frame
                        if result.annotated_frame is not None:
                            with self._frame_lock:
                                self._annotated_frames[camera_id] = result.annotated_frame

                        # Queue result for async processing
                        if (
                            result.new_vehicles
                            or result.new_persons
                            or result.armed_persons
                        ):
                            try:
                                self._result_queue.put_nowait(result)
                            except Exception:
                                pass

            except Exception as e:
                logger.error(f"Detection loop error: {e}")

    def _detect_frame(
        self, camera_id: str, frame: np.ndarray
    ) -> Optional[DetectionResult]:
        """Run detection on a single frame using BoT-SORT tracker.

        Pipeline:
        1. Run YOLO at high confidence (0.55) for primary detections
        2. Extract ReID features for persons (for appearance matching)
        3. Update BoT-SORT tracker (Kalman + Hungarian + combined cost)
        4. Run detection recovery for lost tracks (low conf YOLO + ReID matching)
        5. Draw annotations and return results
        """
        try:
            frame_start = time.time()

            # STEP 1: Run YOLO with LOW base confidence
            # We use the minimum of all class thresholds, then filter per-class below
            base_confidence = min(self.class_confidence.values()) if self.class_confidence else self.config.yolo_confidence

            yolo_start = time.time()
            yolo_results = self.yolo(
                frame,
                verbose=False,
                conf=base_confidence,  # Low base threshold - filter per-class below
                iou=0.5,  # Higher NMS threshold = more aggressive merging of overlapping boxes
                max_det=30,  # Reduced from 50 - scenes rarely have this many unique objects
                agnostic_nms=True,  # Merge overlapping boxes across classes
            )[0]
            yolo_inference_elapsed = (time.time() - yolo_start) * 1000
            self._update_timing("yolo_ms", yolo_inference_elapsed)

            # Separate detections by class (post-processing)
            postprocess_start = time.time()
            vehicle_detections = []
            person_detections = []
            vehicle_classes = ["car", "truck", "bus", "motorcycle", "bicycle"]

            for box in yolo_results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = yolo_results.names[cls]
                xyxy = box.xyxy[0].tolist()

                # FILTER 1: Skip classes not in allowed list
                if label not in ALLOWED_CLASSES:
                    continue

                # FILTER 2: Per-class confidence threshold
                # Different classes have different minimum confidence requirements
                min_conf_for_class = self.class_confidence.get(label, self.config.yolo_confidence)
                if conf < min_conf_for_class:
                    logger.debug(
                        f"Filtered {label}: confidence {conf:.2f} < {min_conf_for_class:.2f}"
                    )
                    continue

                # Convert from xyxy to xywh format
                x1, y1, x2, y2 = xyxy
                bbox = (x1, y1, x2 - x1, y2 - y1)

                # FILTER 3: Skip tiny boxes (noise)
                box_area = (x2 - x1) * (y2 - y1)
                if box_area < MIN_BOX_AREA:
                    continue

                # ReID Feature Extraction Strategy:
                # - ReID extraction is SLOW on CPU (~100ms per detection)
                # - Instead of extracting for ALL detections, we defer to post-tracking
                # - BoT-SORT will assign track IDs first, then we extract ReID only for:
                #   1. NEW tracks that need appearance features
                #   2. Tracks that haven't had ReID extracted yet
                # - This reduces extractions from ~2 per detection to ~1 per new track
                feature = None  # Will be filled in post-tracking step if needed

                # Create Detection object
                det = Detection(
                    bbox=bbox,
                    confidence=conf,
                    class_id=cls,
                    class_name=label,
                    feature=feature,
                )

                if label in vehicle_classes:
                    vehicle_detections.append(det)
                elif label == "person":
                    person_detections.append(det)

            # ADDITIONAL NMS: Merge highly overlapping detections within each class
            # YOLO's NMS may not catch all duplicates (e.g., partial body + full body detections)
            person_detections = self._merge_overlapping_detections(person_detections, iou_threshold=0.5)
            vehicle_detections = self._merge_overlapping_detections(vehicle_detections, iou_threshold=0.5)

            postprocess_elapsed = (time.time() - postprocess_start) * 1000
            self._update_timing("yolo_postprocess_ms", postprocess_elapsed)

            num_detections = len(vehicle_detections) + len(person_detections)
            self._stats["detections"] += num_detections
            self._timing_counts["detections_this_frame"] = num_detections

            # STEP 1.5: Pre-tracking ReID extraction for appearance-based matching
            # CRITICAL OPTIMIZATION: Only extract ReID every N frames (not every frame!)
            # ReID is the BIGGEST bottleneck (~2-3 seconds per frame on CPU)
            # - IoU-based matching works well for most frames
            # - ReID is only needed when tracks are lost or ambiguous
            REID_EVERY_N_FRAMES = int(os.environ.get("REID_EVERY_N_FRAMES", "3"))  # Default: every 3 frames
            self._reid_frame_counter = getattr(self, '_reid_frame_counter', 0) + 1
            should_run_reid = (self._reid_frame_counter % REID_EVERY_N_FRAMES == 0)

            reid_extract_start = time.time()
            reid_extract_count = 0

            if self.use_reid_features and self.bot_sort and should_run_reid:
                # Get all active tracks with ReID features (not just lost ones)
                active_persons = [t for t in self.bot_sort.get_active_tracks("person", camera_id)
                                 if t.feature is not None]
                active_vehicles = [t for t in self.bot_sort.get_active_tracks("vehicle", camera_id)
                                  if t.feature is not None]

                # If there are tracked objects with ReID, extract features for detections
                # This is the key fix for track fragmentation - we need features to match against
                # OPTIMIZATION: Use BATCH extraction instead of one-by-one (6x faster!)
                # OPTIMIZATION 2: Limit to 5 detections max (not 10) - most scenes don't need more
                MAX_REID_BATCH = int(os.environ.get("MAX_REID_BATCH", "5"))

                if active_persons or active_vehicles:
                    reid_pre_count = 0

                    # BATCH extract ReID for person detections (single forward pass)
                    if active_persons and person_detections:
                        # Get detections that need features (up to limit)
                        dets_needing_features = [d for d in person_detections if d.feature is None][:MAX_REID_BATCH]
                        if dets_needing_features:
                            person_features = self._extract_reid_features_batch(
                                frame, dets_needing_features, class_type="person"
                            )
                            # Assign features back to original detections
                            for i, det in enumerate(dets_needing_features):
                                if i in person_features:
                                    det.feature = person_features[i]
                                    reid_pre_count += 1
                                    self._stats["reid_extractions"] += 1

                    # BATCH extract ReID for vehicle detections (single forward pass)
                    if active_vehicles and vehicle_detections:
                        # Get detections that need features (up to limit)
                        dets_needing_features = [d for d in vehicle_detections if d.feature is None][:MAX_REID_BATCH]
                        if dets_needing_features:
                            vehicle_features = self._extract_reid_features_batch(
                                frame, dets_needing_features, class_type="vehicle"
                            )
                            # Assign features back to original detections
                            for i, det in enumerate(dets_needing_features):
                                if i in vehicle_features:
                                    det.feature = vehicle_features[i]
                                    reid_pre_count += 1
                                    self._stats["reid_extractions"] += 1

                    reid_extract_count += reid_pre_count
                    if reid_pre_count > 0:
                        logger.debug(f"Pre-tracking ReID (BATCHED): extracted {reid_pre_count} features")

            reid_extract_elapsed = (time.time() - reid_extract_start) * 1000
            self._update_timing("reid_extract_ms", reid_extract_elapsed)
            self._timing_counts["reid_extractions_this_frame"] = reid_extract_count
            if reid_extract_count > 0:
                self._update_timing("reid_per_detection_ms", reid_extract_elapsed / reid_extract_count)

            # STEP 2: Run detection recovery BEFORE tracker update (if enabled)
            # This avoids double Kalman prediction
            recovery_start = time.time()
            if self.bot_sort and self.config.use_reid_recovery:
                # Get current active tracks to identify lost tracks (filtered by camera)
                current_vehicles = self.bot_sort.get_active_tracks(
                    "vehicle", camera_id=camera_id
                )
                current_persons = self.bot_sort.get_active_tracks(
                    "person", camera_id=camera_id
                )

                # Recover vehicles using IoU matching
                recovered_vehicles = self._recover_missing_detections(
                    frame,
                    current_vehicles,
                    "vehicle",
                    [d.bbox for d in vehicle_detections],
                )
                if recovered_vehicles:
                    logger.debug(
                        f"Recovered {len(recovered_vehicles)} vehicle detections"
                    )
                    vehicle_detections.extend(recovered_vehicles)

                # Recover persons using IoU + ReID matching
                recovered_persons = self._recover_missing_detections(
                    frame,
                    current_persons,
                    "person",
                    [d.bbox for d in person_detections],
                )
                if recovered_persons:
                    logger.debug(f"Recovered {len(recovered_persons)} person detections")
                    person_detections.extend(recovered_persons)
                    self._stats["reid_recoveries"] += len(recovered_persons)

            recovery_elapsed = (time.time() - recovery_start) * 1000
            self._update_timing("recovery_ms", recovery_elapsed)

            # STEP 3: Single BoT-SORT tracker update with merged detections
            tracker_start = time.time()
            tracked_vehicles = []
            tracked_persons = []
            new_vehicles = []
            new_persons = []

            if self.bot_sort:
                # Update BoT-SORT tracker with merged detections (primary + recovered)
                # CRITICAL: Pass camera_id to associate tracks with specific camera
                all_vehicles, new_vehicle_tracks = self.bot_sort.update(
                    vehicle_detections,
                    "vehicle",
                    dt=1 / self.config.detection_fps,
                    camera_id=camera_id,
                )
                all_persons, new_person_tracks = self.bot_sort.update(
                    person_detections,
                    "person",
                    dt=1 / self.config.detection_fps,
                    camera_id=camera_id,
                )

                # CRITICAL: Filter tracks by camera to prevent cross-contamination
                # Only show tracks that were last seen on THIS camera
                # Also filter out stale tracks (not updated in last 5 frames = ~330ms at 15fps)
                # This prevents events from firing for objects that are no longer visible
                MAX_STALE_FRAMES = 5  # Tracks not updated in 5 frames are considered gone
                all_vehicles = [
                    t for t in all_vehicles
                    if t.last_seen_camera == camera_id and t.time_since_update <= MAX_STALE_FRAMES
                ]
                all_persons = [
                    t for t in all_persons
                    if t.last_seen_camera == camera_id and t.time_since_update <= MAX_STALE_FRAMES
                ]
                new_vehicle_tracks = [
                    t for t in new_vehicle_tracks if t.last_seen_camera == camera_id
                ]
                new_person_tracks = [
                    t for t in new_person_tracks if t.last_seen_camera == camera_id
                ]

                # Convert Track objects to dicts for compatibility with rest of pipeline
                tracked_vehicles = [self._track_to_dict(t) for t in all_vehicles]
                tracked_persons = [self._track_to_dict(t) for t in all_persons]

                # New tracks that should be reported
                new_vehicles = [self._track_to_dict(t) for t in new_vehicle_tracks]
                new_persons = [self._track_to_dict(t) for t in new_person_tracks]

                # STEP 2.5: Generate initial cutouts for NEW tracks
                # This ensures objects appear with images in the GID panel immediately
                # (before Gemini analysis provides the enhanced cutout)
                for track_dict in new_vehicles + new_persons:
                    try:
                        bbox = track_dict.get("bbox")
                        if bbox:
                            cutout = self._generate_enhanced_cutout(
                                frame=frame,
                                bbox=bbox,
                                class_name=track_dict.get("class", "unknown"),
                                max_size=300,  # Smaller for initial cutout
                                jpeg_quality=85,  # Slightly lower quality for speed
                                margin_percent=0.20,  # Less margin for quick cutout
                                is_pre_cropped=False,  # Frame is full-size here
                            )
                            if cutout:
                                # Initialize metadata.analysis if needed and add cutout
                                if "metadata" not in track_dict or track_dict["metadata"] is None:
                                    track_dict["metadata"] = {}
                                if "analysis" not in track_dict["metadata"]:
                                    track_dict["metadata"]["analysis"] = {}
                                track_dict["metadata"]["analysis"]["cutout_image"] = cutout
                                logger.debug(f"Added initial cutout for new track {track_dict.get('track_id')}")
                    except Exception as e:
                        logger.debug(f"Failed to generate initial cutout: {e}")
            else:
                # No tracker - just use raw detections (not recommended)
                tracked_vehicles = [
                    self._detection_to_dict(d, f"v_{i}")
                    for i, d in enumerate(vehicle_detections)
                ]
                tracked_persons = [
                    self._detection_to_dict(d, f"p_{i}")
                    for i, d in enumerate(person_detections)
                ]
                new_vehicles = tracked_vehicles.copy()
                new_persons = tracked_persons.copy()

            tracker_elapsed = (time.time() - tracker_start) * 1000
            self._update_timing("tracker_ms", tracker_elapsed)

            # STEP 3.5: Post-tracking ReID extraction (only for NEW tracks)
            # OPTIMIZATION: Only extract for new tracks every N frames (sync with pre-tracking)
            # New tracks need features, but not urgently - can wait a few frames
            reid_post_start = time.time()
            if self.use_reid_features and self.bot_sort and should_run_reid:
                reid_count = 0
                # OPTIMIZATION: Reduced limit - 3 tracks per type is usually enough
                max_reid = self.max_reid_per_frame if self.max_reid_per_frame > 0 else 3  # Reduced from 10

                # Separate new tracks by type for batch processing
                new_person_tracks_needing_features = [
                    t for t in new_person_tracks if t.feature is None
                ][:max_reid]
                new_vehicle_tracks_needing_features = [
                    t for t in new_vehicle_tracks if t.feature is None
                ][:max_reid]

                # BATCH extract for new person tracks (single forward pass)
                if new_person_tracks_needing_features:
                    person_features = self._extract_reid_features_batch(
                        frame, new_person_tracks_needing_features, class_type="person"
                    )
                    for i, track in enumerate(new_person_tracks_needing_features):
                        if i in person_features:
                            track.feature = person_features[i]
                            self._stats["reid_extractions"] += 1
                            reid_count += 1

                # BATCH extract for new vehicle tracks (single forward pass)
                if new_vehicle_tracks_needing_features:
                    vehicle_features = self._extract_reid_features_batch(
                        frame, new_vehicle_tracks_needing_features, class_type="vehicle"
                    )
                    for i, track in enumerate(new_vehicle_tracks_needing_features):
                        if i in vehicle_features:
                            track.feature = vehicle_features[i]
                            self._stats["reid_extractions"] += 1
                            reid_count += 1

                if reid_count > 0:
                    logger.debug(f"Post-tracking ReID (BATCHED): extracted {reid_count} features for new tracks")

            reid_post_elapsed = (time.time() - reid_post_start) * 1000
            # Add to ReID timing (this is more accurate as it's the actual ReID work)
            self._update_timing("reid_extract_ms", reid_post_elapsed)

            # STEP 4: Weapon Detection - detect firearms and mark nearby persons as armed
            # OPTIMIZATIONS:
            # 1. Skip entirely if no persons detected (weapons only matter with people)
            # 2. Run every N frames to reduce overhead (default: every 2 frames)
            # 3. Cache weapon results for association on skipped frames
            weapon_start = time.time()
            detected_weapons = []
            armed_persons_this_frame = []  # Track armed persons for summary log

            # Config for weapon detection frequency (can be tuned via env var)
            WEAPON_EVERY_N_FRAMES = int(os.environ.get("WEAPON_EVERY_N_FRAMES", "2"))

            if self.config.weapon_detector is not None and tracked_persons:
                # OPTIMIZATION 1: Only run weapon detection if persons are present
                # OPTIMIZATION 2: Run every N frames to reduce overhead
                self._weapon_frame_counter = getattr(self, '_weapon_frame_counter', 0) + 1

                should_run_weapon_detection = (self._weapon_frame_counter % WEAPON_EVERY_N_FRAMES == 0)

                if should_run_weapon_detection:
                    try:
                        weapon_results = self.config.weapon_detector(
                            frame,
                            verbose=False,
                            conf=self.config.weapon_confidence,  # Configurable weapon confidence
                            iou=0.4,
                        )[0]

                        # Update weapon cache with new detections
                        self._cached_weapons = []
                        for box in weapon_results.boxes:
                            weapon_bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                            weapon_conf = float(box.conf[0])
                            weapon_class = weapon_results.names[int(box.cls[0])]
                            self._cached_weapons.append({
                                "bbox": weapon_bbox,
                                "confidence": weapon_conf,
                                "class": weapon_class,
                            })
                    except Exception as e:
                        logger.error(f"Weapon detection error: {e}")

                # Use cached weapons for association (whether we just ran detection or not)
                detected_weapons = getattr(self, '_cached_weapons', [])

                # Associate weapons with persons
                for weapon in detected_weapons:
                    weapon_bbox = weapon["bbox"]
                    weapon_conf = weapon["confidence"]
                    weapon_class = weapon["class"]

                    # Find persons near this weapon
                    for person in tracked_persons:
                        person_bbox = person.get("bbox", [])
                        if len(person_bbox) >= 4:
                            # Check if weapon is near person (using IoU or proximity)
                            if self._is_weapon_near_person(weapon_bbox, person_bbox):
                                # Mark person as armed
                                person_meta = person.get("metadata", {})
                                if not isinstance(person_meta, dict):
                                    person_meta = {}

                                person_meta["armed"] = True
                                person_meta["חמוש"] = True
                                person_meta["weaponType"] = weapon_class
                                person_meta["סוג_נשק"] = weapon_class
                                person_meta["weapon_confidence"] = weapon_conf
                                person_meta["detection_method"] = "yolo_weapon_detector"

                                person["metadata"] = person_meta

                                # Update tracker metadata if using BoT-SORT
                                if self.bot_sort:
                                    track_id = person.get("track_id")
                                    # Filter by camera when updating metadata
                                    for track in self.bot_sort.get_active_tracks(
                                        "person", camera_id=camera_id
                                    ):
                                        if track.track_id == track_id:
                                            track.metadata.update(person_meta)
                                            break

                                # Add to armed persons list for summary log
                                armed_persons_this_frame.append({
                                    "track_id": person.get('track_id'),
                                    "weapon": weapon_class,
                                    "conf": weapon_conf
                                })

                # Note: Detailed logging is now handled by the Rule Engine via log_event action
                if detected_weapons:
                    logger.debug(f"Weapon detection: {len(detected_weapons)} weapon(s), {len(armed_persons_this_frame)} armed person(s)")

            weapon_elapsed = (time.time() - weapon_start) * 1000
            self._update_timing("weapon_ms", weapon_elapsed)

            # Check for armed persons
            armed_persons = []
            for p in tracked_persons:
                meta = p.get("metadata", {})
                if meta:
                    if meta.get("armed") or meta.get("חמוש"):
                        armed_persons.append(p)

            # Draw annotations
            draw_start = time.time()
            annotated_frame = None
            if self.config.draw_bboxes:
                annotated_frame = frame.copy()

                # Draw weapons (highlight in red) - configurable, disabled by default
                if detected_weapons and self.config.draw_weapon_bboxes:
                    for weapon in detected_weapons:
                        w_bbox = weapon["bbox"]
                        x1, y1, x2, y2 = [int(v) for v in w_bbox]
                        # Draw weapon box in bright red
                        cv2.rectangle(
                            annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3
                        )
                        # Label
                        label = f"WEAPON {weapon['class']} {weapon['confidence']:.0%}"
                        cv2.putText(
                            annotated_frame,
                            label,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2,
                        )

                # Draw vehicles
                annotated_frame = self.drawer.draw_detections(
                    annotated_frame, tracked_vehicles, "vehicle"
                )

                # Draw persons
                annotated_frame = self.drawer.draw_detections(
                    annotated_frame, tracked_persons, "person"
                )

                # Draw overlay
                annotated_frame = self.drawer.draw_status_overlay(
                    annotated_frame,
                    camera_id,
                    len(tracked_vehicles),
                    len(tracked_persons),
                    len(armed_persons),
                )

            draw_elapsed = (time.time() - draw_start) * 1000
            self._update_timing("drawing_ms", draw_elapsed)

            # Total frame time
            total_elapsed = (time.time() - frame_start) * 1000
            self._update_timing("total_frame_ms", total_elapsed)
            # Note: _timing_samples is now per-key, updated in _update_timing()

            return DetectionResult(
                camera_id=camera_id,
                timestamp=time.time(),
                frame=frame,
                tracked_vehicles=tracked_vehicles,
                tracked_persons=tracked_persons,
                new_vehicles=new_vehicles,
                new_persons=new_persons,
                armed_persons=armed_persons,
                annotated_frame=annotated_frame,
            )

        except Exception as e:
            logger.error(f"Detection error for {camera_id}: {e}")
            return None

    def _extract_reid_feature(
        self, frame: np.ndarray, xyxy: List[float], class_name: str
    ) -> Optional[np.ndarray]:
        """Extract ReID feature vector using appropriate encoder for the class.

        Args:
            frame: Full frame image (BGR format)
            xyxy: Bounding box in [x1, y1, x2, y2] format
            class_name: Object class (e.g., "person", "car", "truck")

        Returns:
            ReID feature vector (L2-normalized) or None if extraction fails
            - Person: 512-dim (OSNet)
            - Vehicle: 768-dim (TransReID)
            - Other: 768-dim (CLIP)
        """
        try:
            # Select appropriate encoder based on class
            vehicle_classes = {"car", "truck", "bus", "motorcycle", "bicycle", "van"}

            if class_name == "person":
                encoder = self.osnet
                encoder_name = "OSNet"
            elif class_name in vehicle_classes:
                encoder = self.vehicle_reid
                encoder_name = "TransReID"
            else:
                encoder = self.universal_reid
                encoder_name = "CLIP"

            # Check if encoder is available
            if encoder is None:
                logger.debug(f"{encoder_name} encoder not available for {class_name}")
                return None

            x1, y1, x2, y2 = [int(v) for v in xyxy]
            h, w = frame.shape[:2]

            # Clamp to frame bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                return None

            # All encoders expect boxes in [x1, y1, x2, y2] format as numpy array
            boxes = np.array([[x1, y1, x2, y2]])

            # Extract features using appropriate encoder (returns L2-normalized features)
            features = encoder.extract_features(frame, boxes)

            if features is not None and len(features) > 0:
                feature = features[0]  # Get first (only) feature vector

                # Log successful extraction
                logger.debug(
                    f"✅ Extracted {encoder_name} ReID feature for {class_name}: "
                    f"shape={feature.shape}, norm={np.linalg.norm(feature):.3f}"
                )

                return feature

            logger.debug(
                f"❌ {encoder_name} feature extraction returned None for {class_name}"
            )
            return None
        except Exception as e:
            logger.debug(f"ReID feature extraction error for {class_name}: {e}")
            return None

    def _extract_reid_feature_cached(
        self, frame: np.ndarray, xyxy: List[float], class_name: str, track_id: int = None
    ) -> Optional[np.ndarray]:
        """Extract ReID feature with caching support.

        If track_id is provided and a cached feature exists, returns the cached version.
        Otherwise extracts a new feature and caches it.

        Args:
            frame: Full frame image (BGR format)
            xyxy: Bounding box in [x1, y1, x2, y2] format
            class_name: Object class (e.g., "person", "car", "truck")
            track_id: Optional track ID for cache lookup

        Returns:
            ReID feature vector (L2-normalized) or None if extraction fails
        """
        # Try cache first if track_id provided
        if track_id is not None:
            cached = self._reid_cache.get(track_id)
            if cached is not None:
                return cached

        # Extract new feature
        feature = self._extract_reid_feature(frame, xyxy, class_name)

        # Cache if successful and track_id provided
        if feature is not None and track_id is not None:
            self._reid_cache.put(track_id, feature)

        return feature

    def _extract_reid_features_batch(
        self,
        frame: np.ndarray,
        detections: List[Any],
        class_type: str = "person",
    ) -> Dict[int, np.ndarray]:
        """Extract ReID features for multiple detections in a single batch.

        This is MUCH faster than extracting one-by-one because:
        1. GPU processes all crops in parallel
        2. Reduced Python overhead (one model call vs N calls)
        3. Better memory throughput

        Args:
            frame: Full frame image (BGR format)
            detections: List of Detection objects with .bbox and optionally .feature
            class_type: "person" or "vehicle" - determines which encoder to use

        Returns:
            Dictionary mapping detection index to feature vector
        """
        if not detections:
            return {}

        # Select encoder based on class type
        if class_type == "person":
            encoder = self.osnet
            encoder_name = "OSNet"
        else:  # vehicle
            encoder = self.vehicle_reid
            encoder_name = "TransReID"

        if encoder is None:
            logger.debug(f"{encoder_name} encoder not available for batch extraction")
            return {}

        try:
            h, w = frame.shape[:2]

            # Collect valid bboxes and their indices
            valid_indices = []
            valid_boxes = []

            for i, det in enumerate(detections):
                # Skip if already has a feature
                if det.feature is not None:
                    continue

                # Get bbox in xyxy format
                if hasattr(det, 'bbox'):
                    x, y, bw, bh = det.bbox  # xywh format
                    x1, y1, x2, y2 = x, y, x + bw, y + bh
                else:
                    continue

                # Clamp to frame bounds
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(w, int(x2)), min(h, int(y2))

                if x2 <= x1 or y2 <= y1:
                    continue

                valid_indices.append(i)
                valid_boxes.append([x1, y1, x2, y2])

            if not valid_boxes:
                return {}

            # Convert to numpy array for batch processing
            boxes_array = np.array(valid_boxes)

            # Extract ALL features in ONE forward pass (the key optimization!)
            features = encoder.extract_features(frame, boxes_array)

            if features is None or len(features) == 0:
                return {}

            # Map features back to detection indices
            result = {}
            for idx, det_idx in enumerate(valid_indices):
                if idx < len(features):
                    result[det_idx] = features[idx]

            logger.debug(
                f"✅ Batch {encoder_name} extracted {len(result)} features "
                f"for {class_type}s in ONE forward pass"
            )

            return result

        except Exception as e:
            logger.warning(f"Batch ReID extraction failed for {class_type}: {e}")
            return {}

    def _track_to_dict(self, track: Track) -> dict:
        """Convert Track object to dictionary for compatibility with rest of pipeline.

        Args:
            track: Track object from BoT-SORT

        Returns:
            Dictionary representation of track
        """
        # Convert bbox from (x, y, w, h) to [x1, y1, x2, y2] format for drawing
        x, y, w, h = track.bbox
        bbox_xyxy = [x, y, x + w, y + h]

        return {
            "track_id": track.track_id,
            "bbox": bbox_xyxy,
            "confidence": track.confidence,
            "class": track.class_name,
            "metadata": track.metadata,
            "consecutive_misses": track.time_since_update,
            "is_predicted": track.time_since_update > 0,
            "avg_confidence": track.avg_confidence,
            "state": track.state,
        }

    def _detection_to_dict(self, detection: Detection, track_id: str) -> dict:
        """Convert Detection object to dictionary.

        Args:
            detection: Detection object
            track_id: Assigned track ID

        Returns:
            Dictionary representation
        """
        # Convert bbox from (x, y, w, h) to [x1, y1, x2, y2] format
        x, y, w, h = detection.bbox
        bbox_xyxy = [x, y, x + w, y + h]

        return {
            "track_id": track_id,
            "bbox": bbox_xyxy,
            "confidence": detection.confidence,
            "class": detection.class_name,
            "metadata": {},
            "consecutive_misses": 0,
            "is_predicted": False,
        }

    def _generate_enhanced_cutout(
        self,
        frame: np.ndarray,
        bbox: tuple,
        class_name: str = "unknown",
        max_size: int = 400,
        jpeg_quality: int = 90,
        margin_percent: float = 0.25,
        is_pre_cropped: bool = False,
    ) -> Optional[str]:
        """Generate an enhanced cutout image for frontend display.

        This creates the image that appears in GlobalIDStore UI.
        It's the ENHANCED optimal frame, not the first detection frame.

        IMPORTANT: The frame may already be a crop from AnalysisBuffer.
        We detect this by comparing frame dimensions to bbox dimensions.
        If frame is already a crop, use the entire frame as cutout.

        Args:
            frame: Enhanced frame (BGR numpy array) - may be full frame or crop
            bbox: Bounding box (x1, y1, x2, y2) - always in original frame coordinates
            class_name: Object class for logging
            max_size: Maximum dimension for output
            jpeg_quality: JPEG quality (1-100)
            margin_percent: Margin around bbox as fraction

        Returns:
            Base64 encoded JPEG string, or None on error
        """
        try:
            if frame is None or frame.size == 0:
                return None

            h, w = frame.shape[:2]
            x1, y1, x2, y2 = [int(v) for v in bbox]

            # Calculate expected bbox dimensions
            bbox_w = x2 - x1
            bbox_h = y2 - y1

            if bbox_w <= 0 or bbox_h <= 0:
                return None

            # Check if frame is already a crop from AnalysisBuffer
            # Use the explicit flag if provided, otherwise try to detect
            if is_pre_cropped:
                is_already_cropped = True
            else:
                # Fallback detection: If frame dimensions are close to bbox dimensions (with margin)
                # AnalysisBuffer uses 25% margin, so expected crop size is ~1.5x bbox
                expected_crop_w = int(bbox_w * 1.5)
                expected_crop_h = int(bbox_h * 1.5)

                # Check if bbox coordinates are outside frame bounds - clear sign of pre-cropped frame
                bbox_outside_frame = (x1 >= w or y1 >= h or x2 > w * 1.5 or y2 > h * 1.5)

                # If frame is small and close to expected crop size, it's already cropped
                # OR if the bbox coordinates don't make sense for this frame size
                is_already_cropped = (
                    bbox_outside_frame or  # Bbox coords don't fit frame = definitely pre-cropped
                    (w < bbox_w or h < bbox_h) or  # Frame smaller than bbox = pre-cropped
                    (
                        w <= expected_crop_w * 1.5 and  # Allow more tolerance
                        h <= expected_crop_h * 1.5 and
                        w < 600 and h < 600  # Reduced threshold - crops are usually smaller
                    )
                )

            if is_already_cropped:
                # Frame is already a crop - use entire frame as cutout
                logger.debug(f"Frame already cropped for {class_name} ({w}x{h}), using entire frame")
                crop = frame.copy()
            else:
                # Frame is full-size - need to crop
                # Add margin around the bbox
                margin_x = int(bbox_w * margin_percent)
                margin_y = int(bbox_h * margin_percent)

                # Expand bbox with margin, clamped to frame bounds
                x1 = max(0, x1 - margin_x)
                y1 = max(0, y1 - margin_y)
                x2 = min(w, x2 + margin_x)
                y2 = min(h, y2 + margin_y)

                if x2 <= x1 or y2 <= y1:
                    return None

                # Crop the region with margin
                crop = frame[y1:y2, x1:x2]

            # Ensure crop has valid data
            if crop is None or crop.size == 0:
                logger.warning(f"Empty crop for {class_name}")
                return None

            # Handle different color formats - ensure BGR for proper JPEG encoding
            if len(crop.shape) == 2:
                # Grayscale - convert to BGR
                crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
            elif crop.shape[2] == 4:
                # RGBA/BGRA - convert to BGR
                crop = cv2.cvtColor(crop, cv2.COLOR_BGRA2BGR)
            # 3 channel is assumed BGR (standard OpenCV format) - no conversion needed

            # Resize if too large (keep aspect ratio)
            crop_h, crop_w = crop.shape[:2]
            if crop_w > max_size or crop_h > max_size:
                scale = max_size / max(crop_w, crop_h)
                new_w, new_h = int(crop_w * scale), int(crop_h * scale)
                crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Encode to JPEG with good quality
            success, buffer = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            if not success:
                logger.warning(f"Failed to encode JPEG for {class_name}")
                return None

            base64_str = base64.b64encode(buffer).decode('utf-8')

            logger.debug(f"Generated enhanced cutout for {class_name}: {len(base64_str)} bytes (from {'crop' if is_already_cropped else 'full'})")
            return base64_str

        except Exception as e:
            logger.warning(f"Failed to generate enhanced cutout: {e}")
            return None

    def _merge_overlapping_detections(
        self,
        detections: List[Detection],
        iou_threshold: float = 0.5
    ) -> List[Detection]:
        """Merge overlapping detections within the same class.

        This is an additional NMS step after YOLO's built-in NMS.
        It catches cases where YOLO detects the same object multiple times
        (e.g., partial body + full body, or reflections).

        Args:
            detections: List of Detection objects
            iou_threshold: IoU threshold for merging

        Returns:
            Filtered list with overlapping detections merged
        """
        if len(detections) <= 1:
            return detections

        # Sort by confidence (highest first)
        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        kept = []

        for det in sorted_dets:
            should_keep = True

            for kept_det in kept:
                iou = self._compute_iou_xywh(det.bbox, kept_det.bbox)

                if iou >= iou_threshold:
                    # Overlapping - merge into the higher confidence one (already kept)
                    should_keep = False
                    break

            if should_keep:
                kept.append(det)

        if len(kept) < len(detections):
            logger.debug(f"Post-NMS merged {len(detections)} -> {len(kept)} detections")

        return kept

    @staticmethod
    def _compute_iou_xywh(bbox1: Tuple, bbox2: Tuple) -> float:
        """Compute IoU between two bboxes in (x, y, w, h) format."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Convert to xyxy
        x1_min, y1_min = x1, y1
        x1_max, y1_max = x1 + w1, y1 + h1
        x2_min, y2_min = x2, y2
        x2_max, y2_max = x2 + w2, y2 + h2

        # Intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

        # Union
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area

        if union_area <= 0:
            return 0.0

        return inter_area / union_area

    def _recover_missing_detections(
        self,
        frame: np.ndarray,
        active_tracks: List[Track],
        object_type: str,
        primary_bboxes: List[Tuple[float, float, float, float]] = None,
    ) -> List[Detection]:
        """Recover missed detections for lost tracks using ReID matching.

        This is the CRITICAL FEATURE for robust tracking. When a track loses detection
        due to momentary occlusion or YOLO failure, we:
        1. Run a low-threshold YOLO pass (conf=0.20)
        2. Match candidates using ReID similarity (for persons) or IoU (for vehicles)
        3. Only accept matches that align with Kalman predictions

        Args:
            frame: Current frame
            active_tracks: All active tracks for this object type
            object_type: 'person' or 'vehicle'
            primary_bboxes: List of primary detection bboxes to avoid duplicates

        Returns:
            List of recovered Detection objects
        """
        try:
            # OPTIMIZATION 1: Quick check - skip if no active tracks at all
            if not active_tracks:
                return []

            # OPTIMIZATION 2: Throttle recovery to run every N frames for performance
            self._recovery_counter += 1
            if self._recovery_counter % RECOVERY_EVERY_N_FRAMES != 0:
                return []

            # STEP 1: Identify tracks that need recovery (OPTIMIZED - early exit)
            # Use generator with any() for early termination check
            has_lost_tracks = any(
                t.time_since_update > 0
                and t.avg_confidence > RECOVERY_MIN_TRACK_CONFIDENCE
                and t.state == "confirmed"
                for t in active_tracks
            )

            if not has_lost_tracks:
                return []

            # Now build the full list (we know there's at least one)
            lost_tracks = [
                t
                for t in active_tracks
                if t.time_since_update > 0
                and t.avg_confidence > RECOVERY_MIN_TRACK_CONFIDENCE
                and t.state == "confirmed"
            ]

            logger.debug(
                f"Recovery for {object_type}: {len(lost_tracks)} lost tracks out of {len(active_tracks)} active"
            )

            # STEP 2: Run low-threshold YOLO detection with smaller input size
            # OPTIMIZATION 3: Use smaller imgsz for faster inference
            low_conf_results = self.yolo(
                frame,
                verbose=False,
                conf=self.config.recovery_confidence,  # Configurable recovery confidence
                iou=0.4,  # Stricter NMS
                max_det=20,  # Reduced from 30 - we only need a few candidates
                agnostic_nms=True,  # Merge across classes
                imgsz=RECOVERY_YOLO_IMGSZ,  # Smaller input = faster inference
            )[0]

            recovery_candidates = []
            target_classes = (
                ["person"]
                if object_type == "person"
                else ["car", "truck", "bus", "motorcycle", "bicycle"]
            )

            # Limit ReID extractions in recovery to avoid performance impact
            MAX_RECOVERY_REID = 3  # Maximum ReID extractions per recovery pass
            reid_extracted = 0

            for box in low_conf_results.boxes:
                cls = int(box.cls[0])
                label = low_conf_results.names[cls]

                if label not in target_classes:
                    continue

                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()

                # Convert to xywh
                x1, y1, x2, y2 = xyxy
                bbox = (x1, y1, x2 - x1, y2 - y1)

                # Extract ReID feature only for first few candidates (expensive operation)
                feature = None
                if reid_extracted < MAX_RECOVERY_REID:
                    try:
                        feature = self._extract_reid_feature_cached(frame, xyxy, label)
                        if feature is not None:
                            reid_extracted += 1
                    except Exception as e:
                        logger.debug(f"Recovery ReID extraction failed for {label}: {e}")

                recovery_candidates.append(
                    {
                        "bbox": bbox,
                        "confidence": conf,
                        "class_id": cls,
                        "class_name": label,
                        "feature": feature,
                    }
                )

            if not recovery_candidates:
                return []

            # STEP 3: Filter out candidates that overlap with primary detections or active tracks
            # (to prevent duplicates)
            all_existing_bboxes = []

            # Add primary detection bboxes
            if primary_bboxes:
                all_existing_bboxes.extend(primary_bboxes)

            # Add recently detected track bboxes
            all_existing_bboxes.extend(
                [
                    t.bbox
                    for t in active_tracks
                    if t.time_since_update == 0  # Recently detected
                ]
            )

            filtered_candidates = []
            for candidate in recovery_candidates:
                # Check if this candidate overlaps significantly with any existing detection
                is_duplicate = False
                for existing_bbox in all_existing_bboxes:
                    if self._calculate_iou(candidate["bbox"], existing_bbox) > 0.5:
                        is_duplicate = True
                        break

                if not is_duplicate:
                    filtered_candidates.append(candidate)

            logger.debug(
                f"Recovery filtering: {len(recovery_candidates)} candidates → "
                f"{len(filtered_candidates)} after deduplication"
            )
            recovery_candidates = filtered_candidates

            if not recovery_candidates:
                return []

            # STEP 4: Match candidates to lost tracks
            recovered = []
            matched_candidate_indices = set()

            for track in lost_tracks:
                # Get Kalman predicted position
                pred_bbox = track.bbox  # Already predicted in track.predict()

                best_match_idx = None
                best_score = 0.0

                for i, candidate in enumerate(recovery_candidates):
                    if i in matched_candidate_indices:
                        continue

                    cand_bbox = candidate["bbox"]

                    # Check IoU with Kalman prediction
                    iou = self._calculate_iou(pred_bbox, cand_bbox)

                    if iou < RECOVERY_IOU_THRESH:
                        continue

                    # For persons, use ReID similarity if available
                    if (
                        object_type == "person"
                        and track.feature is not None
                        and candidate["feature"] is not None
                    ):
                        similarity = np.dot(track.feature, candidate["feature"])

                        if similarity < RECOVERY_REID_THRESH:
                            continue

                        # Combined score (IoU + ReID)
                        score = 0.3 * iou + 0.7 * similarity
                    else:
                        # For vehicles, use only IoU
                        score = iou

                    if score > best_score:
                        best_score = score
                        best_match_idx = i

                # If good match found, create recovered detection
                if best_match_idx is not None:
                    candidate = recovery_candidates[best_match_idx]
                    matched_candidate_indices.add(best_match_idx)

                    recovered.append(
                        Detection(
                            bbox=candidate["bbox"],
                            confidence=candidate["confidence"],
                            class_id=candidate["class_id"],
                            class_name=candidate["class_name"],
                            feature=candidate["feature"],
                        )
                    )

                    logger.debug(
                        f"Recovered {object_type} track {track.track_id} "
                        f"with conf={candidate['confidence']:.2f}, score={best_score:.2f}"
                    )

            if recovered:
                logger.debug(f"Recovered {len(recovered)} {object_type} detections")

            return recovered

        except Exception as e:
            logger.error(f"Detection recovery error: {e}")
            return []

    @staticmethod
    def _calculate_iou(
        bbox1: Tuple[float, float, float, float],
        bbox2: Tuple[float, float, float, float],
    ) -> float:
        """Calculate IoU between two bboxes in (x, y, w, h) format."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Convert to (x1, y1, x2, y2)
        box1 = [x1, y1, x1 + w1, y1 + h1]
        box2 = [x2, y2, x2 + w2, y2 + h2]

        # Intersection
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)

        # Union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _is_weapon_near_person(
        weapon_bbox: List[float],
        person_bbox: List[float],
        proximity_threshold: float = 100,
    ) -> bool:
        """Check if weapon is near a person.

        Args:
            weapon_bbox: Weapon bounding box in [x1, y1, x2, y2] format
            person_bbox: Person bounding box in [x1, y1, x2, y2] format
            proximity_threshold: Maximum distance in pixels for weapon to be considered "near"

        Returns:
            True if weapon is inside person bbox or within proximity threshold
        """
        # Get weapon center
        w_x1, w_y1, w_x2, w_y2 = weapon_bbox
        weapon_center_x = (w_x1 + w_x2) / 2
        weapon_center_y = (w_y1 + w_y2) / 2

        # Get person bbox
        p_x1, p_y1, p_x2, p_y2 = person_bbox

        # Check if weapon center is inside person bbox
        if p_x1 <= weapon_center_x <= p_x2 and p_y1 <= weapon_center_y <= p_y2:
            return True

        # Check proximity - find closest point on person bbox to weapon center
        closest_x = max(p_x1, min(weapon_center_x, p_x2))
        closest_y = max(p_y1, min(weapon_center_y, p_y2))

        # Calculate distance
        distance = np.sqrt(
            (weapon_center_x - closest_x) ** 2 + (weapon_center_y - closest_y) ** 2
        )

        return distance <= proximity_threshold

    async def _handle_results(self):
        """Handle detection results asynchronously."""
        logger.info("Result handler started")

        while self._running:
            try:
                # Get result from queue
                try:
                    result = self._result_queue.get_nowait()
                except Empty:
                    await asyncio.sleep(0.1)
                    continue

                # Analyze new objects with Gemini
                if self.gemini and self.gemini.is_configured():
                    await self._analyze_new_objects(result)

                # Process events through Rule Engine
                await self._process_with_rules(result)

                # Sync tracked objects to backend for persistence
                try:
                    await backend_sync.sync_from_detection_result(
                        camera_id=result.camera_id,
                        camera_name=result.camera_id,  # Use camera_id as name for now
                        tracked_persons=result.tracked_persons,
                        tracked_vehicles=result.tracked_vehicles,
                        new_persons=result.new_persons,
                        new_vehicles=result.new_vehicles,
                        armed_persons=result.armed_persons,
                    )
                except Exception as sync_error:
                    logger.debug(f"Backend sync error (non-critical): {sync_error}")

            except Exception as e:
                logger.error(f"Result handler error: {e}")

    async def _analyze_new_objects(self, result: DetectionResult):
        """Analyze new and unanalyzed objects with Gemini.

        OPTIMAL FRAME SELECTION:
        Instead of immediately analyzing the first frame of a new track,
        we buffer frames and select the best quality frame for analysis.
        This dramatically improves results for license plates, clothing, etc.
        """
        if self._use_frame_selection:
            await self._analyze_with_frame_selection(result)
        else:
            await self._analyze_immediate(result)

    async def _analyze_with_frame_selection(self, result: DetectionResult):
        """Analyze objects using optimal frame selection.

        New tracks are added to the analysis buffer. Each frame updates the buffer
        with quality scores. When the buffer triggers (threshold, timeout, or full),
        the best frame is sent to Gemini.
        """
        # STEP 1: Start buffers for NEW tracks
        for v in result.new_vehicles:
            track_id = v.get("track_id")
            if track_id and not self.analysis_buffer.is_buffering(track_id):
                if not self.analysis_buffer.has_been_analyzed(track_id):
                    self.analysis_buffer.start_buffer(
                        track_id=track_id,
                        object_class=v.get("class", "vehicle"),
                        camera_id=result.camera_id,
                    )

        for p in result.new_persons:
            track_id = p.get("track_id")
            if track_id and not self.analysis_buffer.is_buffering(track_id):
                if not self.analysis_buffer.has_been_analyzed(track_id):
                    self.analysis_buffer.start_buffer(
                        track_id=track_id,
                        object_class="person",
                        camera_id=result.camera_id,
                    )

        # STEP 2: Add frames to buffers for ALL tracked objects (not just new)
        # This allows the buffer to collect multiple frames and select the best
        analysis_tasks = []

        for v in result.tracked_vehicles:
            track_id = v.get("track_id")
            if not track_id:
                continue

            # Skip if already analyzed
            if self.analysis_buffer.has_been_analyzed(track_id):
                continue

            # Add frame to buffer (includes quality scoring)
            quality_start = time.time()
            trigger_result = self.analysis_buffer.add_frame(
                track_id=track_id,
                frame=result.frame,
                bbox=tuple(v.get("bbox", [0, 0, 0, 0])),
                confidence=v.get("confidence", 0.5),
            )
            quality_elapsed = (time.time() - quality_start) * 1000
            self._update_timing("frame_quality_ms", quality_elapsed)

            # If buffer triggered, queue analysis
            if trigger_result:
                best_frame, best_bbox, metadata = trigger_result
                analysis_tasks.append(
                    self._run_vehicle_analysis(track_id, best_frame, best_bbox, metadata, result)
                )

        for p in result.tracked_persons:
            track_id = p.get("track_id")
            if not track_id:
                continue

            # Skip if already analyzed
            if self.analysis_buffer.has_been_analyzed(track_id):
                continue

            # Add frame to buffer (includes quality scoring)
            quality_start = time.time()
            trigger_result = self.analysis_buffer.add_frame(
                track_id=track_id,
                frame=result.frame,
                bbox=tuple(p.get("bbox", [0, 0, 0, 0])),
                confidence=p.get("confidence", 0.5),
            )
            quality_elapsed = (time.time() - quality_start) * 1000
            self._update_timing("frame_quality_ms", quality_elapsed)

            # If buffer triggered, queue analysis
            if trigger_result:
                best_frame, best_bbox, metadata = trigger_result
                analysis_tasks.append(
                    self._run_person_analysis(track_id, best_frame, best_bbox, metadata, result)
                )

        # STEP 3: Run triggered analyses (limit concurrent to avoid overloading Gemini)
        if analysis_tasks:
            # Limit to 3 concurrent analyses
            for task in analysis_tasks[:3]:
                await task

    async def _run_vehicle_analysis(
        self,
        track_id: int,
        frame: np.ndarray,
        bbox: tuple,
        frame_metadata: dict,
        result: DetectionResult,
    ):
        """Run Gemini analysis on optimal vehicle frame with image enhancement."""
        try:
            analysis_start = time.time()

            # Cooldown check
            if time.time() - self._last_gemini.get(track_id, 0) < self.config.gemini_cooldown:
                return

            # ENHANCEMENT: Enhance the frame before Gemini analysis
            enhance_start = time.time()
            if self._use_image_enhancement and self.image_enhancer:
                enhanced_frame = self.image_enhancer.enhance(frame, class_name="vehicle")
            else:
                enhanced_frame = frame
            enhance_elapsed = (time.time() - enhance_start) * 1000
            self._update_timing("image_enhance_ms", enhance_elapsed)

            # Run Gemini analysis on the ENHANCED optimal frame
            gemini_start = time.time()
            analysis = await self.gemini.analyze_vehicle(enhanced_frame, list(bbox))
            gemini_elapsed = (time.time() - gemini_start) * 1000
            self._update_timing("gemini_vehicle_ms", gemini_elapsed)

            # CRITICAL: Generate enhanced cutout for frontend display
            # This is the image that will appear in GlobalIDStore UI
            cutout_start = time.time()
            is_pre_cropped = frame_metadata.get("is_crop", False)
            enhanced_cutout_b64 = self._generate_enhanced_cutout(
                enhanced_frame, bbox, class_name="vehicle", is_pre_cropped=is_pre_cropped
            )
            cutout_elapsed = (time.time() - cutout_start) * 1000
            self._update_timing("cutout_gen_ms", cutout_elapsed)
            if enhanced_cutout_b64:
                analysis["cutout_image"] = enhanced_cutout_b64

            # Add frame selection metadata to analysis
            analysis["_frame_selection"] = {
                "quality_score": frame_metadata.get("quality_score", 0),
                "trigger_reason": frame_metadata.get("trigger_reason", "unknown"),
                "frames_buffered": frame_metadata.get("frames_buffered", 0),
                "enhanced": self._use_image_enhancement,
            }

            metadata = {
                "type": "vehicle",
                "analysis": analysis,
                "camera_id": result.camera_id,
            }

            # Update tracker with metadata
            if self.bot_sort:
                for track in self.bot_sort.get_active_tracks("vehicle", camera_id=result.camera_id):
                    if track.track_id == track_id:
                        track.metadata.update(metadata)
                        break

            self._last_gemini[track_id] = time.time()

            # Log what we're syncing (for debugging cutout issues)
            has_cutout = "cutout_image" in analysis and analysis["cutout_image"]
            cutout_size = len(analysis.get("cutout_image", "")) if has_cutout else 0
            logger.info(
                f"Analyzed vehicle {track_id} (optimal frame): "
                f"{analysis.get('manufacturer', '?')} {analysis.get('color', '?')} "
                f"plate={analysis.get('licensePlate', '?')} "
                f"quality={frame_metadata.get('quality_score', 0):.2f} enhanced={self._use_image_enhancement} "
                f"cutout={cutout_size}B"
            )

            # Sync analysis to backend (includes enhanced cutout)
            sync_start = time.time()
            try:
                await backend_sync.update_analysis(gid=track_id, analysis=analysis)
            except Exception as sync_error:
                logger.warning(f"Backend sync error for vehicle {track_id}: {sync_error}")
            sync_elapsed = (time.time() - sync_start) * 1000
            self._update_timing("backend_sync_ms", sync_elapsed)

            # SCENARIO HOOK: Report vehicle with license plate to scenario manager
            # This will trigger the Armed Attack scenario if the plate is stolen
            license_plate = analysis.get("licensePlate") or analysis.get("לוחית_רישוי")
            if license_plate and license_plate not in ["לא זוהה", "?", "", None]:
                try:
                    vehicle_data = VehicleData(
                        license_plate=license_plate,
                        color=analysis.get("color") or analysis.get("צבע"),
                        make=analysis.get("manufacturer") or analysis.get("יצרן"),
                        model=analysis.get("model") or analysis.get("דגם"),
                        camera_id=result.camera_id,
                        track_id=track_id,
                        confidence=frame_metadata.get("quality_score", 0),
                        bbox=list(bbox) if bbox else None,
                    )
                    await self.scenario_hooks.report_vehicle_detection(vehicle_data)
                except Exception as scenario_error:
                    logger.debug(f"Scenario vehicle hook error: {scenario_error}")

        except Exception as e:
            logger.error(f"Gemini vehicle analysis error: {e}")

    async def _run_person_analysis(
        self,
        track_id: int,
        frame: np.ndarray,
        bbox: tuple,
        frame_metadata: dict,
        result: DetectionResult,
    ):
        """Run Gemini analysis on optimal person frame with image enhancement."""
        try:
            analysis_start = time.time()

            # Cooldown check
            if time.time() - self._last_gemini.get(track_id, 0) < self.config.gemini_cooldown:
                return

            # ENHANCEMENT: Enhance the frame before Gemini analysis
            enhance_start = time.time()
            if self._use_image_enhancement and self.image_enhancer:
                enhanced_frame = self.image_enhancer.enhance(frame, class_name="person")
            else:
                enhanced_frame = frame
            enhance_elapsed = (time.time() - enhance_start) * 1000
            self._update_timing("image_enhance_ms", enhance_elapsed)

            # Run Gemini analysis on the ENHANCED optimal frame
            gemini_start = time.time()
            analysis = await self.gemini.analyze_person(enhanced_frame, list(bbox))
            gemini_elapsed = (time.time() - gemini_start) * 1000
            self._update_timing("gemini_person_ms", gemini_elapsed)

            # CRITICAL: Generate enhanced cutout for frontend display
            # This is the image that will appear in GlobalIDStore UI
            cutout_start = time.time()
            is_pre_cropped = frame_metadata.get("is_crop", False)
            enhanced_cutout_b64 = self._generate_enhanced_cutout(
                enhanced_frame, bbox, class_name="person", is_pre_cropped=is_pre_cropped
            )
            cutout_elapsed = (time.time() - cutout_start) * 1000
            self._update_timing("cutout_gen_ms", cutout_elapsed)
            if enhanced_cutout_b64:
                analysis["cutout_image"] = enhanced_cutout_b64

            # Add frame selection metadata to analysis
            analysis["_frame_selection"] = {
                "quality_score": frame_metadata.get("quality_score", 0),
                "trigger_reason": frame_metadata.get("trigger_reason", "unknown"),
                "frames_buffered": frame_metadata.get("frames_buffered", 0),
                "enhanced": self._use_image_enhancement,
            }

            metadata = {
                "type": "person",
                "analysis": analysis,
                "camera_id": result.camera_id,
            }

            # Update tracker with metadata
            if self.bot_sort:
                for track in self.bot_sort.get_active_tracks("person", camera_id=result.camera_id):
                    if track.track_id == track_id:
                        track.metadata.update(metadata)
                        break

            self._last_gemini[track_id] = time.time()

            # Log what we're syncing (for debugging cutout issues)
            has_cutout = "cutout_image" in analysis and analysis["cutout_image"]
            cutout_size = len(analysis.get("cutout_image", "")) if has_cutout else 0
            logger.info(
                f"Analyzed person {track_id} (optimal frame): "
                f"armed={analysis.get('armed', False)} "
                f"quality={frame_metadata.get('quality_score', 0):.2f} enhanced={self._use_image_enhancement} "
                f"cutout={cutout_size}B"
            )

            # Sync analysis to backend (includes enhanced cutout)
            sync_start = time.time()
            try:
                await backend_sync.update_analysis(gid=track_id, analysis=analysis)
            except Exception as sync_error:
                logger.warning(f"Backend sync error for person {track_id}: {sync_error}")
            sync_elapsed = (time.time() - sync_start) * 1000
            self._update_timing("backend_sync_ms", sync_elapsed)

            # Check if armed
            is_armed = analysis.get("armed") or analysis.get("חמוש")
            if is_armed:
                # Find the person dict in result and update it
                for p in result.tracked_persons:
                    if p.get("track_id") == track_id:
                        p["metadata"] = metadata
                        result.armed_persons.append(p)
                        break

                # Immediately sync armed status to backend
                try:
                    await backend_sync.mark_armed(
                        gid=track_id,
                        weapon_type=analysis.get("weaponType", analysis.get("סוג_נשק")),
                    )
                except Exception as sync_error:
                    logger.debug(f"Backend armed sync error: {sync_error}")

                # SCENARIO HOOK: Report armed person to scenario manager
                # This will count toward the armed persons threshold
                try:
                    person_data = PersonData(
                        track_id=track_id,
                        armed=True,
                        weapon_type=analysis.get("weaponType") or analysis.get("סוג_נשק"),
                        clothing=analysis.get("clothingDescription") or analysis.get("תיאור_ביגוד"),
                        clothing_color=analysis.get("clothingColor") or analysis.get("צבע_ביגוד"),
                        confidence=frame_metadata.get("quality_score", 0),
                        camera_id=result.camera_id,
                        bbox=list(bbox) if bbox else None,
                    )
                    await self.scenario_hooks.report_armed_person(person_data)
                except Exception as scenario_error:
                    logger.debug(f"Scenario armed person hook error: {scenario_error}")

        except Exception as e:
            logger.error(f"Gemini person analysis error: {e}")

    async def _analyze_immediate(self, result: DetectionResult):
        """Legacy immediate analysis (no frame selection).

        Analyzes the first frame immediately when a new track is detected.
        """
        # Collect all vehicles that need analysis (new + existing without analysis)
        vehicles_to_analyze = []

        # Add new vehicles
        for v in result.new_vehicles:
            vehicles_to_analyze.append(v)

        # Add tracked vehicles that don't have analysis yet
        for v in result.tracked_vehicles:
            track_id = v.get("track_id")
            # Check if already in list or has analysis
            if track_id and track_id not in [x.get("track_id") for x in vehicles_to_analyze]:
                metadata = v.get("metadata", {})
                if not metadata.get("analysis"):
                    vehicles_to_analyze.append(v)

        # Analyze vehicles (limit to 3 per frame to avoid overloading Gemini)
        for v in vehicles_to_analyze[:3]:
            track_id = v["track_id"]

            # Cooldown check
            if (
                time.time() - self._last_gemini.get(track_id, 0)
                < self.config.gemini_cooldown
            ):
                continue

            try:
                analysis = await self.gemini.analyze_vehicle(result.frame, v["bbox"])
                metadata = {
                    "type": "vehicle",
                    "analysis": analysis,
                    "camera_id": result.camera_id,
                }

                # Update tracker with metadata
                if self.bot_sort:
                    # Find track and update metadata (filter by camera)
                    for track in self.bot_sort.get_active_tracks(
                        "vehicle", camera_id=result.camera_id
                    ):
                        if track.track_id == track_id:
                            track.metadata.update(metadata)
                            break

                self._last_gemini[track_id] = time.time()
                # Note: gemini_calls now tracked in GeminiAnalyzer.get_call_count()
                logger.debug(
                    f"Analyzed vehicle {track_id}: {analysis.get('manufacturer', '?')} {analysis.get('color', '?')}"
                )

                # Sync analysis to backend
                try:
                    await backend_sync.update_analysis(gid=track_id, analysis=analysis)
                except Exception as sync_error:
                    logger.debug(f"Backend sync error: {sync_error}")

            except Exception as e:
                logger.error(f"Gemini vehicle analysis error: {e}")

        # Collect all persons that need analysis (new + existing without analysis)
        persons_to_analyze = []

        # Add new persons
        for p in result.new_persons:
            persons_to_analyze.append(p)

        # Add tracked persons that don't have analysis yet
        for p in result.tracked_persons:
            track_id = p.get("track_id")
            # Check if already in list or has analysis
            if track_id and track_id not in [x.get("track_id") for x in persons_to_analyze]:
                metadata = p.get("metadata", {})
                if not metadata.get("analysis"):
                    persons_to_analyze.append(p)

        # Analyze persons (limit to 3 per frame)
        for p in persons_to_analyze[:3]:
            track_id = p["track_id"]

            if (
                time.time() - self._last_gemini.get(track_id, 0)
                < self.config.gemini_cooldown
            ):
                continue

            try:
                analysis = await self.gemini.analyze_person(result.frame, p["bbox"])
                metadata = {
                    "type": "person",
                    "analysis": analysis,
                    "camera_id": result.camera_id,
                }

                # Update tracker with metadata
                if self.bot_sort:
                    # Find track and update metadata (filter by camera)
                    for track in self.bot_sort.get_active_tracks(
                        "person", camera_id=result.camera_id
                    ):
                        if track.track_id == track_id:
                            track.metadata.update(metadata)
                            break

                self._last_gemini[track_id] = time.time()
                # Note: gemini_calls now tracked in GeminiAnalyzer.get_call_count()

                # Sync analysis to backend
                try:
                    await backend_sync.update_analysis(gid=track_id, analysis=analysis)
                except Exception as sync_error:
                    logger.debug(f"Backend sync error: {sync_error}")

                # Check if armed (logging handled by Rule Engine)
                if analysis.get("armed") or analysis.get("חמוש"):
                    p["metadata"] = metadata
                    result.armed_persons.append(p)

                    # Immediately sync armed status to backend
                    try:
                        await backend_sync.mark_armed(
                            gid=track_id,
                            weapon_type=analysis.get("weaponType", analysis.get("סוג_נשק")),
                        )
                    except Exception as sync_error:
                        logger.debug(f"Backend armed sync error: {sync_error}")

            except Exception as e:
                logger.error(f"Gemini person analysis error: {e}")

    async def _send_events(self, result: DetectionResult):
        """Send detection events to backend with rate limiting."""
        if not result.new_vehicles and not result.new_persons:
            return

        camera_id = result.camera_id
        now = time.time()

        # Rate limiting - don't spam events
        last_event_time = self._last_event.get(camera_id, 0)
        if now - last_event_time < self.config.event_cooldown:
            self._stats["events_rate_limited"] += 1
            logger.debug(f"Event rate-limited for {camera_id}")
            return

        self._last_event[camera_id] = now

        try:
            # Build event
            event = {
                "type": "detection",
                "severity": "critical" if result.armed_persons else "info",
                "source": result.camera_id,
                "cameraId": result.camera_id,
                "title": self._make_title(result),
                "details": {
                    "tracked_vehicles": len(result.tracked_vehicles),
                    "tracked_persons": len(result.tracked_persons),
                    "new_vehicles": len(result.new_vehicles),
                    "new_persons": len(result.new_persons),
                    "armed": len(result.armed_persons) > 0,
                },
            }

            response = await self._http_client.post(
                f"{self.config.backend_url}/api/events", json=event
            )

            if response.status_code in (200, 201):
                self._stats["events_sent"] += 1
                logger.debug(f"Event sent: {event['title']}")
            else:
                logger.debug(f"Event send failed: {response.status_code}")

        except Exception as e:
            logger.error(f"Failed to send event: {e}")

    async def _send_alert(self, result: DetectionResult):
        """Send emergency alert."""
        camera_id = result.camera_id

        # Cooldown check
        if (
            time.time() - self._last_alert.get(camera_id, 0)
            < self.config.alert_cooldown
        ):
            return

        self._last_alert[camera_id] = time.time()
        self._stats["alerts_sent"] += 1

        armed = result.armed_persons[0]
        meta = armed.get("metadata", {})
        analysis = meta.get("analysis", {})

        alert = {
            "camera_id": camera_id,
            "timestamp": datetime.now().isoformat(),
            "person_count": len(result.tracked_persons),
            "armed_count": len(result.armed_persons),
            "armed": True,
            "weapon_type": analysis.get("weaponType", analysis.get("סוג_נשק")),
        }

        try:
            response = await self._http_client.post(
                f"{self.config.backend_url}/api/alerts", json=alert
            )
            logger.debug(f"Alert sent for {camera_id}: {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")

    async def _process_with_rules(self, result: DetectionResult):
        """Process detection result through the rule engine.

        This method creates rule contexts for different event types and
        processes them through the rule engine, which evaluates conditions,
        runs pipeline processors, and executes actions based on configured rules.

        Fallback: If no rules are loaded, use legacy _send_events and _send_alert.
        """
        try:
            # Ensure rules are loaded
            await self.rule_engine.load_rules()

            # If no rules loaded, fall back to legacy behavior
            # But ONLY if legacy behavior is explicitly enabled via environment variable
            if not self.rule_engine.rules:
                legacy_enabled = os.environ.get("LEGACY_ALERTS_ENABLED", "false").lower() == "true"
                if legacy_enabled:
                    logger.debug("No rules loaded, using legacy event handling (LEGACY_ALERTS_ENABLED=true)")
                    if self.config.send_events:
                        await self._send_events(result)
                    if result.armed_persons:
                        await self._send_alert(result)
                else:
                    logger.debug("No rules loaded and LEGACY_ALERTS_ENABLED=false, skipping legacy alerts")
                return

            # Build list of detections for rule context
            all_detections = []

            # Add tracked persons
            for p in result.tracked_persons:
                all_detections.append({
                    "class": "person",
                    "track_id": p.get("track_id"),
                    "bbox": p.get("bbox"),
                    "confidence": p.get("confidence", 0.5),
                    "metadata": p.get("metadata", {}),
                    "armed": p.get("metadata", {}).get("analysis", {}).get("armed", False)
                })

            # Add tracked vehicles
            for v in result.tracked_vehicles:
                all_detections.append({
                    "class": v.get("class", "vehicle"),
                    "track_id": v.get("track_id"),
                    "bbox": v.get("bbox"),
                    "confidence": v.get("confidence", 0.5),
                    "metadata": v.get("metadata", {})
                })

            # Get attributes from armed persons if any
            attributes = {}
            if result.armed_persons:
                armed = result.armed_persons[0]
                meta = armed.get("metadata", {})
                analysis = meta.get("analysis", {})
                attributes = {
                    "armed": True,
                    "threatLevel": "critical",
                    "weaponType": analysis.get("weaponType", analysis.get("סוג_נשק"))
                }

            # Process detection event
            if result.new_vehicles or result.new_persons or result.tracked_persons or result.tracked_vehicles:
                detection_context = RuleContext(
                    event_type="detection",
                    camera_id=result.camera_id,
                    person_count=len(result.tracked_persons),
                    vehicle_count=len(result.tracked_vehicles),
                    object_counts={
                        "person": len(result.tracked_persons),
                        "vehicle": len(result.tracked_vehicles),
                        "new_person": len(result.new_persons),
                        "new_vehicle": len(result.new_vehicles),
                    },
                    attributes=attributes,
                    frame=result.frame,
                    detections=all_detections,
                    timestamp=result.timestamp
                )

                results = await self.rule_engine.process_event(detection_context)
                if results:
                    logger.debug(f"Rule engine processed {len(results)} rules for detection")

            # Process new track events (for each new person/vehicle)
            for p in result.new_persons:
                new_track_context = RuleContext(
                    event_type="new_track",
                    camera_id=result.camera_id,
                    track_id=p.get("track_id"),
                    object_type="person",
                    confidence=p.get("confidence", 0.5),
                    bbox=p.get("bbox"),
                    attributes=p.get("metadata", {}).get("analysis", {}),
                    frame=result.frame,
                    detections=all_detections,
                    person_count=len(result.tracked_persons),
                    timestamp=result.timestamp
                )
                await self.rule_engine.process_event(new_track_context)

            for v in result.new_vehicles:
                new_track_context = RuleContext(
                    event_type="new_track",
                    camera_id=result.camera_id,
                    track_id=v.get("track_id"),
                    object_type=v.get("class", "vehicle"),
                    confidence=v.get("confidence", 0.5),
                    bbox=v.get("bbox"),
                    attributes=v.get("metadata", {}).get("analysis", {}),
                    frame=result.frame,
                    detections=all_detections,
                    vehicle_count=len(result.tracked_vehicles),
                    timestamp=result.timestamp
                )
                await self.rule_engine.process_event(new_track_context)

        except Exception as e:
            logger.error(f"Rule engine processing error: {e}", exc_info=True)
            # Fall back to legacy behavior on error - but only if explicitly enabled
            legacy_enabled = os.environ.get("LEGACY_ALERTS_ENABLED", "false").lower() == "true"
            if legacy_enabled:
                if self.config.send_events:
                    await self._send_events(result)
                if result.armed_persons:
                    await self._send_alert(result)

    def _make_title(self, result: DetectionResult) -> str:
        """Generate Hebrew title."""
        parts = []

        # Weapon detection is highest priority
        if result.armed_persons:
            weapon_count = len(result.armed_persons)
            parts.append(f"⚠️ {weapon_count} חשודים חמושים - נשק זוהה!")
        elif result.new_persons:
            parts.append(f"{len(result.new_persons)} אנשים")

        if result.new_vehicles:
            parts.append(f"{len(result.new_vehicles)} רכבים חדשים")

        return "זוהו: " + ", ".join(parts) if parts else "זיהוי חדש"

    def _update_timing(self, key: str, elapsed_ms: float):
        """Update timing stat with exponential moving average."""
        samples = self._timing_samples.get(key, 0)

        if samples < 10:
            # Use simple average for first 10 samples
            self._timing_stats[key] = (
                (self._timing_stats[key] * samples + elapsed_ms)
                / (samples + 1)
            )
        else:
            # Use exponential moving average after warmup
            self._timing_stats[key] = (
                self._timing_alpha * elapsed_ms
                + (1 - self._timing_alpha) * self._timing_stats[key]
            )

        self._timing_samples[key] = samples + 1

    def get_stats(self) -> dict:
        """Get loop statistics with comprehensive timing breakdown."""
        # Calculate FPS from frames processed
        uptime = time.time() - getattr(self, '_start_time', time.time())
        actual_fps = self._stats["frames_processed"] / max(1, uptime)

        # Calculate theoretical max FPS from timing
        total_ms = self._timing_stats.get("total_frame_ms", 0)
        theoretical_fps = 1000 / max(1, total_ms) if total_ms > 0 else 0

        # Create formatted timing with sample counts for debugging
        timing_with_samples = {}
        for key, value in self._timing_stats.items():
            samples = self._timing_samples.get(key, 0)
            timing_with_samples[key] = {
                "avg_ms": round(value, 2),
                "samples": samples,
            }

        # Calculate pipeline breakdown percentages (what % of total time each step takes)
        pipeline_breakdown = {}
        if total_ms > 0:
            pipeline_stages = [
                "yolo_ms", "yolo_postprocess_ms", "reid_extract_ms",
                "tracker_ms", "recovery_ms", "weapon_ms", "drawing_ms"
            ]
            for stage in pipeline_stages:
                stage_ms = self._timing_stats.get(stage, 0)
                percentage = (stage_ms / total_ms) * 100
                pipeline_breakdown[stage.replace("_ms", "")] = {
                    "ms": round(stage_ms, 2),
                    "pct": round(percentage, 1),
                }

        # Identify bottlenecks (stages taking >20% of total time)
        bottlenecks = []
        for stage, data in pipeline_breakdown.items():
            if data["pct"] > 20:
                bottlenecks.append(f"{stage}: {data['pct']}%")

        stats = {
            **self._stats,
            "running": self._running,
            "pending_frames": len(self._latest_frames),
            "result_queue_size": self._result_queue.qsize(),
            "active_cameras": list(self._annotated_frames.keys()),
            "actual_fps": round(actual_fps, 1),
            "theoretical_fps": round(theoretical_fps, 1),
            "uptime_seconds": round(uptime, 1),
            "timing": self._timing_stats,
            "timing_detailed": timing_with_samples,
            "pipeline_breakdown": pipeline_breakdown,
            "bottlenecks": bottlenecks if bottlenecks else ["None detected"],
            "config": {
                "detection_fps": self.config.detection_fps,
                "stream_fps": self.config.stream_fps,
                "reader_fps": self.config.reader_fps,
                "recording_fps": self.config.recording_fps,
                "use_reid_recovery": self.config.use_reid_recovery,
                "yolo_confidence": self.config.yolo_confidence,
                "weapon_confidence": self.config.weapon_confidence,
                "recovery_confidence": self.config.recovery_confidence,
            },
        }

        # Include BoT-SORT tracker stats
        if self.bot_sort:
            stats["bot_sort"] = self.bot_sort.get_stats()

        # Include ReID cache stats
        stats["reid_cache"] = self._reid_cache.get_stats()

        # Include recovery config
        stats["recovery_config"] = {
            "every_n_frames": RECOVERY_EVERY_N_FRAMES,
            "yolo_imgsz": RECOVERY_YOLO_IMGSZ,
        }

        # Include frame selection stats
        if self._use_frame_selection and self.analysis_buffer:
            stats["frame_selection"] = self.analysis_buffer.get_stats()

        # Include image enhancement stats
        if self._use_image_enhancement and self.image_enhancer:
            stats["image_enhancement"] = self.image_enhancer.get_stats()

        # Include parallel detection stats
        if self._use_parallel_detection and self.parallel_integration:
            stats["parallel_detection"] = self.parallel_integration.get_stats()
            stats["parallel_detection_enabled"] = True
        else:
            stats["parallel_detection_enabled"] = False

        return stats

    def set_fps(
        self,
        detection_fps: Optional[int] = None,
        stream_fps: Optional[int] = None,
        reader_fps: Optional[int] = None,
        recording_fps: Optional[int] = None
    ):
        """Change FPS settings dynamically.

        Args:
            detection_fps: How often to run YOLO detection (1-30)
            stream_fps: FPS for streaming annotated video (1-30)
            reader_fps: FPS for reading from RTSP camera (1-30)
            recording_fps: FPS for saved recordings (1-30)
        """
        if detection_fps is not None:
            detection_fps = max(1, min(30, detection_fps))
            self.config.detection_fps = detection_fps
            logger.info(f"Detection FPS changed to: {detection_fps}")

        if stream_fps is not None:
            stream_fps = max(1, min(30, stream_fps))
            self.config.stream_fps = stream_fps
            logger.info(f"Stream FPS changed to: {stream_fps}")

        if reader_fps is not None:
            reader_fps = max(1, min(30, reader_fps))
            self.config.reader_fps = reader_fps
            logger.info(f"Reader FPS changed to: {reader_fps}")

        if recording_fps is not None:
            recording_fps = max(1, min(30, recording_fps))
            self.config.recording_fps = recording_fps
            logger.info(f"Recording FPS changed to: {recording_fps}")

    def set_reid_recovery(self, enabled: bool):
        """Toggle ReID recovery on/off.

        Args:
            enabled: True to enable, False to disable
        """
        self.config.use_reid_recovery = enabled
        logger.info(f"ReID recovery {'enabled' if enabled else 'disabled'}")

    def set_confidence(
        self,
        yolo_confidence: Optional[float] = None,
        weapon_confidence: Optional[float] = None,
        recovery_confidence: Optional[float] = None,
    ):
        """Change confidence thresholds dynamically.

        Args:
            yolo_confidence: Main YOLO confidence (0.0-1.0)
            weapon_confidence: Weapon detection confidence (0.0-1.0)
            recovery_confidence: Recovery pass confidence (0.0-1.0)
        """
        if yolo_confidence is not None:
            yolo_confidence = max(0.0, min(1.0, yolo_confidence))
            self.config.yolo_confidence = yolo_confidence
            logger.info(f"YOLO confidence changed to: {yolo_confidence}")

        if weapon_confidence is not None:
            weapon_confidence = max(0.0, min(1.0, weapon_confidence))
            self.config.weapon_confidence = weapon_confidence
            logger.info(f"Weapon confidence changed to: {weapon_confidence}")

        if recovery_confidence is not None:
            recovery_confidence = max(0.0, min(1.0, recovery_confidence))
            self.config.recovery_confidence = recovery_confidence
            logger.info(f"Recovery confidence changed to: {recovery_confidence}")


# Global instance
_detection_loop: Optional[DetectionLoop] = None


def init_detection_loop(
    yolo_model, reid_tracker, gemini_analyzer, config: Optional[LoopConfig] = None
) -> DetectionLoop:
    """Initialize the global detection loop."""
    global _detection_loop
    _detection_loop = DetectionLoop(yolo_model, reid_tracker, gemini_analyzer, config)
    return _detection_loop


def get_detection_loop() -> Optional[DetectionLoop]:
    """Get the global detection loop."""
    return _detection_loop


def get_active_camera_ids() -> List[str]:
    """Get list of active camera IDs from the detection loop."""
    if _detection_loop:
        stats = _detection_loop.get_stats()
        return stats.get('active_cameras', [])
    return []
