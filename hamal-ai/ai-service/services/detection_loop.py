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
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from queue import Queue, Empty
import threading

from .detection import get_bot_sort_tracker, Detection, Track

logger = logging.getLogger(__name__)

# Detection Recovery Configuration
RECOVERY_CONFIDENCE = 0.15  # Very low threshold for recovery pass
RECOVERY_IOU_THRESH = 0.3  # Minimum IoU with Kalman prediction (lowered for better recovery)
RECOVERY_REID_THRESH = 0.5  # Minimum ReID similarity (for persons) (lowered for better recovery)
RECOVERY_MIN_TRACK_CONFIDENCE = 0.25  # Only recover tracks with decent history


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
                if weapon_type and weapon_type != "לא רלוונטי":
                    label_parts.append(weapon_type)

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

    backend_url: str = "http://localhost:3000"
    detection_fps: int = 15  # Process all frames (15 FPS) for continuous analysis
    stream_fps: int = 15  # Stream at 15 fps for smooth viewing
    alert_cooldown: float = 30.0
    gemini_cooldown: float = 5.0  # Don't analyze same track more than once per 5 sec
    event_cooldown: float = 5.0  # Minimum seconds between events per camera
    draw_bboxes: bool = True
    send_events: bool = True
    use_bot_sort: bool = True  # Use BoT-SORT tracker with full Kalman filter
    use_reid_recovery: bool = True  # Enable ReID-based detection recovery


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
        self.reid_tracker = reid_tracker  # Used for ReID feature extraction
        self.gemini = gemini_analyzer
        self.config = config or LoopConfig()
        self.drawer = BBoxDrawer()

        # Use BoT-SORT tracker for advanced tracking
        self.bot_sort = get_bot_sort_tracker() if self.config.use_bot_sort else None

        self._running = False
        self._frame_queue: Queue = Queue(maxsize=50)
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
            "gemini_calls": 0,
        }

    def on_frame(self, camera_id: str, frame: np.ndarray):
        """Callback when frame received from RTSP reader."""
        try:
            self._frame_queue.put_nowait((camera_id, frame, time.time()))
        except:
            pass  # Queue full, drop frame

    def get_annotated_frame(self, camera_id: str) -> Optional[np.ndarray]:
        """Get latest annotated frame for streaming."""
        with self._frame_lock:
            return self._annotated_frames.get(camera_id)

    async def start(self):
        """Start the detection loop."""
        if self._running:
            return

        self._running = True
        self._http_client = httpx.AsyncClient(timeout=30.0)

        # Start processing thread
        self._process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._process_thread.start()

        # Start async result handler
        asyncio.create_task(self._handle_results())

        logger.info("Detection loop started")

    async def stop(self):
        """Stop the detection loop."""
        self._running = False

        if self._http_client:
            await self._http_client.aclose()

        logger.info("Detection loop stopped")

    def _process_loop(self):
        """Process frames in background thread (blocking operations)."""
        logger.info("Detection processing thread started")

        frame_interval = 1.0 / self.config.detection_fps
        last_process_time: Dict[str, float] = {}

        while self._running:
            try:
                # Get frame from queue
                try:
                    camera_id, frame, timestamp = self._frame_queue.get(timeout=0.5)
                except Empty:
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

                # Run detection
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
                        except:
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
            # STEP 1: Run YOLO with moderate confidence threshold (lowered to reduce flickering)
            yolo_results = self.yolo(frame, verbose=False, conf=0.45)[0]

            # Separate detections by class
            vehicle_detections = []
            person_detections = []
            vehicle_classes = ["car", "truck", "bus", "motorcycle", "bicycle"]

            for box in yolo_results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = yolo_results.names[cls]
                xyxy = box.xyxy[0].tolist()

                # Convert from xyxy to xywh format
                x1, y1, x2, y2 = xyxy
                bbox = (x1, y1, x2 - x1, y2 - y1)

                # Extract ReID feature for persons (if tracker available)
                feature = None
                if label == "person" and self.reid_tracker:
                    try:
                        # Extract ReID embedding from person crop
                        feature = self._extract_reid_feature(frame, xyxy)
                    except Exception as e:
                        logger.debug(f"ReID extraction failed: {e}")

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

            self._stats["detections"] += len(vehicle_detections) + len(
                person_detections
            )

            # STEP 2: Update BoT-SORT tracker
            tracked_vehicles = []
            tracked_persons = []
            new_vehicles = []
            new_persons = []

            if self.bot_sort:
                # Update BoT-SORT tracker with detections
                all_vehicles, new_vehicle_tracks = self.bot_sort.update(
                    vehicle_detections, "vehicle", dt=1 / self.config.detection_fps
                )
                all_persons, new_person_tracks = self.bot_sort.update(
                    person_detections, "person", dt=1 / self.config.detection_fps
                )

                # STEP 3: Run detection recovery for lost tracks (if enabled)
                if self.config.use_reid_recovery:
                    # Recover vehicles using IoU matching
                    recovered_vehicles = self._recover_missing_detections(
                        frame, all_vehicles, "vehicle"
                    )
                    if recovered_vehicles:
                        logger.debug(f"Recovering {len(recovered_vehicles)} vehicles")
                        _, _ = self.bot_sort.update(
                            recovered_vehicles,
                            "vehicle",
                            dt=1 / self.config.detection_fps,
                        )
                        # Refresh track list
                        all_vehicles = self.bot_sort.get_active_tracks("vehicle")

                    # Recover persons using IoU + ReID matching
                    recovered_persons = self._recover_missing_detections(
                        frame, all_persons, "person"
                    )
                    if recovered_persons:
                        logger.debug(f"Recovering {len(recovered_persons)} persons")
                        _, _ = self.bot_sort.update(
                            recovered_persons,
                            "person",
                            dt=1 / self.config.detection_fps,
                        )
                        # Refresh track list
                        all_persons = self.bot_sort.get_active_tracks("person")

                # Convert Track objects to dicts for compatibility with rest of pipeline
                tracked_vehicles = [self._track_to_dict(t) for t in all_vehicles]
                tracked_persons = [self._track_to_dict(t) for t in all_persons]

                # New tracks that should be reported
                new_vehicles = [self._track_to_dict(t) for t in new_vehicle_tracks]
                new_persons = [self._track_to_dict(t) for t in new_person_tracks]
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

            # Check for armed persons
            armed_persons = []
            for p in tracked_persons:
                meta = p.get("metadata", {})
                if meta:
                    if meta.get("armed") or meta.get("חמוש"):
                        armed_persons.append(p)

            # Draw annotations
            annotated_frame = None
            if self.config.draw_bboxes:
                annotated_frame = frame.copy()

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
        self, frame: np.ndarray, xyxy: List[float]
    ) -> Optional[np.ndarray]:
        """Extract ReID feature vector from person crop.

        Args:
            frame: Full frame image
            xyxy: Bounding box in [x1, y1, x2, y2] format

        Returns:
            ReID feature vector (normalized) or None if extraction fails
        """
        try:
            if not self.reid_tracker:
                return None

            x1, y1, x2, y2 = [int(v) for v in xyxy]
            h, w = frame.shape[:2]

            # Clamp to frame bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                return None

            # Crop person
            crop = frame[y1:y2, x1:x2]

            # Use ReID tracker to extract feature (assuming it has embedding method)
            # This is a placeholder - adapt to your actual ReID tracker interface
            if hasattr(self.reid_tracker, "extract_features"):
                feature = self.reid_tracker.extract_features(crop)
                # Normalize
                if feature is not None and len(feature) > 0:
                    norm = np.linalg.norm(feature)
                    if norm > 0:
                        return feature / norm

            return None
        except Exception as e:
            logger.debug(f"ReID feature extraction error: {e}")
            return None

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

    def _recover_missing_detections(
        self, frame: np.ndarray, active_tracks: List[Track], object_type: str
    ) -> List[Detection]:
        """Recover missed detections for lost tracks using ReID matching.

        This is the CRITICAL FEATURE for robust tracking. When a track loses detection
        due to momentary occlusion or YOLO failure, we:
        1. Run a low-threshold YOLO pass (conf=0.15)
        2. Match candidates using ReID similarity (for persons) or IoU (for vehicles)
        3. Only accept matches that align with Kalman predictions

        Args:
            frame: Current frame
            active_tracks: All active tracks for this object type
            object_type: 'person' or 'vehicle'

        Returns:
            List of recovered Detection objects
        """
        try:
            # STEP 1: Identify tracks that need recovery
            lost_tracks = [
                t
                for t in active_tracks
                if t.time_since_update > 0  # Lost detection this frame
                and t.avg_confidence
                > RECOVERY_MIN_TRACK_CONFIDENCE  # Good track history
                and t.state == "confirmed"  # Only recover confirmed tracks
            ]

            if not lost_tracks:
                return []

            logger.debug(
                f"Recovery for {object_type}: {len(lost_tracks)} lost tracks out of {len(active_tracks)} active"
            )

            # STEP 2: Run low-threshold YOLO detection
            low_conf_results = self.yolo(
                frame, verbose=False, conf=RECOVERY_CONFIDENCE
            )[0]

            recovery_candidates = []
            target_classes = (
                ["person"]
                if object_type == "person"
                else ["car", "truck", "bus", "motorcycle", "bicycle"]
            )

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

                # Extract ReID feature for persons
                feature = None
                if object_type == "person" and self.reid_tracker:
                    feature = self._extract_reid_feature(frame, xyxy)

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

            # STEP 3: Match candidates to lost tracks
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
                logger.info(f"Recovered {len(recovered)} {object_type} detections")

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

                # Send events
                if self.config.send_events:
                    await self._send_events(result)

                # Check for alerts
                if result.armed_persons:
                    await self._send_alert(result)

            except Exception as e:
                logger.error(f"Result handler error: {e}")

    async def _analyze_new_objects(self, result: DetectionResult):
        """Analyze new objects with Gemini."""
        # Analyze new vehicles
        for v in result.new_vehicles[:3]:  # Limit to 3 per frame
            track_id = v["track_id"]

            # Cooldown check
            if (
                time.time() - self._last_gemini.get(track_id, 0)
                < self.config.gemini_cooldown
            ):
                continue

            try:
                analysis = await self.gemini.analyze_veihcle(result.frame, v["bbox"])
                metadata = {
                    "type": "vehicle",
                    "analysis": analysis,
                    "camera_id": result.camera_id,
                }

                # Update tracker with metadata
                if self.bot_sort:
                    # Find track and update metadata
                    for track in self.bot_sort.get_active_tracks("vehicle"):
                        if track.track_id == track_id:
                            track.metadata.update(metadata)
                            break

                self._last_gemini[track_id] = time.time()
                self._stats["gemini_calls"] += 1
                logger.info(
                    f"Analyzed vehicle {track_id}: {analysis.get('manufacturer', '?')} {analysis.get('color', '?')}"
                )
            except Exception as e:
                logger.error(f"Gemini vehicle analysis error: {e}")

        # Analyze new persons
        for p in result.new_persons[:3]:
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
                    # Find track and update metadata
                    for track in self.bot_sort.get_active_tracks("person"):
                        if track.track_id == track_id:
                            track.metadata.update(metadata)
                            break

                self._last_gemini[track_id] = time.time()
                self._stats["gemini_calls"] += 1

                # Check if armed
                if analysis.get("armed") or analysis.get("חמוש"):
                    logger.warning(f"ARMED PERSON DETECTED: {track_id}")
                    p["metadata"] = metadata
                    result.armed_persons.append(p)

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
                logger.info(f"Event sent: {event['title']}")
            else:
                logger.warning(f"Event send failed: {response.status_code}")

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
            logger.warning(f"ALERT sent for {camera_id}: {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")

    def _make_title(self, result: DetectionResult) -> str:
        """Generate Hebrew title."""
        parts = []

        if result.new_vehicles:
            parts.append(f"{len(result.new_vehicles)} רכבים חדשים")

        if result.new_persons:
            if result.armed_persons:
                parts.append(f"{len(result.armed_persons)} חשודים חמושים!")
            else:
                parts.append(f"{len(result.new_persons)} אנשים")

        return "זוהו: " + ", ".join(parts) if parts else "זיהוי חדש"

    def get_stats(self) -> dict:
        """Get loop statistics."""
        stats = {
            **self._stats,
            "running": self._running,
            "frame_queue_size": self._frame_queue.qsize(),
            "result_queue_size": self._result_queue.qsize(),
            "active_cameras": list(self._annotated_frames.keys()),
        }

        # Include BoT-SORT tracker stats
        if self.bot_sort:
            stats["bot_sort"] = self.bot_sort.get_stats()

        return stats


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
