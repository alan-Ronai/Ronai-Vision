"""Detection Loop - Connects RTSP cameras to AI detection pipeline.

Flow: RTSP → FFmpeg → Frames → YOLO → StableTracker → Gemini → Draw BBoxes → Events

Uses StableTracker for:
- Persistent object tracking with IoU matching
- Objects must be seen for MIN_FRAMES before reported as "new"
- Objects must be missing for MAX_MISSES frames before removed
- Same object won't be reported as "new" twice
- Event rate limiting per camera
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

from .detection import get_stable_tracker, TrackedObject

logger = logging.getLogger(__name__)


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
        'person': (0, 255, 0),       # Green
        'person_armed': (0, 0, 255), # Red
        'car': (255, 165, 0),        # Orange
        'truck': (255, 100, 0),      # Dark Orange
        'motorcycle': (255, 255, 0), # Cyan
        'bus': (128, 0, 128),        # Purple
        'bicycle': (0, 255, 255),    # Yellow
        'vehicle': (255, 165, 0),    # Orange for generic vehicle
        'predicted': (128, 128, 128), # Gray for predicted positions
        'default': (200, 200, 200)   # Gray
    }

    @staticmethod
    def draw_detections(
        frame: np.ndarray,
        detections: List[Dict],
        detection_type: str = 'object'
    ) -> np.ndarray:
        """Draw bounding boxes and labels on frame."""
        annotated = frame.copy()

        for det in detections:
            bbox = det.get('bbox', det.get('box', []))
            if not bbox or len(bbox) < 4:
                continue

            x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
            track_id = det.get('track_id', det.get('id', '?'))
            label = det.get('class', det.get('label', detection_type))
            confidence = det.get('confidence', det.get('conf', 0))
            metadata = det.get('metadata', {})

            # Determine if armed
            is_armed = False
            if metadata:
                analysis = metadata.get('analysis', metadata)
                is_armed = analysis.get('armed', False) or analysis.get('חמוש', False)

            # Check if this is a predicted (not detected) position
            is_predicted = det.get('is_predicted', False) or det.get('consecutive_misses', 0) > 0

            # Select color
            if is_armed:
                color = BBoxDrawer.COLORS['person_armed']
            elif is_predicted:
                color = BBoxDrawer.COLORS['predicted']
            else:
                color = BBoxDrawer.COLORS.get(label, BBoxDrawer.COLORS['default'])

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
                weapon_type = analysis.get('weaponType', analysis.get('סוג_נשק', ''))
                label_parts.append(f"ARMED")
                if weapon_type and weapon_type != 'לא רלוונטי':
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
                -1  # Filled
            )

            # Text
            cv2.putText(
                annotated,
                label_text,
                (label_x + 5, label_y),
                font,
                font_scale,
                (255, 255, 255),  # White text
                font_thickness
            )

            # Draw armed indicator
            if is_armed:
                # Flashing border effect
                cv2.rectangle(annotated, (x1-2, y1-2), (x2+2, y2+2), (0, 0, 255), 4)

        return annotated

    @staticmethod
    def draw_status_overlay(
        frame: np.ndarray,
        camera_id: str,
        vehicle_count: int,
        person_count: int,
        armed_count: int,
        fps: float = 0
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
            2
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
            2
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
                2
            )
            # Red border
            cv2.rectangle(annotated, (0, 0), (w-1, h-1), (0, 0, 255), 4)

        # Timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(
            annotated,
            timestamp,
            (w - 100, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1
        )

        return annotated


@dataclass
class LoopConfig:
    """Detection loop configuration."""
    backend_url: str = "http://localhost:3000"
    detection_fps: int = 15  # Process all frames (15 FPS) for continuous analysis
    stream_fps: int = 15     # Stream at 15 fps for smooth viewing
    alert_cooldown: float = 30.0
    gemini_cooldown: float = 5.0  # Don't analyze same track more than once per 5 sec
    event_cooldown: float = 5.0  # Minimum seconds between events per camera
    draw_bboxes: bool = True
    send_events: bool = True
    use_stable_tracker: bool = True  # Use StableTracker instead of ReID


class DetectionLoop:
    """Main detection loop - connects cameras to detection pipeline.

    Uses StableTracker for persistent tracking:
    - Objects must be seen for 3 frames before reported as "new"
    - Objects must be missing for 30 frames before removed
    - Same object won't be reported as "new" twice
    """

    def __init__(
        self,
        yolo_model,
        reid_tracker,
        gemini_analyzer,
        config: Optional[LoopConfig] = None
    ):
        self.yolo = yolo_model
        self.reid_tracker = reid_tracker  # Keep for backward compatibility
        self.gemini = gemini_analyzer
        self.config = config or LoopConfig()
        self.drawer = BBoxDrawer()

        # Use StableTracker for persistent tracking
        self.stable_tracker = get_stable_tracker() if self.config.use_stable_tracker else None

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
            "gemini_calls": 0
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
                    if result.new_vehicles or result.new_persons or result.armed_persons:
                        try:
                            self._result_queue.put_nowait(result)
                        except:
                            pass

            except Exception as e:
                logger.error(f"Detection loop error: {e}")

    def _detect_frame(self, camera_id: str, frame: np.ndarray) -> Optional[DetectionResult]:
        """Run detection on a single frame.

        Uses StableTracker for persistent tracking:
        - Objects must be seen for 3 frames before reported as "new"
        - Same object won't be reported twice
        """
        try:
            # Run YOLO with higher confidence threshold to reduce false detections
            yolo_results = self.yolo(frame, verbose=False, conf=0.55)[0]

            # Separate detections
            vehicle_dets = []
            person_dets = []
            vehicle_classes = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']

            for box in yolo_results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = yolo_results.names[cls]
                bbox = box.xyxy[0].tolist()

                det = {
                    'bbox': bbox,
                    'confidence': conf,
                    'class': label
                }

                if label in vehicle_classes:
                    vehicle_dets.append(det)
                elif label == 'person':
                    person_dets.append(det)

            self._stats["detections"] += len(vehicle_dets) + len(person_dets)

            # Track with StableTracker (preferred) or ReID
            tracked_vehicles = []
            tracked_persons = []
            new_vehicles = []
            new_persons = []

            if self.stable_tracker:
                # Use StableTracker for persistent tracking
                all_vehicles, newly_reportable_vehicles = self.stable_tracker.update(
                    vehicle_dets, 'vehicle'
                )
                all_persons, newly_reportable_persons = self.stable_tracker.update(
                    person_dets, 'person'
                )

                # Convert TrackedObject to dict for compatibility
                tracked_vehicles = [
                    {
                        'track_id': v.track_id,
                        'bbox': list(v.bbox),
                        'confidence': v.confidence,
                        'class': v.class_name,
                        'metadata': v.metadata,
                        'consecutive_misses': v.consecutive_misses,
                        'is_predicted': v.consecutive_misses > 0
                    }
                    for v in all_vehicles
                ]
                tracked_persons = [
                    {
                        'track_id': p.track_id,
                        'bbox': list(p.bbox),
                        'confidence': p.confidence,
                        'class': p.class_name,
                        'metadata': p.metadata,
                        'consecutive_misses': p.consecutive_misses,
                        'is_predicted': p.consecutive_misses > 0
                    }
                    for p in all_persons
                ]

                # Only report truly NEW objects (seen enough times, not reported before)
                new_vehicles = [
                    {
                        'track_id': v.track_id,
                        'bbox': list(v.bbox),
                        'confidence': v.confidence,
                        'class': v.class_name,
                        'metadata': v.metadata
                    }
                    for v in newly_reportable_vehicles
                ]
                new_persons = [
                    {
                        'track_id': p.track_id,
                        'bbox': list(p.bbox),
                        'confidence': p.confidence,
                        'class': p.class_name,
                        'metadata': p.metadata
                    }
                    for p in newly_reportable_persons
                ]

            elif self.reid_tracker:
                # Fallback to ReID tracker
                if vehicle_dets:
                    v_bboxes = [d['bbox'] for d in vehicle_dets]
                    v_confs = [d['confidence'] for d in vehicle_dets]
                    tracked_vehicles = self.reid_tracker.update_vehicles(
                        list(zip(v_bboxes, v_confs, [d['class'] for d in vehicle_dets])),
                        frame
                    )

                if person_dets:
                    p_bboxes = [d['bbox'] for d in person_dets]
                    p_confs = [d['confidence'] for d in person_dets]
                    tracked_persons = self.reid_tracker.update_persons(
                        list(zip(p_bboxes, p_confs, ['person'] * len(person_dets))),
                        frame
                    )

                # Find new objects (not analyzed yet)
                for v in tracked_vehicles:
                    if not self.reid_tracker.has_been_analyzed(v['track_id']):
                        new_vehicles.append(v)

                for p in tracked_persons:
                    if not self.reid_tracker.has_been_analyzed(p['track_id']):
                        new_persons.append(p)
            else:
                # No tracker, just use raw detections (not recommended)
                for i, d in enumerate(vehicle_dets):
                    d['track_id'] = f"v_{i}"
                    tracked_vehicles.append(d)
                for i, d in enumerate(person_dets):
                    d['track_id'] = f"p_{i}"
                    tracked_persons.append(d)

            # Check for armed persons
            armed_persons = []
            for p in tracked_persons:
                meta = p.get('metadata', {})
                if meta:
                    if meta.get('armed') or meta.get('חמוש'):
                        armed_persons.append(p)

            # Draw annotations
            annotated_frame = None
            if self.config.draw_bboxes:
                annotated_frame = frame.copy()

                # Draw vehicles
                annotated_frame = self.drawer.draw_detections(
                    annotated_frame, tracked_vehicles, 'vehicle'
                )

                # Draw persons
                annotated_frame = self.drawer.draw_detections(
                    annotated_frame, tracked_persons, 'person'
                )

                # Draw overlay
                annotated_frame = self.drawer.draw_status_overlay(
                    annotated_frame,
                    camera_id,
                    len(tracked_vehicles),
                    len(tracked_persons),
                    len(armed_persons)
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
                annotated_frame=annotated_frame
            )

        except Exception as e:
            logger.error(f"Detection error for {camera_id}: {e}")
            return None

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
            track_id = v['track_id']

            # Cooldown check
            if time.time() - self._last_gemini.get(track_id, 0) < self.config.gemini_cooldown:
                continue

            try:
                analysis = await self.gemini.analyze_vehicle(result.frame, v['bbox'])
                metadata = {
                    'type': 'vehicle',
                    'analysis': analysis,
                    'camera_id': result.camera_id
                }

                # Update tracker with metadata
                if self.stable_tracker:
                    self.stable_tracker.mark_analyzed(track_id, metadata)
                elif self.reid_tracker:
                    self.reid_tracker.save_metadata(track_id, metadata)

                self._last_gemini[track_id] = time.time()
                self._stats["gemini_calls"] += 1
                logger.info(f"Analyzed vehicle {track_id}: {analysis.get('manufacturer', '?')} {analysis.get('color', '?')}")
            except Exception as e:
                logger.error(f"Gemini vehicle analysis error: {e}")

        # Analyze new persons
        for p in result.new_persons[:3]:
            track_id = p['track_id']

            if time.time() - self._last_gemini.get(track_id, 0) < self.config.gemini_cooldown:
                continue

            try:
                analysis = await self.gemini.analyze_person(result.frame, p['bbox'])
                metadata = {
                    'type': 'person',
                    'analysis': analysis,
                    'camera_id': result.camera_id
                }

                # Update tracker with metadata
                if self.stable_tracker:
                    self.stable_tracker.mark_analyzed(track_id, metadata)
                elif self.reid_tracker:
                    self.reid_tracker.save_metadata(track_id, metadata)

                self._last_gemini[track_id] = time.time()
                self._stats["gemini_calls"] += 1

                # Check if armed
                if analysis.get('armed') or analysis.get('חמוש'):
                    logger.warning(f"ARMED PERSON DETECTED: {track_id}")
                    p['metadata'] = metadata
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
                    "armed": len(result.armed_persons) > 0
                }
            }

            response = await self._http_client.post(
                f"{self.config.backend_url}/api/events",
                json=event
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
        if time.time() - self._last_alert.get(camera_id, 0) < self.config.alert_cooldown:
            return

        self._last_alert[camera_id] = time.time()
        self._stats["alerts_sent"] += 1

        armed = result.armed_persons[0]
        meta = armed.get('metadata', {})
        analysis = meta.get('analysis', {})

        alert = {
            "camera_id": camera_id,
            "timestamp": datetime.now().isoformat(),
            "person_count": len(result.tracked_persons),
            "armed_count": len(result.armed_persons),
            "armed": True,
            "weapon_type": analysis.get('weaponType', analysis.get('סוג_נשק')),
        }

        try:
            response = await self._http_client.post(
                f"{self.config.backend_url}/api/alerts",
                json=alert
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
            "active_cameras": list(self._annotated_frames.keys())
        }

        # Include StableTracker stats
        if self.stable_tracker:
            stats["stable_tracker"] = self.stable_tracker.get_stats()

        return stats


# Global instance
_detection_loop: Optional[DetectionLoop] = None

def init_detection_loop(yolo_model, reid_tracker, gemini_analyzer, config: Optional[LoopConfig] = None) -> DetectionLoop:
    """Initialize the global detection loop."""
    global _detection_loop
    _detection_loop = DetectionLoop(yolo_model, reid_tracker, gemini_analyzer, config)
    return _detection_loop

def get_detection_loop() -> Optional[DetectionLoop]:
    """Get the global detection loop."""
    return _detection_loop
