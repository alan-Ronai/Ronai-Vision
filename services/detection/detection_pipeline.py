"""Main detection pipeline integrating YOLO, ReID, and Gemini.

This is the core detection flow:
1. YOLO detects objects (fast)
2. ReID tracks them with persistent IDs
3. Gemini analyzes NEW objects only (saves API calls)
4. Alerts triggered if armed persons detected
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

# Import services
from services.gemini.analyzer import get_gemini_analyzer, GeminiAnalyzer
from services.tracking.reid_tracker import (
    get_reid_tracker,
    ReIDTracker,
    TrackedObject
)
from services.tts.google_tts import get_tts_service

# Try to import YOLO
YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    logger.warning("ultralytics not installed. YOLO detection disabled.")


@dataclass
class DetectionResult:
    """Result from detection pipeline."""
    camera_id: str
    timestamp: datetime
    tracked_vehicles: List[TrackedObject] = field(default_factory=list)
    tracked_persons: List[TrackedObject] = field(default_factory=list)
    armed_persons: List[TrackedObject] = field(default_factory=list)
    new_vehicles: List[TrackedObject] = field(default_factory=list)
    new_persons: List[TrackedObject] = field(default_factory=list)
    alert_triggered: bool = False
    alert_data: Optional[Dict] = None

    def to_dict(self) -> Dict:
        return {
            "camera_id": self.camera_id,
            "timestamp": self.timestamp.isoformat(),
            "tracked_vehicles": len(self.tracked_vehicles),
            "tracked_persons": len(self.tracked_persons),
            "armed_persons": len(self.armed_persons),
            "new_vehicles": len(self.new_vehicles),
            "new_persons": len(self.new_persons),
            "alert_triggered": self.alert_triggered,
            "vehicles": [v.to_dict() for v in self.tracked_vehicles],
            "persons": [p.to_dict() for p in self.tracked_persons]
        }


class DetectionPipeline:
    """Main detection pipeline with ReID and Gemini integration.

    Flow:
    1. YOLO detects objects in frame
    2. ReID tracker assigns persistent IDs
    3. Gemini analyzes NEW objects only (saves API calls)
    4. If armed person detected, trigger alert with TTS
    """

    # YOLO classes we care about
    VEHICLE_CLASSES = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']
    PERSON_CLASSES = ['person']

    def __init__(
        self,
        yolo_model: str = "yolov8x.pt",
        weapon_model: Optional[str] = None,
        device: str = "auto",
        confidence: float = 0.25
    ):
        """Initialize detection pipeline.

        Args:
            yolo_model: Path to YOLO model weights
            weapon_model: Optional path to weapon detection model
            device: Device to run on ('auto', 'cpu', 'cuda', 'mps')
            confidence: Detection confidence threshold
        """
        # Detect device
        if device == "auto":
            try:
                import torch
                if torch.backends.mps.is_available():
                    self.device = "mps"
                elif torch.cuda.is_available():
                    self.device = "cuda"
                else:
                    self.device = "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        self.confidence = confidence
        logger.info(f"Detection pipeline using device: {self.device}")

        # Load YOLO
        if YOLO_AVAILABLE:
            try:
                self.yolo = YOLO(yolo_model)
                self.yolo.to(self.device)
                logger.info(f"Loaded YOLO model: {yolo_model}")
            except Exception as e:
                logger.error(f"Failed to load YOLO model: {e}")
                self.yolo = None

            # Optional weapon model
            if weapon_model:
                try:
                    self.weapon_detector = YOLO(weapon_model)
                    self.weapon_detector.to(self.device)
                    logger.info(f"Loaded weapon model: {weapon_model}")
                except Exception as e:
                    logger.warning(f"Failed to load weapon model: {e}")
                    self.weapon_detector = None
            else:
                self.weapon_detector = None
        else:
            self.yolo = None
            self.weapon_detector = None

        # Get services
        self.tracker = get_reid_tracker()
        self.gemini = get_gemini_analyzer()
        self.tts = get_tts_service()

        # Alert callback (set by application)
        self.on_alert: Optional[Callable[[Dict], Awaitable[None]]] = None

        logger.info("DetectionPipeline initialized")
        logger.info(f"  - YOLO: {'✅' if self.yolo else '❌'}")
        logger.info(f"  - Weapon: {'✅' if self.weapon_detector else '❌'}")
        logger.info(f"  - ReID: {'✅' if self.tracker else '❌'}")
        logger.info(f"  - Gemini: {'✅' if self.gemini else '❌'}")
        logger.info(f"  - TTS: {'✅' if self.tts else '❌'}")

    async def process_frame(
        self,
        frame: np.ndarray,
        camera_id: str = "unknown"
    ) -> DetectionResult:
        """Process a single frame through the detection pipeline.

        Args:
            frame: BGR image (H, W, 3) uint8
            camera_id: Camera identifier

        Returns:
            DetectionResult with all detections and tracking info
        """
        timestamp = datetime.now()
        new_vehicles: List[TrackedObject] = []
        new_persons: List[TrackedObject] = []
        armed_persons: List[TrackedObject] = []
        alert_triggered = False
        alert_data = None

        # Step 1: YOLO detection
        vehicle_dets, person_dets = self._run_yolo(frame)

        # Step 2: ReID tracking
        tracked_vehicles: List[TrackedObject] = []
        tracked_persons: List[TrackedObject] = []

        if self.tracker:
            tracked_vehicles = self.tracker.update_vehicles(vehicle_dets, frame)
            tracked_persons = self.tracker.update_persons(person_dets, frame)

        # Step 3: Analyze NEW objects with Gemini
        if self.gemini and self.tracker:
            # Analyze new vehicles
            for vehicle in tracked_vehicles:
                if not self.tracker.is_analyzed(vehicle.track_id, "vehicle"):
                    try:
                        analysis = await self.gemini.analyze_car_async(frame, vehicle.bbox)
                        self.tracker.save_metadata(vehicle.track_id, "vehicle", analysis)
                        new_vehicles.append(vehicle)
                        logger.info(
                            f"Analyzed vehicle {vehicle.track_id}: "
                            f"{analysis.get('צבע')} {analysis.get('דגם')}"
                        )
                    except Exception as e:
                        logger.error(f"Vehicle analysis failed: {e}")

            # Analyze new persons
            for person in tracked_persons:
                if not self.tracker.is_analyzed(person.track_id, "person"):
                    try:
                        analysis = await self.gemini.analyze_person_async(frame, person.bbox)

                        # Check if armed
                        is_armed = self.gemini.is_person_armed(analysis)
                        analysis["armed"] = is_armed

                        self.tracker.save_metadata(person.track_id, "person", analysis)
                        new_persons.append(person)

                        if is_armed:
                            armed_persons.append(person)
                            logger.warning(f"ARMED PERSON DETECTED: track {person.track_id}")

                        logger.info(f"Analyzed person {person.track_id}: armed={is_armed}")
                    except Exception as e:
                        logger.error(f"Person analysis failed: {e}")

        # Step 4: Check for alerts
        if armed_persons:
            alert_triggered = True
            alert_data = await self._trigger_alert(
                camera_id=camera_id,
                armed_persons=armed_persons,
                all_persons=tracked_persons,
                all_vehicles=tracked_vehicles
            )

        return DetectionResult(
            camera_id=camera_id,
            timestamp=timestamp,
            tracked_vehicles=tracked_vehicles,
            tracked_persons=tracked_persons,
            armed_persons=armed_persons,
            new_vehicles=new_vehicles,
            new_persons=new_persons,
            alert_triggered=alert_triggered,
            alert_data=alert_data
        )

    def _run_yolo(self, frame: np.ndarray) -> tuple:
        """Run YOLO detection on frame.

        Returns:
            (vehicle_detections, person_detections)
            Each is a list of (bbox, confidence, class_name) tuples
        """
        if not self.yolo:
            return [], []

        results = self.yolo(frame, verbose=False, conf=self.confidence)[0]

        vehicle_dets = []
        person_dets = []

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = results.names[cls]
            bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

            if label in self.VEHICLE_CLASSES:
                vehicle_dets.append((bbox, conf, label))
            elif label in self.PERSON_CLASSES:
                person_dets.append((bbox, conf, label))

        return vehicle_dets, person_dets

    async def _trigger_alert(
        self,
        camera_id: str,
        armed_persons: List[TrackedObject],
        all_persons: List[TrackedObject],
        all_vehicles: List[TrackedObject]
    ) -> Dict:
        """Trigger emergency alert.

        Returns:
            Alert data dictionary
        """
        # Gather vehicle info
        vehicle_info = None
        if all_vehicles:
            v = all_vehicles[0]
            vehicle_info = v.metadata if v.metadata else None

        # Get weapon type from first armed person
        weapon_type = None
        for ap in armed_persons:
            if ap.metadata:
                weapon_type = ap.metadata.get("סוג_נשק")
                if weapon_type and weapon_type != "לא רלוונטי":
                    break

        alert_data = {
            "camera_id": camera_id,
            "timestamp": datetime.now().isoformat(),
            "person_count": len(all_persons),
            "armed_count": len(armed_persons),
            "armed": True,
            "weapon_type": weapon_type,
            "vehicle": vehicle_info,
            "armed_persons": [ap.to_dict() for ap in armed_persons]
        }

        # Generate TTS alert
        if self.tts:
            try:
                audio_path = self.tts.generate_emergency_alert(
                    camera_id=camera_id,
                    person_count=len(all_persons),
                    armed=True,
                    weapon_type=weapon_type,
                    vehicle=vehicle_info
                )
                alert_data["audio_path"] = audio_path
            except Exception as e:
                logger.error(f"TTS generation failed: {e}")

        # Call alert callback if set
        if self.on_alert:
            try:
                await self.on_alert(alert_data)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

        logger.warning(f"EMERGENCY ALERT: camera={camera_id}, armed={len(armed_persons)}")
        return alert_data

    async def check_threat_neutralized(self, frame: np.ndarray) -> Dict:
        """Check if threat is neutralized (for body cam analysis).

        Args:
            frame: Frame from body camera

        Returns:
            Threat status dict
        """
        if not self.gemini:
            return {"error": "Gemini not available"}

        result = await self.gemini.analyze_threat_neutralized_async(frame)

        # Generate end incident audio if neutralized
        if result.get("איום_נוטרל") and self.tts:
            try:
                audio_path = self.tts.generate_end_incident()
                result["audio_path"] = audio_path
            except Exception as e:
                logger.error(f"End incident TTS failed: {e}")

        return result

    def get_stats(self) -> Dict:
        """Get pipeline statistics."""
        stats = {
            "yolo_available": self.yolo is not None,
            "weapon_detector_available": self.weapon_detector is not None,
            "gemini_available": self.gemini is not None,
            "tts_available": self.tts is not None,
            "device": self.device
        }

        if self.tracker:
            stats.update(self.tracker.get_stats())

        return stats

    def reset(self):
        """Reset the pipeline state."""
        if self.tracker:
            self.tracker.reset()
        logger.info("DetectionPipeline reset")


# Global singleton
_pipeline: Optional[DetectionPipeline] = None


def get_detection_pipeline(
    yolo_model: str = "yolov8x.pt",
    device: str = "auto"
) -> Optional[DetectionPipeline]:
    """Get or create detection pipeline."""
    global _pipeline

    if _pipeline is None:
        try:
            _pipeline = DetectionPipeline(
                yolo_model=yolo_model,
                device=device
            )
        except Exception as e:
            logger.error(f"Failed to create detection pipeline: {e}")
            return None

    return _pipeline


def reset_detection_pipeline():
    """Reset global detection pipeline."""
    global _pipeline
    if _pipeline:
        _pipeline.reset()
    _pipeline = None
