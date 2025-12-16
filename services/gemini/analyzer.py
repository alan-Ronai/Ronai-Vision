"""Gemini AI service for detailed object analysis.

This module provides Gemini-powered analysis for:
- Car: model, license plate number, color, vehicle type, condition
- Person: clothing, physical features, carried items, armed status, threat level
- Threat neutralization check (body camera analysis)
- Audio transcription (Hebrew)
- Alert text generation for TTS

All prompts and responses are in Hebrew.
"""

import logging
import time
import os
import asyncio
from typing import Optional, Dict, List, Any
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# Try to import google.generativeai (optional dependency)
GEMINI_AVAILABLE = False
genai = None
try:
    import google.generativeai as genai  # noqa: F401

    GEMINI_AVAILABLE = True
except ImportError:
    logger.warning(
        "google-generativeai not installed. Gemini analysis will be disabled."
    )


class GeminiAnalyzer:
    """Gemini AI analyzer for detailed object description."""

    # Hebrew prompts for analysis
    CAR_PROMPT_HE = """
תאר את הרכב בתמונה בפורמט JSON בדיוק כזה:
{
  "דגם": "יצרן ודגם הרכב",
  "מספר_רישוי": "מספר הרישוי אם נראה, אחרת null",
  "צבע": "צבע הרכב",
  "סוג_רכב": "רכב פרטי/משאית/רכב צבאי/ג'יפ/אופנוע/רכב מסחרי",
  "מצב_רכב": "תקין/פגום/חשוד"
}

אם לא ניתן לזהות מידע מסוים, השתמש ב-"לא ידוע". השב רק ב-JSON, ללא טקסט נוסף.
"""

    PERSON_PROMPT_HE = """
תאר את האדם בתמונה בפירוט רב בפורמט JSON בדיוק כזה:
{
  "לבוש": {
    "חולצה": "צבע וסוג",
    "מכנסיים": "צבע וסוג",
    "נעליים": "סוג וצבע",
    "כיסוי_ראש": "כובע/קסדה/כיפה/ללא",
    "אביזרים": ["אפוד", "תיק", "משקפיים"]
  },
  "מאפיינים": {
    "מין_משוער": "זכר/נקבה/לא ברור",
    "גיל_משוער": "טווח גיל",
    "צבע_עור": "גוון",
    "צבע_שיער": "צבע"
  },
  "חמוש": true/false,
  "סוג_נשק": "רובה/אקדח/סכין/לא רלוונטי",
  "פריטים_בידיים": ["רשימת פריטים"] או null,
  "רמת_איום": "גבוהה/בינונית/נמוכה/ללא",
  "תיאור_נוסף": "מידע נוסף רלוונטי"
}

שים לב במיוחד: האם האדם מחזיק נשק או פריט חשוד? האם הוא לבוש בצורה צבאית או חשודה?
השב רק ב-JSON, ללא טקסט נוסף.
"""

    THREAT_CHECK_PROMPT_HE = """
בדוק את התמונה ממצלמת הגוף של הלוחם. החזר JSON:
{
  "אדם_על_הרצפה": true/false,
  "נשק_על_הרצפה": true/false,
  "איום_נוטרל": true/false,
  "מספר_אנשים_בסצנה": מספר,
  "תיאור_סצנה": "תיאור קצר של מה שרואים"
}

האם יש אדם שוכב/מושלך על הרצפה? האם יש נשק מונח ליד?
השב רק ב-JSON.
"""

    AUDIO_TRANSCRIPTION_PROMPT_HE = """
תמלל את האודיו הזה לעברית.
החזר רק את הטקסט המתומלל, ללא הסברים נוספים.
אם יש מילים לא ברורות, נסה לנחש לפי הקונטקסט הצבאי/ביטחוני.
"""

    def __init__(
        self, api_key: Optional[str] = None, model_name: str = "gemini-1.5-flash-latest"
    ):
        """Initialize Gemini analyzer.

        Args:
            api_key: Gemini API key (reads from GEMINI_API_KEY env if not provided)
            model_name: Model to use (default: gemini-1.5-flash-latest)
        """
        if not GEMINI_AVAILABLE or genai is None:
            raise RuntimeError(
                "google-generativeai not installed. Install with: pip install google-generativeai"
            )

        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key not provided. Set GEMINI_API_KEY env var or pass api_key parameter."
            )

        # Configure the API key
        genai.configure(api_key=self.api_key)  # type: ignore

        # Initialize model - try gemini-1.5-flash-latest first, fallback to others
        self.model_name = model_name
        try:
            self.model = genai.GenerativeModel(model_name=model_name)  # type: ignore
            logger.info(f"GeminiAnalyzer initialized with model: {model_name}")
        except Exception as e:
            # Try alternative model names
            logger.warning(f"Failed to load {model_name}: {e}, trying alternatives...")
            for alt_model in ["gemini-1.5-flash", "gemini-pro-vision", "gemini-pro"]:
                try:
                    self.model = genai.GenerativeModel(model_name=alt_model)  # type: ignore
                    self.model_name = alt_model
                    logger.info(
                        f"GeminiAnalyzer initialized with fallback model: {alt_model}"
                    )
                    break
                except Exception as e2:
                    logger.warning(f"Failed to load {alt_model}: {e2}")
                    continue
            else:
                raise RuntimeError(f"Failed to initialize any Gemini model: {e}")

        # Rate limiting tracking
        self._last_request_time = 0.0
        self._min_request_interval = 1.0  # Minimum 1 second between requests

    def analyze_car(self, image: np.ndarray, bbox: Optional[np.ndarray] = None) -> Dict:
        """Analyze a car image and extract details in Hebrew.

        Args:
            image: Full frame (H, W, 3) BGR uint8
            bbox: Optional bounding box [x1, y1, x2, y2] to crop car region

        Returns:
            Dictionary with analysis results in Hebrew:
            {
                "דגם": "car model",
                "מספר_רישוי": "license plate" or None,
                "צבע": "color",
                "סוג_רכב": "vehicle type",
                "מצב_רכב": "condition",
                "timestamp": unix timestamp,
                "error": None or error message
            }
        """
        try:
            # Crop to bbox if provided
            if bbox is not None:
                bbox_arr = np.array(bbox) if not isinstance(bbox, np.ndarray) else bbox
                x1, y1, x2, y2 = bbox_arr.astype(int)
                crop = image[y1:y2, x1:x2]
            else:
                crop = image

            # Enforce rate limiting
            self._rate_limit()

            # Convert to PIL-compatible format
            image_data = self._prepare_image(crop)

            # Call Gemini API
            response = self.model.generate_content([self.CAR_PROMPT_HE, image_data])
            logger.debug(f"Gemini car analysis response: {response}")

            # Parse response
            result = self._parse_json_response(response.text)
            result["timestamp"] = time.time()
            result["error"] = None

            logger.info(f"Car analysis completed: {result}")
            return result

        except Exception as e:
            logger.error(f"Car analysis failed: {e}")
            return {
                "דגם": "לא ידוע",
                "מספר_רישוי": None,
                "צבע": "לא ידוע",
                "סוג_רכב": "לא ידוע",
                "מצב_רכב": "לא ידוע",
                "timestamp": time.time(),
                "error": str(e),
            }

    def analyze_person(
        self, image: np.ndarray, bbox: Optional[np.ndarray] = None
    ) -> Dict:
        """Analyze a person image and extract detailed features in Hebrew.

        Args:
            image: Full frame (H, W, 3) BGR uint8
            bbox: Optional bounding box [x1, y1, x2, y2] to crop person region

        Returns:
            Dictionary with analysis results in Hebrew:
            {
                "לבוש": {
                    "חולצה": "color and type",
                    "מכנסיים": "color and type",
                    "נעליים": "type and color",
                    "כיסוי_ראש": "hat/helmet/none",
                    "אביזרים": [...]
                },
                "מאפיינים": {
                    "מין_משוער": "זכר/נקבה/לא ברור",
                    "גיל_משוער": "age range",
                    "צבע_עור": "skin tone",
                    "צבע_שיער": "hair color"
                },
                "חמוש": true/false,
                "סוג_נשק": "weapon type or לא רלוונטי",
                "פריטים_בידיים": [...] or None,
                "רמת_איום": "גבוהה/בינונית/נמוכה/ללא",
                "תיאור_נוסף": "additional details",
                "timestamp": unix timestamp,
                "error": None or error message
            }
        """
        try:
            # Crop to bbox if provided
            if bbox is not None:
                bbox_arr = np.array(bbox) if not isinstance(bbox, np.ndarray) else bbox
                x1, y1, x2, y2 = bbox_arr.astype(int)
                crop = image[y1:y2, x1:x2]
            else:
                crop = image

            # Enforce rate limiting
            self._rate_limit()

            # Convert to PIL-compatible format
            image_data = self._prepare_image(crop)

            # Call Gemini API
            response = self.model.generate_content([self.PERSON_PROMPT_HE, image_data])
            logger.debug(f"Gemini person analysis response: {response}")

            # Parse response
            result = self._parse_json_response(response.text)
            result["timestamp"] = time.time()
            result["error"] = None

            logger.info(f"Person analysis completed: {result}")
            return result

        except Exception as e:
            logger.error(f"Person analysis failed: {e}")
            return {
                "לבוש": {
                    "חולצה": "לא ידוע",
                    "מכנסיים": "לא ידוע",
                    "נעליים": "לא ידוע",
                    "כיסוי_ראש": "לא ידוע",
                    "אביזרים": []
                },
                "מאפיינים": {
                    "מין_משוער": "לא ברור",
                    "גיל_משוער": "לא ידוע",
                    "צבע_עור": "לא ידוע",
                    "צבע_שיער": "לא ידוע"
                },
                "חמוש": False,
                "סוג_נשק": "לא רלוונטי",
                "פריטים_בידיים": None,
                "רמת_איום": "לא ידוע",
                "תיאור_נוסף": "",
                "timestamp": time.time(),
                "error": str(e),
            }

    def _prepare_image(self, image: np.ndarray) -> Dict:
        """Prepare image for Gemini API.

        Args:
            image: (H, W, 3) BGR uint8 numpy array

        Returns:
            Dictionary compatible with Gemini API
        """
        # Convert BGR to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Encode as JPEG
        success, buffer = cv2.imencode(".jpg", rgb)
        if not success:
            raise ValueError("Failed to encode image as JPEG")

        # Convert to base64
        jpg_bytes = buffer.tobytes()

        # Return in format expected by Gemini
        return {"mime_type": "image/jpeg", "data": jpg_bytes}

    def _parse_json_response(self, response_text: str) -> Dict:
        """Parse JSON response from Gemini.

        Args:
            response_text: Response text from Gemini

        Returns:
            Parsed dictionary
        """
        import json
        import re

        # Remove markdown code blocks if present
        text = response_text.strip()
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*", "", text)
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {text}")
            raise ValueError(f"Invalid JSON response from Gemini: {e}")

    def _rate_limit(self):
        """Enforce rate limiting between API calls."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            sleep_time = self._min_request_interval - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)

        self._last_request_time = time.time()

    # =====================================================================
    # NEW METHODS - Threat Analysis, Audio, Alert Generation
    # =====================================================================

    def analyze_threat_neutralized(self, image: np.ndarray) -> Dict:
        """Check if threat is neutralized (person on ground, weapon visible).

        Used for body camera analysis at end of incident.

        Args:
            image: Full frame from body camera (H, W, 3) BGR uint8

        Returns:
            Dictionary with threat status:
            {
                "אדם_על_הרצפה": true/false,
                "נשק_על_הרצפה": true/false,
                "איום_נוטרל": true/false,
                "מספר_אנשים_בסצנה": number,
                "תיאור_סצנה": "scene description",
                "timestamp": unix timestamp,
                "error": None or error message
            }
        """
        try:
            self._rate_limit()
            image_data = self._prepare_image(image)

            response = self.model.generate_content([self.THREAT_CHECK_PROMPT_HE, image_data])
            result = self._parse_json_response(response.text)
            result["timestamp"] = time.time()
            result["error"] = None

            logger.info(f"Threat check completed: {result}")
            return result

        except Exception as e:
            logger.error(f"Threat check failed: {e}")
            return {
                "אדם_על_הרצפה": False,
                "נשק_על_הרצפה": False,
                "איום_נוטרל": False,
                "מספר_אנשים_בסצנה": 0,
                "תיאור_סצנה": "שגיאה בניתוח",
                "timestamp": time.time(),
                "error": str(e)
            }

    def transcribe_audio(self, audio_path: str) -> Dict:
        """Transcribe Hebrew audio using Gemini.

        Args:
            audio_path: Path to audio file (WAV/MP3)

        Returns:
            Dictionary with transcription:
            {
                "תמלול": "transcribed text",
                "timestamp": unix timestamp,
                "error": None or error message
            }
        """
        try:
            self._rate_limit()

            # Upload audio file to Gemini
            audio_file = genai.upload_file(audio_path)

            response = self.model.generate_content([self.AUDIO_TRANSCRIPTION_PROMPT_HE, audio_file])

            return {
                "תמלול": response.text.strip(),
                "timestamp": time.time(),
                "error": None
            }

        except Exception as e:
            logger.error(f"Audio transcription failed: {e}")
            return {
                "תמלול": "",
                "timestamp": time.time(),
                "error": str(e)
            }

    def generate_alert_text(self, detection_data: Dict) -> str:
        """Generate Hebrew alert text for TTS from detection data.

        Args:
            detection_data: Dictionary with detection info:
                - camera_id: Camera identifier
                - person_count: Number of suspects
                - armed: Whether suspects are armed
                - weapon_type: Type of weapon
                - vehicle: Vehicle info dict (צבע, דגם, מספר_רישוי)

        Returns:
            Hebrew text ready for TTS
        """
        parts = ["חדירה ודאית. אירוע אמת."]

        # Location
        if detection_data.get("camera_id"):
            parts.append(f"מצלמה {detection_data['camera_id']}.")

        # People count
        person_count = detection_data.get("person_count", 0)
        if person_count > 0:
            parts.append(f"מספר חשודים: {person_count}.")

        # Armed status
        if detection_data.get("armed"):
            weapon = detection_data.get("weapon_type", "לא ידוע")
            parts.append(f"החשודים חמושים. סוג נשק: {weapon}.")

        # Vehicle info
        vehicle = detection_data.get("vehicle")
        if vehicle:
            color = vehicle.get("צבע", "")
            model = vehicle.get("דגם", "")
            plate = vehicle.get("מספר_רישוי")

            if color or model:
                parts.append(f"רכב: {color} {model}.".strip())
            if plate:
                # Read plate number digit by digit for clarity
                plate_spoken = " ".join(str(plate).replace("-", " "))
                parts.append(f"מספר רכב: {plate_spoken}.")

        parts.append("קו דיווח 13.")

        return " ".join(parts)

    def is_person_armed(self, analysis_result: Dict) -> bool:
        """Check if person analysis indicates armed status.

        Args:
            analysis_result: Result from analyze_person()

        Returns:
            True if person appears armed
        """
        # Check explicit armed field
        if analysis_result.get("חמוש") == True:
            return True

        # Check weapon type
        weapon = str(analysis_result.get("סוג_נשק", "")).lower()
        if weapon and weapon not in ["לא רלוונטי", "אין", "null", "", "none"]:
            return True

        # Check threat level
        threat = str(analysis_result.get("רמת_איום", "")).lower()
        if threat in ["גבוהה", "high"]:
            return True

        # Check items in hands (legacy field)
        items = analysis_result.get("פריטים_בידיים") or []
        if isinstance(items, list):
            weapon_keywords = ["נשק", "רובה", "אקדח", "סכין", "רימון", "gun", "knife", "weapon", "rifle", "pistol"]
            for item in items:
                if any(kw in str(item).lower() for kw in weapon_keywords):
                    return True

        return False

    # =====================================================================
    # ASYNC VERSIONS - For better performance in async contexts
    # =====================================================================

    async def analyze_car_async(self, image: np.ndarray, bbox: Optional[np.ndarray] = None) -> Dict:
        """Async version of analyze_car."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.analyze_car, image, bbox)

    async def analyze_person_async(self, image: np.ndarray, bbox: Optional[np.ndarray] = None) -> Dict:
        """Async version of analyze_person."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.analyze_person, image, bbox)

    async def analyze_threat_neutralized_async(self, image: np.ndarray) -> Dict:
        """Async version of analyze_threat_neutralized."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.analyze_threat_neutralized, image)

    async def transcribe_audio_async(self, audio_path: str) -> Dict:
        """Async version of transcribe_audio."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.transcribe_audio, audio_path)


# Global singleton instance
_gemini_analyzer: Optional[GeminiAnalyzer] = None


def get_gemini_analyzer(
    api_key: Optional[str] = None, model_name: str = "gemini-1.5-flash"
) -> Optional[GeminiAnalyzer]:
    """Get or create global Gemini analyzer instance.

    Args:
        api_key: Optional API key (only used on first call)
        model_name: Model name (only used on first call)

    Returns:
        GeminiAnalyzer instance or None if not available
    """
    global _gemini_analyzer

    if not GEMINI_AVAILABLE:
        return None

    if _gemini_analyzer is None:
        try:
            _gemini_analyzer = GeminiAnalyzer(api_key=api_key, model_name=model_name)
        except Exception as e:
            logger.error(f"Failed to initialize Gemini analyzer: {e}")
            return None

    return _gemini_analyzer


def reset_gemini_analyzer():
    """Reset global Gemini analyzer (for testing)."""
    global _gemini_analyzer
    _gemini_analyzer = None
