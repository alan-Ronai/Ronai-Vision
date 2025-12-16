"""
Unified Gemini Analyzer Service

Uses Google's Gemini model for ALL analysis tasks:
- Vehicle analysis (color, model, manufacturer, license plate)
- Person analysis (clothing, armed status, weapon type)
- Threat assessment (is person on ground / neutralized)
- Audio transcription (Hebrew radio communications)
- Scene analysis (general security assessment)

This replaces multiple specialized models with a single unified approach.
"""

import os
import json
import cv2
import asyncio
import logging
from io import BytesIO
from typing import Dict, Any, Optional, List
from PIL import Image

logger = logging.getLogger(__name__)

# Try to import Google Generative AI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("google-generativeai not installed - Gemini features disabled")


class GeminiAnalyzer:
    """
    Unified analyzer using Google Gemini for all visual and audio analysis.

    Gemini handles:
    - Vehicle identification (color, model, plate)
    - Person analysis (clothing, weapons)
    - Threat neutralization detection
    - Audio transcription
    - General scene analysis
    """

    def __init__(self):
        self.model = None
        self.vision_model = None

        if not GEMINI_AVAILABLE:
            logger.warning("⚠️ Gemini not available - install google-generativeai")
            return

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.warning("⚠️ GEMINI_API_KEY not set - Gemini features disabled")
            return

        try:
            genai.configure(api_key=api_key)

            # Use gemini-2.0-flash-exp for speed and latest capabilities
            self.model = genai.GenerativeModel("gemini-2.0-flash-exp")
            self.vision_model = genai.GenerativeModel("gemini-2.0-flash-exp")

            logger.info("✅ Gemini analyzer initialized (gemini-2.0-flash-exp)")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            self.model = None

    def is_configured(self) -> bool:
        """Check if Gemini is properly configured"""
        return self.model is not None

    def _frame_to_pil(self, frame, bbox: Optional[List[float]] = None) -> Image.Image:
        """
        Convert OpenCV frame to PIL Image, optionally cropping to bbox.

        Args:
            frame: OpenCV BGR numpy array
            bbox: Optional [x1, y1, x2, y2] to crop

        Returns:
            PIL Image in RGB format
        """
        if bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)
            # Clamp to frame bounds
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            frame = frame[y1:y2, x1:x2]

        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Parse JSON from Gemini response, handling markdown code blocks"""
        try:
            # Remove markdown code blocks if present
            clean = text.strip()
            if clean.startswith("```json"):
                clean = clean[7:]
            elif clean.startswith("```"):
                clean = clean[3:]
            if clean.endswith("```"):
                clean = clean[:-3]
            clean = clean.strip()

            return json.loads(clean)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON: {e}")
            return {"raw_response": text, "parse_error": str(e)}

    async def analyze_vehicle(
        self,
        frame,
        bbox: List[float]
    ) -> Dict[str, Any]:
        """
        Analyze a vehicle in the frame.

        Args:
            frame: OpenCV frame (BGR numpy array)
            bbox: Bounding box [x1, y1, x2, y2]

        Returns:
            Dict with: color, model, manufacturer, licensePlate, vehicleType
        """
        if not self.model:
            return {"error": "Gemini not configured"}

        try:
            img = self._frame_to_pil(frame, bbox)

            prompt = """
            נתח את הרכב בתמונה בקפידה. החזר JSON בלבד (ללא markdown):
            {
                "color": "צבע הרכב בעברית (אדום/כחול/לבן/שחור/כסוף/אפור וכו')",
                "model": "דגם הרכב אם ניתן לזהות, אחרת null",
                "manufacturer": "יצרן הרכב (טויוטה/יונדאי/מזדה וכו'), אחרת null",
                "licensePlate": "מספר הרכב אם נראה בבירור, אחרת null",
                "vehicleType": "סוג: רכב פרטי/משאית/טנדר/אוטובוס/אופנוע/רכב צבאי/טרקטורון",
                "condition": "תקין/פגום/מוסווה/חשוד",
                "confidence": 0.0-1.0
            }
            חשוב: החזר רק JSON תקין, בלי טקסט נוסף לפני או אחרי.
            """

            response = await self.model.generate_content_async([prompt, img])
            result = self._parse_json(response.text)

            logger.info(f"Vehicle analyzed: {result.get('color', '?')} {result.get('manufacturer', '?')}")
            return result

        except Exception as e:
            logger.error(f"Vehicle analysis error: {e}")
            return {"error": str(e)}

    async def analyze_person(
        self,
        frame,
        bbox: List[float]
    ) -> Dict[str, Any]:
        """
        Analyze a person in the frame - clothing and armed status.

        Args:
            frame: OpenCV frame (BGR numpy array)
            bbox: Bounding box [x1, y1, x2, y2]

        Returns:
            Dict with: shirtColor, pantsColor, headwear, armed, weaponType
        """
        if not self.model:
            return {"error": "Gemini not configured"}

        try:
            img = self._frame_to_pil(frame, bbox)

            prompt = """
            נתח את האדם בתמונה בזהירות מנקודת מבט ביטחונית.
            החזר JSON בלבד (ללא markdown):
            {
                "shirtColor": "צבע החולצה/הגופייה בעברית",
                "pantsColor": "צבע המכנסיים בעברית",
                "headwear": "כובע/קסדה/כיפה/כאפייה/מסכה/ללא",
                "additionalClothing": ["אפוד", "מעיל", "תיק גב"],
                "faceCovered": true/false,
                "armed": true/false,
                "weaponType": "סוג הנשק אם חמוש: רובה/אקדח/סכין/מטען חבלה, אחרת null",
                "weaponVisible": true/false,
                "posture": "עומד/יושב/רוכן/רץ/שוכב",
                "suspiciousLevel": 1-5,
                "description": "תיאור קצר של המראה"
            }
            חשוב: החזר רק JSON תקין, בלי טקסט נוסף.
            """

            response = await self.model.generate_content_async([prompt, img])
            result = self._parse_json(response.text)

            if result.get("armed"):
                logger.warning(f"⚠️ Armed person detected! Weapon: {result.get('weaponType')}")

            return result

        except Exception as e:
            logger.error(f"Person analysis error: {e}")
            return {"error": str(e)}

    async def analyze_threat_neutralized(
        self,
        frame
    ) -> Dict[str, Any]:
        """
        Check if threat is neutralized (person lying on ground).
        Used for body camera analysis at end of incident.

        Args:
            frame: OpenCV frame (BGR numpy array)

        Returns:
            Dict with: personOnGround, weaponVisible, threatNeutralized
        """
        if not self.model:
            return {"error": "Gemini not configured"}

        try:
            img = self._frame_to_pil(frame)

            prompt = """
            בדוק את התמונה מנקודת מבט ביטחונית.
            האם יש אדם שוכב על הרצפה/קרקע?
            החזר JSON בלבד:
            {
                "personOnGround": true/false,
                "personCount": מספר_אנשים_שוכבים,
                "weaponVisible": true/false,
                "weaponSecured": true/false,
                "handsVisible": true/false,
                "threatNeutralized": true/false,
                "securityPersonnelPresent": true/false,
                "description": "תיאור קצר של הסצנה"
            }
            """

            response = await self.model.generate_content_async([prompt, img])
            result = self._parse_json(response.text)

            if result.get("threatNeutralized"):
                logger.info("✅ Threat neutralized detected")

            return result

        except Exception as e:
            logger.error(f"Threat neutralization check error: {e}")
            return {"error": str(e)}

    async def analyze_scene(
        self,
        frame,
        custom_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        General scene analysis for security assessment.

        Args:
            frame: OpenCV frame (BGR numpy array)
            custom_prompt: Optional custom analysis prompt

        Returns:
            Dict with scene analysis results
        """
        if not self.model:
            return {"error": "Gemini not configured"}

        try:
            img = self._frame_to_pil(frame)

            if custom_prompt:
                prompt = f"{custom_prompt}\nהחזר תשובה בעברית בפורמט JSON אם אפשר."
            else:
                prompt = """
                תאר את הסצנה בתמונה מנקודת מבט ביטחונית. החזר JSON:
                {
                    "description": "תיאור כללי של הסצנה",
                    "locationType": "סוג המיקום: רחוב/חניון/מבנה/שטח פתוח/גדר/שער",
                    "timeOfDay": "יום/לילה/שקיעה/זריחה",
                    "visibility": "טובה/בינונית/גרועה",
                    "peopleCount": מספר,
                    "vehiclesCount": מספר,
                    "crowdDensity": "ריק/דליל/בינוני/צפוף",
                    "anomalies": ["חריגה 1", "חריגה 2"],
                    "securityConcerns": ["דאגה 1", "דאגה 2"],
                    "threatLevel": 1-5,
                    "recommendedAction": "המלצה לפעולה"
                }
                """

            response = await self.model.generate_content_async([prompt, img])
            return self._parse_json(response.text)

        except Exception as e:
            logger.error(f"Scene analysis error: {e}")
            return {"error": str(e)}

    async def transcribe_audio(
        self,
        audio_path: str
    ) -> str:
        """
        Transcribe Hebrew audio using Gemini.

        Args:
            audio_path: Path to audio file

        Returns:
            Transcribed text in Hebrew
        """
        if not self.model:
            return ""

        try:
            # Upload audio file to Gemini
            audio_file = genai.upload_file(audio_path)

            prompt = """
            תמלל את האודיו הזה לעברית.
            החזר רק את הטקסט המתומלל, בלי הערות או תוספות.
            אם יש מילים לא ברורות, נסה לנחש לפי הקונטקסט הצבאי/ביטחוני.
            """

            response = await self.model.generate_content_async([prompt, audio_file])
            transcript = response.text.strip()

            logger.info(f"Audio transcribed: {transcript[:50]}...")
            return transcript

        except Exception as e:
            logger.error(f"Audio transcription error: {e}")
            return ""

    async def analyze_multiple_frames(
        self,
        frames: List,
        prompt: str
    ) -> Dict[str, Any]:
        """
        Analyze multiple frames together (e.g., for tracking verification).

        Args:
            frames: List of OpenCV frames
            prompt: Analysis prompt

        Returns:
            Analysis results
        """
        if not self.model:
            return {"error": "Gemini not configured"}

        try:
            images = [self._frame_to_pil(f) for f in frames]

            response = await self.model.generate_content_async([prompt, *images])
            return self._parse_json(response.text)

        except Exception as e:
            logger.error(f"Multi-frame analysis error: {e}")
            return {"error": str(e)}

    async def verify_same_vehicle(
        self,
        frame1,
        bbox1: List[float],
        frame2,
        bbox2: List[float]
    ) -> Dict[str, Any]:
        """
        Verify if two vehicle detections are the same vehicle.
        Useful for cross-camera tracking.

        Returns:
            Dict with: sameVehicle, confidence, reasoning
        """
        if not self.model:
            return {"error": "Gemini not configured"}

        try:
            img1 = self._frame_to_pil(frame1, bbox1)
            img2 = self._frame_to_pil(frame2, bbox2)

            prompt = """
            בדוק את שתי התמונות. האם זה אותו רכב?
            החזר JSON:
            {
                "sameVehicle": true/false,
                "confidence": 0.0-1.0,
                "matchingFeatures": ["צבע", "דגם", "מספר רכב"],
                "differingFeatures": ["זווית צילום"],
                "reasoning": "הסבר קצר"
            }
            """

            response = await self.model.generate_content_async([prompt, img1, img2])
            return self._parse_json(response.text)

        except Exception as e:
            logger.error(f"Vehicle verification error: {e}")
            return {"error": str(e)}

    async def verify_same_person(
        self,
        frame1,
        bbox1: List[float],
        frame2,
        bbox2: List[float]
    ) -> Dict[str, Any]:
        """
        Verify if two person detections are the same person.
        Based on clothing, not face recognition.

        Returns:
            Dict with: samePerson, confidence, reasoning
        """
        if not self.model:
            return {"error": "Gemini not configured"}

        try:
            img1 = self._frame_to_pil(frame1, bbox1)
            img2 = self._frame_to_pil(frame2, bbox2)

            prompt = """
            בדוק את שתי התמונות. האם זה אותו אדם? (לפי ביגוד, לא פנים)
            החזר JSON:
            {
                "samePerson": true/false,
                "confidence": 0.0-1.0,
                "matchingFeatures": ["צבע חולצה", "מכנסיים"],
                "differingFeatures": [],
                "reasoning": "הסבר קצר"
            }
            """

            response = await self.model.generate_content_async([prompt, img1, img2])
            return self._parse_json(response.text)

        except Exception as e:
            logger.error(f"Person verification error: {e}")
            return {"error": str(e)}

    async def generate_emergency_summary(
        self,
        vehicle_data: Optional[Dict] = None,
        person_data: Optional[List[Dict]] = None,
        location: str = "לא ידוע"
    ) -> str:
        """
        Generate a Hebrew text summary for emergency announcement.

        Args:
            vehicle_data: Vehicle analysis from Gemini
            person_data: List of person analyses from Gemini
            location: Location description

        Returns:
            Hebrew text for TTS announcement
        """
        parts = ["חדירה ודאית. אירוע אמת."]

        if location:
            parts.append(f"מיקום: {location}.")

        if person_data:
            armed_count = sum(1 for p in person_data if p.get("armed"))
            total_count = len(person_data)

            parts.append(f"מספר מחבלים: {total_count}.")

            if armed_count > 0:
                weapons = [p.get("weaponType") for p in person_data if p.get("weaponType")]
                if weapons:
                    parts.append(f"המחבלים חמושים. נשק: {', '.join(set(weapons))}.")
                else:
                    parts.append("המחבלים חמושים.")

                # Clothing description for identification
                for i, person in enumerate(person_data[:3], 1):  # Max 3 descriptions
                    shirt = person.get("shirtColor", "")
                    pants = person.get("pantsColor", "")
                    if shirt or pants:
                        parts.append(f"מחבל {i}: חולצה {shirt}, מכנס {pants}.")

        if vehicle_data and not vehicle_data.get("error"):
            color = vehicle_data.get("color", "")
            manufacturer = vehicle_data.get("manufacturer", "")
            model = vehicle_data.get("model", "")
            plate = vehicle_data.get("licensePlate")

            vehicle_desc = f"{color} {manufacturer} {model}".strip()
            if vehicle_desc:
                parts.append(f"רכב: {vehicle_desc}.")
            if plate:
                parts.append(f"מספר רכב: {plate}.")

        parts.append("קו דיווח 13.")

        return " ".join(parts)
