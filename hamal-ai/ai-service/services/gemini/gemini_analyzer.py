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
import base64
from io import BytesIO
from typing import Dict, Any, Optional, List
from PIL import Image

logger = logging.getLogger(__name__)

# Import image enhancer for quality improvement before Gemini analysis
try:
    from ..frame_selection.image_enhancer import get_image_enhancer, EnhancementLevel
    IMAGE_ENHANCER_AVAILABLE = True
except ImportError:
    IMAGE_ENHANCER_AVAILABLE = False
    logger.warning("Image enhancer not available - using raw images for Gemini")

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
        self._call_count = 0  # Centralized API call counter
        self._image_enhancer = None

        # Initialize image enhancer for better Gemini analysis
        use_enhancement = os.getenv("USE_IMAGE_ENHANCEMENT", "true").lower() in ("true", "1", "yes")
        if use_enhancement and IMAGE_ENHANCER_AVAILABLE:
            try:
                self._image_enhancer = get_image_enhancer()
                # Set to moderate level for good quality without too much processing
                enhancement_level = os.getenv("IMAGE_ENHANCEMENT_LEVEL", "moderate").lower()
                if enhancement_level == "light":
                    self._image_enhancer.set_level(EnhancementLevel.LIGHT)
                elif enhancement_level == "aggressive":
                    self._image_enhancer.set_level(EnhancementLevel.AGGRESSIVE)
                else:
                    self._image_enhancer.set_level(EnhancementLevel.MODERATE)
                logger.info(f"✅ Image enhancement enabled: {enhancement_level}")
            except Exception as e:
                logger.warning(f"Failed to initialize image enhancer: {e}")
                self._image_enhancer = None

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

    def get_call_count(self) -> int:
        """Get the total number of Gemini API calls made"""
        return self._call_count

    def reset_call_count(self) -> None:
        """Reset the API call counter"""
        self._call_count = 0

    async def _generate_content(self, content_parts: list):
        """
        Wrapper for model.generate_content_async that increments the call counter.
        Use this instead of calling generate_content_async directly.
        """
        self._call_count += 1
        return await self.model.generate_content_async(content_parts)

    def _frame_to_pil(self, frame, bbox: Optional[List[float]] = None, margin_percent: float = 0.50, class_name: str = "unknown") -> Image.Image:
        """
        Convert OpenCV frame to PIL Image, optionally cropping to bbox WITH MARGIN.
        Applies image enhancement for better Gemini analysis.

        Args:
            frame: OpenCV BGR numpy array
            bbox: Optional [x1, y1, x2, y2] to crop
            margin_percent: Percentage of bbox size to add as margin (default 50%)
            class_name: Object class for class-specific enhancement (e.g., "car", "person")

        Returns:
            PIL Image in RGB format with the subject fully visible in the center
        """
        if bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)
            h, w = frame.shape[:2]

            # Calculate bbox dimensions
            bbox_w = x2 - x1
            bbox_h = y2 - y1

            # Add margin around the bbox (50% of bbox size on each side for better context)
            margin_x = int(bbox_w * margin_percent)
            margin_y = int(bbox_h * margin_percent)

            # Expand bbox with margin
            x1 = max(0, x1 - margin_x)
            y1 = max(0, y1 - margin_y)
            x2 = min(w, x2 + margin_x)
            y2 = min(h, y2 + margin_y)

            frame = frame[y1:y2, x1:x2]

        # Apply image enhancement for better Gemini analysis
        if self._image_enhancer is not None:
            try:
                frame = self._image_enhancer.enhance(frame, class_name=class_name)
                logger.debug(f"Applied image enhancement for {class_name}")
            except Exception as e:
                logger.warning(f"Image enhancement failed: {e}")

        # Handle color conversion based on frame format
        if len(frame.shape) == 2:
            # Grayscale - convert to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:
            # RGBA/BGRA - convert to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        elif frame.shape[2] == 3:
            # Assume BGR (standard OpenCV) - convert to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            # Unknown format - use as-is
            rgb = frame

        return Image.fromarray(rgb)

    def _get_cutout_base64(self, frame, bbox: List[float], max_size: int = 400, margin_percent: float = 0.40) -> Optional[str]:
        """
        Get a base64 encoded JPEG of the cropped bbox area WITH MARGIN.

        The subject will be fully visible in the center of the image with
        additional context around it.

        IMPORTANT: The frame may already be a crop from AnalysisBuffer.
        If the bbox coordinates are outside the frame bounds, we use the entire frame.

        Args:
            frame: OpenCV BGR numpy array (may be full frame or pre-cropped)
            bbox: Bounding box [x1, y1, x2, y2] (in original frame coordinates)
            max_size: Maximum dimension for the thumbnail (increased to 400 for better quality)
            margin_percent: Percentage of bbox size to add as margin (default 40% for better context)

        Returns:
            Base64 encoded JPEG string or None on error
        """
        try:
            if frame is None or frame.size == 0:
                return None

            x1, y1, x2, y2 = map(int, bbox)
            h, w = frame.shape[:2]

            # Calculate bbox dimensions
            bbox_w = x2 - x1
            bbox_h = y2 - y1

            if bbox_w <= 0 or bbox_h <= 0:
                return None

            # Check if frame is already a crop from AnalysisBuffer
            # If bbox coordinates don't fit frame, or frame is small, use entire frame
            bbox_outside_frame = (x1 >= w or y1 >= h or x2 > w * 1.5 or y2 > h * 1.5)
            frame_is_small_crop = (w < bbox_w or h < bbox_h) or (w < 600 and h < 600 and w < bbox_w * 2)

            if bbox_outside_frame or frame_is_small_crop:
                # Frame is already a crop - use entire frame as cutout
                logger.debug(f"Frame appears pre-cropped ({w}x{h}), using entire frame")
                crop = frame.copy()
            else:
                # Frame is full-size - need to crop
                # Add margin around the bbox (40% of bbox size on each side for better context)
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
                return None

            # Resize if too large (keep aspect ratio)
            crop_h, crop_w = crop.shape[:2]
            if crop_w > max_size or crop_h > max_size:
                scale = max_size / max(crop_w, crop_h)
                new_w, new_h = int(crop_w * scale), int(crop_h * scale)
                crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Encode to JPEG with good quality
            success, buffer = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not success:
                return None

            return base64.b64encode(buffer).decode('utf-8')

        except Exception as e:
            logger.error(f"Failed to create cutout: {e}")
            return None

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

    def _is_valid_plate(self, plate: Optional[str]) -> bool:
        """
        Check if a license plate value is valid (not a placeholder).

        Invalid values include:
        - None, empty string
        - "null" (string)
        - "לא זוהה", "לא נראה בבירור", "לא נראה", etc.
        - Plates that are too short (less than 5 characters)

        Args:
            plate: License plate string to validate

        Returns:
            True if the plate appears to be a valid plate number
        """
        if not plate:
            return False

        # Convert to string and strip whitespace
        plate_str = str(plate).strip()

        # Check for empty or null-like values
        if not plate_str or plate_str.lower() == "null":
            return False

        # List of invalid placeholder values (Hebrew)
        invalid_values = [
            "לא זוהה",
            "לא נראה",
            "לא נראה בבירור",
            "לא ידוע",
            "לא מזוהה",
            "אין",
            "ללא",
            "חסר",
            "לא קריא",
            "לא ניתן לזהות",
        ]

        # Check against invalid values (case-insensitive for Hebrew)
        plate_lower = plate_str.lower()
        for invalid in invalid_values:
            if invalid in plate_lower:
                return False

        # Valid Israeli plates are typically 7-8 characters
        # But allow 5+ to catch partial plates that might still be valid
        if len(plate_str) < 5:
            return False

        # Check that it contains at least some digits (plates have numbers)
        if not any(c.isdigit() for c in plate_str):
            return False

        return True

    async def analyze_vehicle(
        self,
        frame,
        bbox: List[float],
        include_cutout: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze a vehicle in the frame.

        Args:
            frame: OpenCV frame (BGR numpy array)
            bbox: Bounding box [x1, y1, x2, y2]
            include_cutout: Whether to include base64 image cutout

        Returns:
            Dict with: color, model, manufacturer, licensePlate, vehicleType, cutout_image
        """
        if not self.model:
            return {"error": "Gemini not configured"}

        try:
            # Use "vehicle" class for enhanced license plate region processing
            img = self._frame_to_pil(frame, bbox, class_name="vehicle")

            prompt = """אתה מנתח ביטחוני. נתח את הרכב בתמונה בקפידה רבה.
זהה את כל הפרטים הנראים לעין. תן תשובה מלאה ומפורטת.

החזר JSON בלבד (ללא markdown, ללא טקסט לפני או אחרי):
{
    "color": "צבע הרכב בעברית - בחר אחד: לבן/שחור/אפור/כסוף/אדום/כחול/ירוק/צהוב/כתום/חום/בז'/זהב/בורדו/תכלת/סגול",
    "secondaryColor": "צבע משני אם יש (גג, פסים) או null",
    "manufacturer": "יצרן הרכב - נסה לזהות לפי הלוגו, צורת הפנסים, גריל: טויוטה/יונדאי/קיא/מזדה/הונדה/ניסאן/מיצובישי/סוזוקי/פולקסווגן/סקודה/שברולט/פורד/BMW/מרצדס/אאודי/סובארו/וולוו/פיג'ו/סיטרואן/רנו/דאצ'יה/סיאט",
    "model": "דגם הרכב אם ניתן לזהות - נסה לזהות לפי צורת הרכב, או לא זוהה",
    "vehicleType": "סוג הרכב - בחר אחד: רכב פרטי/SUV/ג'יפ/טנדר/משאית/אוטובוס/מיניבוס/אופנוע/קטנוע/אופניים/רכב מסחרי/רכב צבאי/אמבולנס/משטרה",
    "licensePlate": "מספר הרכב אם נראה - רשום את כל הספרות והאותיות שניתן לקרוא, או לא זוהה",
    "licensePlatePartial": true/false,
    "bodyStyle": "סדאן/האצ'בק/סטיישן/קופה/קבריולט/וואן/פיקאפ",
    "approximateYear": "משוער: ישן (לפני 2010)/בינוני (2010-2018)/חדש (2018+)",
    "condition": "מצב הרכב: תקין/פגום/מלוכלך/נקי/משופץ",
    "distinguishingFeatures": ["מאפיינים בולטים: מדבקות, פגיעות, ספוילר, חלונות כהים, גגון, מזוודות"],
    "confidence": 0.85
}

הוראות חשובות:
1. מלא את כל השדות - אל תשאיר שדות ריקים
2. אם לא ניתן לזהות ערך, כתוב "לא זוהה"
3. נסה מאוד לזהות את היצרן - בדוק לוגו, צורת גריל, פנסים
4. נסה לזהות צבע מדויק - לא רק "בהיר" או "כהה"
5. אם רואים חלק מלוח הרישוי, רשום מה שנראה"""

            response = await self._generate_content([prompt, img])
            result = self._parse_json(response.text)

            # Add cutout image if requested
            if include_cutout:
                cutout = self._get_cutout_base64(frame, bbox)
                if cutout:
                    result["cutout_image"] = cutout
                    logger.debug(f"Added cutout image to vehicle analysis ({len(cutout)} bytes)")
                else:
                    logger.warning("Failed to generate cutout image for vehicle")

            # Check if license plate is stolen (Feature 1)
            # Only check if we have a valid plate (not placeholder values)
            license_plate = result.get("licensePlate")
            if self._is_valid_plate(license_plate):
                stolen_info = await self._check_stolen_plate(license_plate)
                result["stolen"] = stolen_info.get("stolen", False)
                if result["stolen"]:
                    logger.warning(f"STOLEN VEHICLE DETECTED: {license_plate}")
            else:
                result["stolen"] = False

            logger.info(f"Vehicle analyzed: {result.get('color', '?')} {result.get('manufacturer', '?')} (stolen: {result.get('stolen', False)})")
            return result

        except Exception as e:
            logger.error(f"Vehicle analysis error: {e}")
            return {"error": str(e)}

    async def _check_stolen_plate(self, plate: str) -> Dict[str, Any]:
        """
        Check if a license plate is in the stolen plates database.
        Feature 1: Stolen Vehicle Detection.

        Args:
            plate: License plate number to check

        Returns:
            Dict with: stolen (bool), plate (str), record (optional)
        """
        import httpx
        backend_url = os.environ.get("BACKEND_URL", "http://localhost:3000")

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{backend_url}/api/stolen-plates/check/{plate}",
                    timeout=5.0
                )
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.debug(f"Stolen plate check failed: HTTP {response.status_code}")
                    return {"stolen": False, "plate": plate}
        except Exception as e:
            logger.debug(f"Stolen plate check error: {e}")
            return {"stolen": False, "plate": plate}

    async def analyze_person(
        self,
        frame,
        bbox: List[float],
        include_cutout: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze a person in the frame - clothing and armed status.

        Args:
            frame: OpenCV frame (BGR numpy array)
            bbox: Bounding box [x1, y1, x2, y2]
            include_cutout: Whether to include base64 image cutout

        Returns:
            Dict with: shirtColor, pantsColor, headwear, armed, weaponType, cutout_image
        """
        if not self.model:
            return {"error": "Gemini not configured"}

        try:
            # Use "person" class for person-specific enhancement
            img = self._frame_to_pil(frame, bbox, class_name="person")

            prompt = """אתה מנתח ביטחוני מקצועי. נתח את האדם בתמונה בקפידה רבה.
תאר את כל מה שאתה רואה בפירוט מלא. זה חשוב לזיהוי.

החזר JSON בלבד (ללא markdown, ללא טקסט לפני או אחרי):
{
    "gender": "זכר/נקבה/לא ניתן לקבוע",
    "approximateAge": "ילד (0-12)/נער (13-19)/צעיר (20-35)/מבוגר (36-55)/קשיש (55+)",
    "shirtColor": "צבע החולצה - בחר: לבן/שחור/אפור/כחול/אדום/ירוק/צהוב/כתום/ורוד/סגול/חום/בז'/חאקי/צבאי/פסים/משובץ",
    "shirtType": "סוג: חולצה/טישרט/חולצה מכופתרת/גופייה/סווטשירט/ז'קט/מעיל/אפודה",
    "pantsColor": "צבע המכנסיים - בחר: שחור/כחול/ג'ינס/אפור/חום/בז'/חאקי/צבאי/לבן",
    "pantsType": "סוג: מכנסיים ארוכים/שורטס/חצאית/שמלה",
    "footwear": "סוג נעליים: נעלי ספורט/סנדלים/מגפיים/נעליים רגילות/יחף/לא נראה",
    "footwearColor": "צבע הנעליים",
    "headwear": "כיסוי ראש: כובע מצחייה/כובע צמר/כיפה/כאפייה/מטפחת/קסדה/ברדס/ללא",
    "headwearColor": "צבע כיסוי הראש",
    "accessories": ["משקפיים", "משקפי שמש", "שעון", "שרשרת", "עגילים", "תיק גב", "תיק צד", "תיק יד"],
    "facialHair": "זקן מלא/זקן קצר/שפם/מגולח/לא נראה",
    "hairColor": "צבע שיער: שחור/חום/בלונד/ג'ינג'י/אפור/לבן/קירח/לא נראה",
    "hairStyle": "קצר/ארוך/קוקו/צמות/קירח/לא נראה",
    "build": "מבנה גוף: רזה/ממוצע/שרירי/מוצק/כבד",
    "posture": "עומד/הולך/רץ/יושב/רוכן/שוכב",
    "faceCovered": false,
    "maskType": "אם הפנים מכוסות: מסכה רפואית/מסכת סקי/כאפייה על הפנים/ברדס/לא מכוסה",
    "carryingItems": ["תיק", "שקית", "חפץ לא מזוהה"],
    "armed": false,
    "weaponType": "אם חמוש: רובה/אקדח/סכין/מקל/לא חמוש",
    "weaponLocation": "אם חמוש: ביד/בחגורה/על הגב/לא רלוונטי",
    "suspiciousIndicators": ["התנהגות חשודה", "ביגוד לא תואם מזג אוויר", "מסתיר פנים"],
    "suspiciousLevel": 1,
    "description": "תיאור מילולי קצר וממוקד לזיהוי: גבר צעיר בחולצה כחולה ומכנסי ג'ינס עם תיק גב שחור"
}

הוראות קריטיות:
1. מלא את כל השדות - זה חשוב לזיהוי!
2. תאר צבעים ספציפיים, לא "בהיר" או "כהה"
3. שדה description חייב לכלול תיאור שימושי לזיהוי
4. armed=true רק אם רואים נשק בבירור
5. אם לא רואים פרט מסוים, כתוב "לא נראה" """

            response = await self._generate_content([prompt, img])
            result = self._parse_json(response.text)

            # Add cutout image if requested
            if include_cutout:
                cutout = self._get_cutout_base64(frame, bbox)
                if cutout:
                    result["cutout_image"] = cutout
                    logger.debug(f"Added cutout image to person analysis ({len(cutout)} bytes)")
                else:
                    logger.warning("Failed to generate cutout image for person")

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

            response = await self._generate_content([prompt, img])
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

            response = await self._generate_content([prompt, img])
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

            response = await self._generate_content([prompt, audio_file])
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

            response = await self._generate_content([prompt, *images])
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

            response = await self._generate_content([prompt, img1, img2])
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

            response = await self._generate_content([prompt, img1, img2])
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
