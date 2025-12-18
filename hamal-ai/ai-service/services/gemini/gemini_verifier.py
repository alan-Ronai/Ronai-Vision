"""
Gemini Vision Verification Service

Uses Google's Gemini model for advanced image analysis:
- Vehicle verification
- People clothing analysis
- Scene analysis
"""

import os
import json
import cv2
from io import BytesIO
from PIL import Image

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️ google-generativeai not installed - Gemini features disabled")


class GeminiVerifier:
    def __init__(self):
        self.model = None

        if not GEMINI_AVAILABLE:
            print("⚠️ Gemini not available - install google-generativeai")
            return

        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            try:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel("gemini-1.5-flash")
                print("✅ Gemini configured successfully")
            except Exception as e:
                print(f"⚠️ Gemini configuration error: {e}")
        else:
            print("⚠️ GEMINI_API_KEY not set - verification disabled")

    def is_configured(self):
        """Check if Gemini is properly configured"""
        return self.model is not None

    async def verify_vehicle(self, frames):
        """
        Analyze vehicle from one or more frames

        Args:
            frames: List of OpenCV frames (BGR numpy arrays)

        Returns:
            dict with vehicle analysis
        """
        if not self.model:
            return {"error": "Gemini not configured"}

        try:
            images = []
            for frame in frames:
                # Convert BGR to RGB and to PIL Image
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                images.append(img)

            prompt = """
            בדוק את התמונות וזהה את הרכב הראשי בתמונה. החזר JSON בלבד:
            {
                "vehicleType": "סוג הרכב בעברית (מכונית/משאית/אופנוע וכו')",
                "color": "צבע הרכב בעברית",
                "licensePlate": "מספר רישוי אם נראה, אחרת null",
                "model": "דגם/יצרן אם ניתן לזהות, אחרת null",
                "condition": "מצב הרכב (תקין/פגום/מוסתר)",
                "confidence": 0.0-1.0,
                "notes": "הערות נוספות אם יש"
            }
            חשוב: החזר רק JSON תקין, בלי טקסט נוסף.
            """

            response = await self.model.generate_content_async([prompt, *images])
            return self._parse_json(response.text)

        except Exception as e:
            return {"error": str(e)}

    async def analyze_people(self, frame):
        """
        Analyze people in frame - clothing, positioning, weapons

        Args:
            frame: OpenCV frame (BGR numpy array)

        Returns:
            dict with people analysis
        """
        if not self.model:
            return {"error": "Gemini not configured"}

        try:
            # Convert to PIL Image
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)

            prompt = """
            נתח את האנשים בתמונה בזהירות. החזר JSON בלבד:
            {
                "count": מספר_אנשים,
                "people": [
                    {
                        "clothing": ["פריט ביגוד 1", "פריט ביגוד 2"],
                        "clothingColors": ["צבע 1", "צבע 2"],
                        "armed": true/false,
                        "weaponType": "סוג נשק אם נראה, אחרת null",
                        "position": "שמאל/מרכז/ימין",
                        "posture": "עומד/יושב/רוכן/רץ",
                        "faceCovered": true/false,
                        "suspiciousLevel": 1-5
                    }
                ],
                "overallThreatLevel": 1-5,
                "notes": "הערות חשובות"
            }
            חשוב: החזר רק JSON תקין, בלי טקסט נוסף.
            """

            response = await self.model.generate_content_async([prompt, img])
            return self._parse_json(response.text)

        except Exception as e:
            return {"error": str(e)}

    async def analyze_scene(self, frame, custom_prompt=None):
        """
        General scene analysis

        Args:
            frame: OpenCV frame (BGR numpy array)
            custom_prompt: Optional custom analysis prompt

        Returns:
            dict with scene analysis
        """
        if not self.model:
            return {"error": "Gemini not configured"}

        try:
            # Convert to PIL Image
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)

            if custom_prompt:
                prompt = f"{custom_prompt}\nהחזר תשובה בעברית בפורמט JSON אם אפשר."
            else:
                prompt = """
                תאר את הסצנה בתמונה מנקודת מבט ביטחונית. החזר JSON:
                {
                    "description": "תיאור כללי של הסצנה",
                    "location": "סוג המיקום (רחוב/חניון/מבנה וכו')",
                    "timeOfDay": "יום/לילה/שקיעה וכו'",
                    "peopleCount": מספר,
                    "vehiclesCount": מספר,
                    "anomalies": ["חריגה 1", "חריגה 2"],
                    "securityConcerns": ["דאגה 1", "דאגה 2"],
                    "threatLevel": 1-5
                }
                """

            response = await self.model.generate_content_async([prompt, img])
            return self._parse_json(response.text)

        except Exception as e:
            return {"error": str(e)}

    def _parse_json(self, text):
        """Parse JSON from model response, handling markdown code blocks"""
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
        except json.JSONDecodeError:
            # Return raw text if JSON parsing fails
            return {"raw_response": text}
