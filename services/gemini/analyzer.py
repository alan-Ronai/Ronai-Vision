"""Gemini AI service for detailed object analysis.

This module provides Gemini-powered analysis for:
- Car: model, license plate number, color
- Person: clothing, physical features, carried items

All prompts and responses are in Hebrew.
"""

import logging
import time
import os
from typing import Optional, Dict
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
  "צבע": "צבע הרכב"
}

אם לא ניתן לזהות מידע מסוים, השתמש ב-"לא ידוע". השב רק ב-JSON, ללא טקסט נוסף.
"""

    PERSON_PROMPT_HE = """
תאר את האדם בתמונה בפירוט רב בפורמט JSON בדיוק כזה:
{
  "לבוש": "תיאור מפורט של הבגדים - סוג, צבע, סגנון",
  "צבע_עור": "גוון העור",
  "צבע_שיער": "צבע השיער",
  "מין_משוער": "זכר/נקבה/לא ברור",
  "גיל_משוער": "טווח גיל משוער",
  "פריטים_בידיים": "רשימת פריטים שהאדם מחזיק, אחרת null",
  "תיאור_נוסף": "כל מידע נוסף רלוונטי - תסרוקת, אביזרים, תכונות בולטות"
}

תאר בפירוט רב ככל האפשר. השב רק ב-JSON, ללא טקסט נוסף.
"""

    def __init__(
        self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"
    ):
        """Initialize Gemini analyzer.

        Args:
            api_key: Gemini API key (reads from GEMINI_API_KEY env if not provided)
            model_name: Model to use (default: gemini-2.5-flash)
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

        # Initialize model - try models that support vision
        # Updated model names as of Dec 2024 - Gemini 1.5 has been replaced with 2.x
        self.model_name = model_name

        # Try models in order of preference (all support vision)
        model_candidates = [
            "gemini-2.5-flash",  # Current stable vision model (Dec 2024)
            "gemini-flash-latest",  # Alias to latest flash model
            "gemini-2.5-pro",  # More capable vision model
            "gemini-pro-latest",  # Alias to latest pro model
        ]

        # If user specified a model, try it first
        if model_name not in model_candidates:
            model_candidates.insert(0, model_name)

        last_error = None
        for model_to_try in model_candidates:
            try:
                self.model = genai.GenerativeModel(  # type: ignore
                    model_name=model_to_try,
                    generation_config={  # type: ignore
                        "temperature": 0.4,
                        "top_p": 1,
                        "top_k": 32,
                        "max_output_tokens": 2048,
                    },
                )
                self.model_name = model_to_try
                logger.info(f"GeminiAnalyzer initialized with model: {model_to_try}")
                break
            except Exception as e:
                logger.warning(f"Failed to load {model_to_try}: {e}")
                last_error = e
                continue
        else:
            raise RuntimeError(
                f"Failed to initialize any Gemini model. Last error: {last_error}"
            )

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
                "timestamp": unix timestamp,
                "error": None or error message
            }
        """
        try:
            # Crop to bbox if provided
            if bbox is not None:
                x1, y1, x2, y2 = bbox.astype(int)
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
                "לבוש": "clothing description",
                "צבע_עור": "skin tone",
                "צבע_שיער": "hair color",
                "מין_משוער": "זכר/נקבה/לא ברור",
                "גיל_משוער": "age range",
                "פריטים_בידיים": [...] or None,
                "תיאור_נוסף": "additional details",
                "timestamp": unix timestamp,
                "error": None or error message
            }
        """
        try:
            # Crop to bbox if provided
            if bbox is not None:
                x1, y1, x2, y2 = bbox.astype(int)
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
                "לבוש": "לא ידוע",
                "צבע_עור": "לא ידוע",
                "צבע_שיער": "לא ידוע",
                "מין_משוער": "לא ברור",
                "גיל_משוער": "לא ידוע",
                "פריטים_בידיים": None,
                "תיאור_נוסף": "",
                "timestamp": time.time(),
                "error": str(e),
            }

    def _prepare_image(self, image: np.ndarray):
        """Prepare image for Gemini API.

        Args:
            image: (H, W, 3) BGR uint8 numpy array

        Returns:
            PIL Image object that Gemini SDK can process
        """
        from PIL import Image

        # Convert BGR to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image (this is what the Python SDK expects)
        pil_image = Image.fromarray(rgb)

        return pil_image

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


# Global singleton instance
_gemini_analyzer: Optional[GeminiAnalyzer] = None


def get_gemini_analyzer(
    api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"
) -> Optional[GeminiAnalyzer]:
    """Get or create global Gemini analyzer instance.

    Args:
        api_key: Optional API key (only used on first call)
        model_name: Model name (only used on first call, default: gemini-2.5-flash)

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
