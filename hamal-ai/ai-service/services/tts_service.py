"""
Text-to-Speech Service for Hebrew Announcements

Uses Gemini 2.5 Pro TTS with Sulafat voice for high-quality Hebrew speech.
Configured for urgent emergency announcements בלשון זכר.
"""

import os
import asyncio
import logging
import uuid
import wave
import struct
import base64
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Gemini TTS Configuration
# Sulafat - Hebrew female voice with clear pronunciation
GEMINI_TTS_VOICE = os.environ.get("GEMINI_TTS_VOICE", "Sulafat")
GEMINI_TTS_SAMPLE_RATE = int(os.environ.get("GEMINI_TTS_SAMPLE_RATE", "24000"))

# Try to import Google GenAI SDK
GENAI_AVAILABLE = False
genai_client = None

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    logger.warning("google-genai package not installed. Install with: pip install google-genai")


class TTSService:
    """
    Text-to-Speech service for Hebrew emergency announcements.

    Uses Gemini 2.5 Pro TTS with Sulafat voice for high-quality Hebrew speech.
    Configured for urgent announcements בלשון זכר (male grammar).
    """

    def __init__(self):
        self.output_dir = Path(os.getenv("TTS_OUTPUT_DIR", "audio_output"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.voice = GEMINI_TTS_VOICE
        self.sample_rate = GEMINI_TTS_SAMPLE_RATE
        self.api_key = os.environ.get("GEMINI_API_KEY")
        # Use Gemini 2.5 Pro TTS model
        self.model_id = "gemini-2.5-pro-preview-tts"

        self.client = None
        self._init_client()

    def _init_client(self):
        """Initialize the Gemini client"""
        if not GENAI_AVAILABLE:
            logger.error("❌ TTS: google-genai package not installed")
            return

        if not self.api_key:
            logger.warning("⚠️ TTS: GEMINI_API_KEY not set, TTS will be disabled")
            return

        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
            logger.info(f"✅ TTS: Initialized with Gemini 2.5 Pro TTS, voice: {self.voice}")
        except Exception as e:
            logger.error(f"❌ TTS: Failed to initialize client: {e}")

    def is_configured(self) -> bool:
        """Check if TTS is properly configured"""
        return self.client is not None

    async def generate(
        self,
        text: str,
        filename: Optional[str] = None,
        urgent: bool = True
    ) -> str:
        """
        Generate speech from Hebrew text using Gemini TTS.

        Args:
            text: Hebrew text to convert to speech
            filename: Optional output filename
            urgent: Whether to use urgent tone (default True for emergency)

        Returns:
            Path to generated audio file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            uid = uuid.uuid4().hex[:8]
            filename = f"tts_{timestamp}_{uid}.wav"

        output_path = self.output_dir / filename

        try:
            if self.client:
                await self._generate_gemini_tts(text, output_path, urgent)
            else:
                # Generate silent placeholder if TTS not configured
                self._generate_silent_placeholder(text, output_path)

            logger.info(f"TTS generated: {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"TTS generation error: {e}")
            import traceback
            traceback.print_exc()
            # Generate silent placeholder on error
            self._generate_silent_placeholder(text, output_path)
            return str(output_path)

    async def _generate_gemini_tts(self, text: str, output_path: Path, urgent: bool = True):
        """Generate speech using Gemini 2.5 Pro TTS"""
        from google.genai import types

        # Build the prompt with urgency and male grammar instructions
        if urgent:
            # For urgent announcements - speak with urgency and slight panic
            prompt = f"""זוהי הודעת חירום דחופה! דבר בלשון זכר, בטון דחוף ומבוהל מעט, כאילו מדובר באירוע ביטחוני אמיתי.
הקרא את ההודעה הבאה בדחיפות:

{text}"""
        else:
            prompt = f"""דבר בלשון זכר, בבהירות ובנחישות.
הקרא את ההודעה הבאה:

{text}"""

        def _generate_sync():
            try:
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=types.SpeechConfig(
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name=self.voice
                                )
                            )
                        )
                    )
                )

                # Extract audio data from response
                if response.candidates and len(response.candidates) > 0:
                    candidate = response.candidates[0]
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            if hasattr(part, 'inline_data') and part.inline_data:
                                audio_data = part.inline_data.data
                                mime_type = part.inline_data.mime_type or ""

                                logger.info(f"TTS response mime_type: {mime_type}, data length: {len(audio_data)}")

                                # The audio is typically raw PCM L16 at 24kHz
                                # Save as WAV with proper header
                                self._save_pcm_as_wav(audio_data, output_path, mime_type)
                                return True

                logger.error("No audio data in Gemini TTS response")
                logger.error(f"Response: {response}")
                return False

            except Exception as e:
                logger.error(f"Gemini TTS generation error: {e}")
                raise

        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(None, _generate_sync)

        if not success:
            raise RuntimeError("Failed to generate TTS audio")

    def _save_pcm_as_wav(self, audio_data: bytes, output_path: Path, mime_type: str = ""):
        """Save PCM audio data as WAV file with proper header"""
        # Parse sample rate from mime type if available
        # Format: audio/L16;rate=24000 or audio/pcm;rate=24000
        sample_rate = self.sample_rate
        if "rate=" in mime_type:
            try:
                rate_str = mime_type.split("rate=")[1].split(";")[0].split(",")[0]
                sample_rate = int(rate_str)
            except:
                pass

        logger.info(f"Saving WAV: {len(audio_data)} bytes, sample_rate={sample_rate}")

        with wave.open(str(output_path), 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit (2 bytes per sample)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data)

    def _generate_silent_placeholder(self, text: str, output_path: Path):
        """Generate a short silent WAV file as placeholder when TTS is not available"""
        logger.warning(f"TTS not available, generating silent placeholder for: {text[:50]}...")

        # Generate 0.5 seconds of silence
        duration_seconds = 0.5
        num_samples = int(self.sample_rate * duration_seconds)

        with wave.open(str(output_path), 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)
            # Write silence (zeros)
            silent_data = struct.pack('<' + 'h' * num_samples, *([0] * num_samples))
            wav_file.writeframes(silent_data)

    async def generate_emergency_announcement(
        self,
        details: Dict[str, Any]
    ) -> str:
        """
        Generate full emergency announcement with all incident details.
        Uses בלשון זכר (male grammar) and urgent tone.

        Args:
            details: Dict containing incident information

        Returns:
            Path to generated audio file
        """
        # Build announcement text (male grammar - בלשון זכר)
        parts = ["חדירה ודאית! אירוע אמת!"]

        location = details.get("location")
        if location:
            parts.append(f"מיקום: {location}.")

        person_count = details.get("person_count", 0)
        if person_count:
            parts.append(f"זוהו {person_count} מחבלים.")

        if details.get("armed"):
            weapon_type = details.get("weapon_type", "לא ידוע")
            parts.append(f"המחבלים חמושים! סוג נשק: {weapon_type}.")

        # Person descriptions - use rich Gemini analysis data
        persons = details.get("persons", [])
        for i, person in enumerate(persons[:3], 1):
            if isinstance(person, dict):
                # Priority 1: Use the full description from Gemini (most detailed)
                description = person.get("description", "")
                if description and description not in ["לא זוהה", "null", ""]:
                    parts.append(f"אדם {i}: {description}.")
                    continue

                # Priority 2: Use aggregated clothing info
                desc_parts = []
                clothing = person.get("clothing", "")
                clothing_color = person.get("clothingColor", "")
                age_range = person.get("ageRange", "")
                gender = person.get("gender", "")

                if gender and gender not in ["לא ניתן לקבוע", "לא זוהה"]:
                    desc_parts.append(gender)
                if age_range and age_range not in ["לא זוהה", "null"]:
                    desc_parts.append(age_range)
                if clothing_color and clothing_color not in ["לא זוהה", "null"]:
                    desc_parts.append(clothing_color)
                elif clothing and clothing not in ["לא זוהה", "null"]:
                    desc_parts.append(f"לבוש {clothing}")

                # Priority 3: Fallback to raw fields
                if not desc_parts:
                    shirt = person.get("shirtColor", "")
                    pants = person.get("pantsColor", "")
                    headwear = person.get("headwear", "")

                    if shirt:
                        desc_parts.append(f"חולצה {shirt}")
                    if pants:
                        desc_parts.append(f"מכנס {pants}")
                    if headwear and headwear != "ללא":
                        desc_parts.append(headwear)

                if desc_parts:
                    parts.append(f"אדם {i}: {', '.join(desc_parts)}.")

        # Vehicle info
        vehicle = details.get("vehicle")
        if vehicle and isinstance(vehicle, dict) and not vehicle.get("error"):
            color = vehicle.get("color", "")
            manufacturer = vehicle.get("manufacturer", "")
            model = vehicle.get("model", "")
            plate = vehicle.get("licensePlate")

            vehicle_desc = " ".join(filter(None, [color, manufacturer, model]))
            if vehicle_desc:
                parts.append(f"רכב: {vehicle_desc}.")
            if plate:
                parts.append(f"מספר רכב: {plate}.")

        parts.append("קו דיווח 13!")

        full_text = " ".join(parts)
        logger.info(f"Emergency announcement text: {full_text}")

        return await self.generate(full_text, urgent=True)

    async def generate_event_end_announcement(
        self,
        additional_info: Optional[str] = None
    ) -> str:
        """Generate end-of-event announcement."""
        text = "חדל! האיום נוטרל! סוף אירוע!"

        if additional_info:
            text += f" {additional_info}"

        return await self.generate(text, urgent=True)

    async def generate_simulation_announcement(
        self,
        simulation_type: str
    ) -> str:
        """Generate announcement for simulation events."""
        announcements = {
            "drone_dispatch": "רחפן הוקפץ לנקודת האירוע!",
            "phone_call": "מבצע חיוג למפקד התורן.",
            "pa_announcement": "מתבצעת כריזה למגורים!",
            "code_broadcast": "קוד צפרדע שודר שלוש פעמים ברשת הקשר!",
            "threat_neutralized": "חדל! האיום נוטרל! סוף אירוע!"
        }

        text = announcements.get(simulation_type, f"סימולציה: {simulation_type}")
        return await self.generate(text, urgent=True)

    async def generate_stolen_vehicle_announcement(
        self,
        vehicle_info: Dict[str, Any]
    ) -> str:
        """Generate announcement for stolen vehicle detection."""
        parts = ["רכב גנוב זוהה!"]

        color = vehicle_info.get("color", "")
        manufacturer = vehicle_info.get("manufacturer", "")
        model = vehicle_info.get("model", "")
        plate = vehicle_info.get("licensePlate", "")

        if color or manufacturer or model:
            vehicle_desc = " ".join(filter(None, [color, manufacturer, model]))
            parts.append(f"תיאור רכב: {vehicle_desc}.")

        if plate:
            parts.append(f"מספר רכב: {plate}.")

        parts.append("יש לפעול בהתאם לנהלים!")

        full_text = " ".join(parts)
        return await self.generate(full_text, urgent=True)

    def get_engine_info(self) -> Dict[str, Any]:
        """Get information about the current TTS engine"""
        return {
            "engine": "gemini_tts",
            "model": self.model_id,
            "voice": self.voice,
            "sample_rate": self.sample_rate,
            "output_dir": str(self.output_dir),
            "configured": self.is_configured()
        }


# Singleton instance
_tts_service: Optional[TTSService] = None


def get_tts_service() -> TTSService:
    """Get or create TTS service singleton"""
    global _tts_service
    if _tts_service is None:
        _tts_service = TTSService()
    return _tts_service
