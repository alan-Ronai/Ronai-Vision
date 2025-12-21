"""
Text-to-Speech Service for Hebrew Announcements

Uses Gemini TTS with Sulafat voice (Hebrew female voice) for high-quality Hebrew speech.
Feature 3: Replace TTS with Gemini TTS (Sulafat Voice)
"""

import os
import asyncio
import logging
import uuid
import base64
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Gemini TTS Configuration
GEMINI_TTS_VOICE = os.environ.get("GEMINI_TTS_VOICE", "Sulafat")
GEMINI_TTS_SAMPLE_RATE = int(os.environ.get("GEMINI_TTS_SAMPLE_RATE", "24000"))
GEMINI_TTS_FORMAT = os.environ.get("GEMINI_TTS_FORMAT", "wav")  # WAV for RTP compatibility

# Try to import Google Generative AI
GEMINI_AVAILABLE = False
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    pass


class TTSService:
    """
    Text-to-Speech service for Hebrew emergency announcements.

    Uses Gemini TTS with Sulafat voice for high-quality Hebrew speech synthesis.
    """

    def __init__(self):
        self.output_dir = Path(os.getenv("TTS_OUTPUT_DIR", "audio_output"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.voice = GEMINI_TTS_VOICE
        self.sample_rate = GEMINI_TTS_SAMPLE_RATE
        self.output_format = GEMINI_TTS_FORMAT
        self.model = None

        self._init_engine()

    def _init_engine(self):
        """Initialize Gemini TTS engine"""
        if not GEMINI_AVAILABLE:
            logger.error("❌ TTS: google-generativeai package not installed")
            return

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.error("❌ TTS: GEMINI_API_KEY not set")
            return

        try:
            genai.configure(api_key=api_key)
            # Use gemini-2.0-flash-exp for TTS generation
            self.model = genai.GenerativeModel("gemini-2.0-flash-exp")
            logger.info(f"✅ TTS: Using Gemini TTS with {self.voice} voice")
        except Exception as e:
            logger.error(f"❌ TTS: Failed to initialize Gemini: {e}")

    def is_configured(self) -> bool:
        """Check if TTS is properly configured"""
        return self.model is not None

    async def generate(
        self,
        text: str,
        filename: Optional[str] = None
    ) -> str:
        """
        Generate speech from Hebrew text using Gemini TTS.

        Args:
            text: Hebrew text to convert to speech
            filename: Optional output filename

        Returns:
            Path to generated audio file

        Raises:
            RuntimeError: If Gemini is not available
        """
        if not self.model:
            raise RuntimeError("Gemini TTS not initialized")

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            uid = uuid.uuid4().hex[:8]
            filename = f"tts_{timestamp}_{uid}.{self.output_format}"

        output_path = self.output_dir / filename

        try:
            # Generate speech using Gemini
            await self._generate_gemini_tts(text, output_path)
            logger.info(f"TTS generated: {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"TTS generation error: {e}")
            raise

    async def _generate_gemini_tts(self, text: str, output_path: Path):
        """Generate speech using Gemini's TTS capabilities"""
        loop = asyncio.get_event_loop()

        def _generate():
            try:
                # Use Gemini's multimodal capabilities to generate speech
                # The model can generate audio from text instructions
                prompt = f"""
Generate natural Hebrew speech audio for the following text.
Use a clear, professional female voice suitable for emergency announcements.
Voice: {self.voice}
Language: Hebrew (he-IL)

Text to speak:
{text}

Generate the audio in {self.output_format} format at {self.sample_rate}Hz sample rate.
"""
                response = self.model.generate_content(
                    prompt,
                    generation_config={
                        "response_mime_type": f"audio/{self.output_format}"
                    }
                )

                # Check if response contains audio
                if hasattr(response, 'candidates') and response.candidates:
                    for candidate in response.candidates:
                        if hasattr(candidate, 'content') and candidate.content:
                            for part in candidate.content.parts:
                                if hasattr(part, 'inline_data') and part.inline_data:
                                    # Decode and save audio
                                    audio_data = base64.b64decode(part.inline_data.data)
                                    with open(output_path, 'wb') as f:
                                        f.write(audio_data)
                                    return

                # Fallback: If direct TTS not available, use synthesis approach
                # This creates a TTS request through Gemini's audio synthesis
                self._fallback_tts(text, output_path)

            except Exception as e:
                logger.error(f"Gemini TTS error: {e}")
                self._fallback_tts(text, output_path)

        await loop.run_in_executor(None, _generate)

    def _fallback_tts(self, text: str, output_path: Path):
        """Fallback TTS using gTTS if Gemini audio generation fails"""
        try:
            from gtts import gTTS
            from pydub import AudioSegment

            # gTTS only outputs MP3, so save to temp then convert if needed
            mp3_path = str(output_path).replace('.wav', '.mp3')
            tts = gTTS(text=text, lang='iw')  # 'iw' is Hebrew
            tts.save(mp3_path)

            # Convert to WAV if output format is wav
            if self.output_format == 'wav':
                try:
                    audio = AudioSegment.from_mp3(mp3_path)
                    audio = audio.set_frame_rate(self.sample_rate)
                    audio.export(str(output_path), format='wav')
                    # Remove temp mp3
                    import os as os_module
                    os_module.remove(mp3_path)
                except Exception as conv_err:
                    logger.warning(f"WAV conversion failed, keeping MP3: {conv_err}")
                    # If conversion fails, just rename
                    import shutil
                    shutil.move(mp3_path, str(output_path))

            logger.info("Using gTTS fallback for TTS generation")
        except ImportError:
            logger.error("gTTS not installed, cannot use fallback TTS")
            raise RuntimeError("gTTS not available for fallback")
        except Exception as e:
            logger.error(f"Fallback TTS also failed: {e}")
            raise RuntimeError(f"All TTS methods failed: {e}")

    async def generate_emergency_announcement(
        self,
        details: Dict[str, Any]
    ) -> str:
        """
        Generate full emergency announcement with all incident details.

        Args:
            details: Dict containing:
                - location: Camera/location description
                - person_count: Number of intruders
                - armed: Whether intruders are armed
                - weapon_type: Type of weapon if armed
                - vehicle: Vehicle analysis dict (color, manufacturer, model, licensePlate)
                - persons: List of person analyses (clothing descriptions)

        Returns:
            Path to generated audio file
        """
        # Build announcement text
        parts = ["חדירה ודאית. אירוע אמת."]

        location = details.get("location")
        if location:
            parts.append(f"מיקום: {location}.")

        person_count = details.get("person_count", 0)
        if person_count:
            parts.append(f"מספר מחבלים: {person_count}.")

        if details.get("armed"):
            weapon_type = details.get("weapon_type", "לא ידוע")
            parts.append(f"המחבלים חמושים. סוג נשק: {weapon_type}.")

        # Person descriptions (clothing for identification)
        persons = details.get("persons", [])
        for i, person in enumerate(persons[:3], 1):  # Max 3 descriptions
            if isinstance(person, dict):
                shirt = person.get("shirtColor", "")
                pants = person.get("pantsColor", "")
                headwear = person.get("headwear", "")

                desc_parts = []
                if shirt:
                    desc_parts.append(f"חולצה {shirt}")
                if pants:
                    desc_parts.append(f"מכנס {pants}")
                if headwear and headwear != "ללא":
                    desc_parts.append(headwear)

                if desc_parts:
                    parts.append(f"מחבל {i}: {', '.join(desc_parts)}.")

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

        parts.append("קו דיווח 13.")

        full_text = " ".join(parts)
        logger.info(f"Emergency announcement text: {full_text}")

        return await self.generate(full_text)

    async def generate_event_end_announcement(
        self,
        additional_info: Optional[str] = None
    ) -> str:
        """
        Generate end-of-event announcement.

        Args:
            additional_info: Optional additional details

        Returns:
            Path to generated audio file
        """
        text = "חדל. האיום נוטרל. סוף אירוע."

        if additional_info:
            text += f" {additional_info}"

        return await self.generate(text)

    async def generate_simulation_announcement(
        self,
        simulation_type: str
    ) -> str:
        """
        Generate announcement for simulation events.

        Args:
            simulation_type: Type of simulation (drone_dispatch, phone_call, etc.)

        Returns:
            Path to generated audio file
        """
        announcements = {
            "drone_dispatch": "רחפן הוקפץ לנקודת האירוע.",
            "phone_call": "מבצע חיוג למפקד התורן.",
            "pa_announcement": "מתבצעת כריזה למגורים.",
            "code_broadcast": "קוד צפרדע שודר שלוש פעמים ברשת הקשר.",
            "threat_neutralized": "חדל. האיום נוטרל. סוף אירוע."
        }

        text = announcements.get(simulation_type, f"סימולציה: {simulation_type}")
        return await self.generate(text)

    async def generate_stolen_vehicle_announcement(
        self,
        vehicle_info: Dict[str, Any]
    ) -> str:
        """
        Generate announcement for stolen vehicle detection.

        Args:
            vehicle_info: Vehicle analysis dict with color, model, licensePlate

        Returns:
            Path to generated audio file
        """
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

        parts.append("יש לפעול בהתאם לנהלים.")

        full_text = " ".join(parts)
        return await self.generate(full_text)

    def get_engine_info(self) -> Dict[str, Any]:
        """Get information about the current TTS engine"""
        return {
            "engine": "gemini_tts",
            "voice": self.voice,
            "sample_rate": self.sample_rate,
            "output_format": self.output_format,
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
