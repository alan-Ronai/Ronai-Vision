"""
Text-to-Speech Service for Hebrew Announcements

Supports multiple TTS backends (in priority order):
1. Google Cloud TTS - High quality Hebrew (requires service account)
2. gTTS (Google Text-to-Speech) - Online, good quality
3. pyttsx3 - Offline fallback

Google Cloud TTS provides the best Hebrew voice quality.
"""

import os
import asyncio
import logging
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Try imports in priority order
GOOGLE_CLOUD_TTS_AVAILABLE = False
GTTS_AVAILABLE = False
PYTTSX3_AVAILABLE = False

try:
    from google.cloud import texttospeech
    GOOGLE_CLOUD_TTS_AVAILABLE = True
except ImportError:
    pass

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    pass

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    pass


class TTSService:
    """
    Text-to-Speech service for Hebrew emergency announcements.

    Automatically selects the best available TTS engine:
    1. Google Cloud TTS (highest quality)
    2. gTTS (good quality, online)
    3. pyttsx3 (offline fallback)
    """

    def __init__(self):
        self.output_dir = Path(os.getenv("TTS_OUTPUT_DIR", "audio_output"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.engine_type = None
        self.cloud_client = None
        self.pyttsx3_engine = None

        self._init_engine()

    def _init_engine(self):
        """Initialize the best available TTS engine"""

        # Try Google Cloud TTS first (best quality)
        if GOOGLE_CLOUD_TTS_AVAILABLE:
            creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if creds_path and os.path.exists(creds_path):
                try:
                    self.cloud_client = texttospeech.TextToSpeechClient()
                    self.engine_type = "google_cloud"
                    logger.info("✅ TTS: Using Google Cloud TTS (highest quality Hebrew)")
                    return
                except Exception as e:
                    logger.warning(f"Google Cloud TTS init failed: {e}")
            else:
                logger.info("Google Cloud credentials not found, trying alternatives")

        # Try gTTS (online, good quality)
        if GTTS_AVAILABLE:
            self.engine_type = "gtts"
            logger.info("✅ TTS: Using gTTS (online)")
            return

        # Fall back to pyttsx3 (offline)
        if PYTTSX3_AVAILABLE:
            try:
                self.pyttsx3_engine = pyttsx3.init()
                self.engine_type = "pyttsx3"
                logger.info("✅ TTS: Using pyttsx3 (offline)")

                # Try to find Hebrew voice
                voices = self.pyttsx3_engine.getProperty('voices')
                for voice in voices:
                    if 'hebrew' in voice.name.lower() or 'he' in voice.id.lower():
                        self.pyttsx3_engine.setProperty('voice', voice.id)
                        logger.info(f"  Found Hebrew voice: {voice.name}")
                        break

                self.pyttsx3_engine.setProperty('rate', 150)
                return
            except Exception as e:
                logger.warning(f"pyttsx3 init failed: {e}")

        logger.warning("⚠️ No TTS engine available")
        self.engine_type = None

    def is_configured(self) -> bool:
        """Check if TTS is properly configured"""
        return self.engine_type is not None

    async def generate(
        self,
        text: str,
        filename: Optional[str] = None
    ) -> str:
        """
        Generate speech from Hebrew text.

        Args:
            text: Hebrew text to convert to speech
            filename: Optional output filename

        Returns:
            Path to generated audio file

        Raises:
            RuntimeError: If no TTS engine is available
        """
        if not self.engine_type:
            raise RuntimeError("No TTS engine available")

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            uid = uuid.uuid4().hex[:8]
            filename = f"tts_{timestamp}_{uid}.mp3"

        output_path = self.output_dir / filename

        if self.engine_type == "google_cloud":
            await self._generate_google_cloud(text, output_path)
        elif self.engine_type == "gtts":
            await self._generate_gtts(text, output_path)
        elif self.engine_type == "pyttsx3":
            await self._generate_pyttsx3(text, output_path)

        logger.info(f"TTS generated: {output_path}")
        return str(output_path)

    async def _generate_google_cloud(self, text: str, output_path: Path):
        """Generate speech using Google Cloud TTS"""
        loop = asyncio.get_event_loop()

        def _generate():
            synthesis_input = texttospeech.SynthesisInput(text=text)

            # Use high quality Hebrew voice
            voice = texttospeech.VoiceSelectionParams(
                language_code="he-IL",
                name="he-IL-Wavenet-A",  # Wavenet = highest quality
                ssml_gender=texttospeech.SsmlVoiceGender.MALE
            )

            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=1.0,
                pitch=0.0
            )

            response = self.cloud_client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )

            with open(output_path, "wb") as out:
                out.write(response.audio_content)

        await loop.run_in_executor(None, _generate)

    async def _generate_gtts(self, text: str, output_path: Path):
        """Generate speech using gTTS"""
        loop = asyncio.get_event_loop()

        def _generate():
            tts = gTTS(text=text, lang='iw')  # 'iw' is Hebrew
            tts.save(str(output_path))

        await loop.run_in_executor(None, _generate)

    async def _generate_pyttsx3(self, text: str, output_path: Path):
        """Generate speech using pyttsx3"""
        loop = asyncio.get_event_loop()

        def _generate():
            # pyttsx3 saves to wav, adjust filename
            wav_path = str(output_path).replace('.mp3', '.wav')
            self.pyttsx3_engine.save_to_file(text, wav_path)
            self.pyttsx3_engine.runAndWait()
            return wav_path

        await loop.run_in_executor(None, _generate)

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

    def speak_sync(self, text: str):
        """
        Speak text directly and synchronously (blocking).
        Only works with pyttsx3 engine.
        For real-time announcements.
        """
        if self.engine_type == "pyttsx3" and self.pyttsx3_engine:
            self.pyttsx3_engine.say(text)
            self.pyttsx3_engine.runAndWait()
        else:
            logger.warning(f"speak_sync not available with {self.engine_type}")

    def get_engine_info(self) -> Dict[str, Any]:
        """Get information about the current TTS engine"""
        return {
            "engine": self.engine_type,
            "output_dir": str(self.output_dir),
            "google_cloud_available": GOOGLE_CLOUD_TTS_AVAILABLE,
            "gtts_available": GTTS_AVAILABLE,
            "pyttsx3_available": PYTTSX3_AVAILABLE
        }
