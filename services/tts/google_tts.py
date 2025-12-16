"""Google Cloud Text-to-Speech service for high-quality Hebrew audio.

Generates natural Hebrew speech for emergency announcements and alerts.
Includes fallback to gTTS (Google Translate TTS) if Cloud TTS unavailable.
"""

import logging
import os
import uuid
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import Google Cloud TTS
GOOGLE_TTS_AVAILABLE = False
try:
    from google.cloud import texttospeech
    GOOGLE_TTS_AVAILABLE = True
except ImportError:
    logger.warning(
        "google-cloud-texttospeech not installed. "
        "Install with: pip install google-cloud-texttospeech"
    )

# Try to import gTTS as fallback
GTTS_AVAILABLE = False
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    logger.warning(
        "gTTS not installed. Install with: pip install gtts"
    )


class GoogleTTSService:
    """Google Cloud TTS service for Hebrew speech synthesis."""

    # Available Hebrew voices
    VOICES = {
        "male_1": "he-IL-Wavenet-A",
        "male_2": "he-IL-Wavenet-C",
        "female_1": "he-IL-Wavenet-B",
        "female_2": "he-IL-Wavenet-D",
        "standard_male": "he-IL-Standard-A",
        "standard_female": "he-IL-Standard-B"
    }

    def __init__(
        self,
        output_dir: str = "audio_output",
        voice: str = "male_1",
        speaking_rate: float = 1.0,
        pitch: float = 0.0
    ):
        """Initialize Google TTS service.

        Args:
            output_dir: Directory to save audio files
            voice: Voice preset (male_1, male_2, female_1, female_2, standard_male, standard_female)
            speaking_rate: Speech speed (0.25 to 4.0, default 1.0)
            pitch: Voice pitch (-20.0 to 20.0, default 0.0)
        """
        if not GOOGLE_TTS_AVAILABLE:
            raise RuntimeError(
                "google-cloud-texttospeech not installed. "
                "Install with: pip install google-cloud-texttospeech"
            )

        # Check for credentials
        creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if not creds_path:
            logger.warning(
                "GOOGLE_APPLICATION_CREDENTIALS not set. "
                "Set it to your service account JSON file path."
            )
        elif not os.path.exists(creds_path):
            logger.warning(
                f"GOOGLE_APPLICATION_CREDENTIALS file not found: {creds_path}"
            )

        self.client = texttospeech.TextToSpeechClient()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Voice settings
        self.voice_name = self.VOICES.get(voice, self.VOICES["male_1"])
        self.speaking_rate = speaking_rate
        self.pitch = pitch

        logger.info(f"GoogleTTSService initialized with voice: {self.voice_name}")

    def generate(self, text: str, filename: Optional[str] = None) -> str:
        """Generate Hebrew speech from text.

        Args:
            text: Hebrew text to convert to speech
            filename: Optional filename (without extension)

        Returns:
            Path to generated MP3 file
        """
        try:
            # Prepare synthesis input
            synthesis_input = texttospeech.SynthesisInput(text=text)

            # Voice selection
            voice = texttospeech.VoiceSelectionParams(
                language_code="he-IL",
                name=self.voice_name,
                ssml_gender=texttospeech.SsmlVoiceGender.MALE
                if "A" in self.voice_name or "C" in self.voice_name
                else texttospeech.SsmlVoiceGender.FEMALE
            )

            # Audio config
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=self.speaking_rate,
                pitch=self.pitch
            )

            # Generate speech
            response = self.client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )

            # Save to file
            if filename is None:
                filename = f"tts_{uuid.uuid4().hex[:8]}"

            filepath = self.output_dir / f"{filename}.mp3"

            with open(filepath, "wb") as out:
                out.write(response.audio_content)

            logger.info(f"Generated TTS audio: {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            raise

    def generate_emergency_alert(
        self,
        camera_id: Optional[str] = None,
        person_count: int = 0,
        armed: bool = False,
        weapon_type: Optional[str] = None,
        vehicle: Optional[Dict[str, Any]] = None,
        location: Optional[str] = None
    ) -> str:
        """Generate emergency alert announcement.

        Args:
            camera_id: Camera identifier
            person_count: Number of suspects
            armed: Whether suspects are armed
            weapon_type: Type of weapon
            vehicle: Vehicle info dict
            location: Location description

        Returns:
            Path to generated MP3 file
        """
        parts = ["חדירה ודאית.", "אירוע אמת."]

        # Location
        if location:
            parts.append(f"{location}.")
        elif camera_id:
            parts.append(f"מצלמה {camera_id}.")

        # People count
        if person_count > 0:
            parts.append(f"מספר חשודים: {person_count}.")

        # Armed status
        if armed:
            if weapon_type:
                parts.append(f"החשודים חמושים. סוג נשק: {weapon_type}.")
            else:
                parts.append("החשודים חמושים.")

        # Vehicle info
        if vehicle:
            color = vehicle.get("צבע", "")
            model = vehicle.get("דגם", "")
            plate = vehicle.get("מספר_רישוי")
            vehicle_type = vehicle.get("סוג_רכב", "")

            vehicle_desc = " ".join(filter(None, [color, vehicle_type, model]))
            if vehicle_desc:
                parts.append(f"רכב: {vehicle_desc}.")

            if plate:
                # Read plate clearly
                plate_spoken = " ".join(str(plate).replace("-", " "))
                parts.append(f"מספר רכב: {plate_spoken}.")

        parts.append("קו דיווח 13.")

        text = " ".join(parts)
        return self.generate(text, filename=f"emergency_{uuid.uuid4().hex[:8]}")

    def generate_end_incident(self) -> str:
        """Generate end of incident announcement."""
        text = "חדל. האיום נוטרל. סוף אירוע."
        return self.generate(text, filename=f"end_incident_{uuid.uuid4().hex[:8]}")

    def generate_code_broadcast(self, code_word: str, repeat: int = 3) -> str:
        """Generate code word broadcast.

        Args:
            code_word: Code word to broadcast (e.g., "צפרדע")
            repeat: Number of times to repeat

        Returns:
            Path to generated MP3 file
        """
        words = [code_word] * repeat
        text = ". ".join(words) + "."
        return self.generate(text, filename=f"code_{code_word}_{uuid.uuid4().hex[:8]}")

    def generate_drone_dispatch(self, location: Optional[str] = None) -> str:
        """Generate drone dispatch announcement."""
        if location:
            text = f"רחפן הוקפץ לנקודה: {location}."
        else:
            text = "רחפן הוקפץ לנקודת האירוע."
        return self.generate(text, filename=f"drone_{uuid.uuid4().hex[:8]}")

    def generate_phone_call(self, recipient: Optional[str] = None) -> str:
        """Generate phone call announcement."""
        if recipient:
            text = f"מבצע חיוג ל{recipient}."
        else:
            text = "מבצע חיוג למפקד התורן."
        return self.generate(text, filename=f"phone_{uuid.uuid4().hex[:8]}")

    def generate_pa_announcement(self) -> str:
        """Generate PA announcement notification."""
        text = "מתבצעת כריזה למגורים."
        return self.generate(text, filename=f"pa_{uuid.uuid4().hex[:8]}")


class SimpleTTSService:
    """Simple TTS using gTTS (Google Translate TTS) - no API key needed."""

    def __init__(self, output_dir: str = "audio_output"):
        if not GTTS_AVAILABLE:
            raise RuntimeError("gTTS not installed. Install with: pip install gtts")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("SimpleTTSService initialized (gTTS)")

    def generate(self, text: str, filename: Optional[str] = None) -> str:
        """Generate Hebrew speech from text."""
        try:
            tts = gTTS(text=text, lang='iw')  # 'iw' is Hebrew

            if filename is None:
                filename = f"tts_{uuid.uuid4().hex[:8]}"

            filepath = self.output_dir / f"{filename}.mp3"
            tts.save(str(filepath))

            logger.info(f"Generated TTS audio: {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            raise

    def generate_emergency_alert(
        self,
        camera_id: Optional[str] = None,
        person_count: int = 0,
        armed: bool = False,
        weapon_type: Optional[str] = None,
        vehicle: Optional[Dict[str, Any]] = None,
        location: Optional[str] = None
    ) -> str:
        """Generate emergency alert announcement (same interface as GoogleTTSService)."""
        parts = ["חדירה ודאית.", "אירוע אמת."]

        if location:
            parts.append(f"{location}.")
        elif camera_id:
            parts.append(f"מצלמה {camera_id}.")

        if person_count > 0:
            parts.append(f"מספר חשודים: {person_count}.")

        if armed:
            if weapon_type:
                parts.append(f"החשודים חמושים. סוג נשק: {weapon_type}.")
            else:
                parts.append("החשודים חמושים.")

        if vehicle:
            color = vehicle.get("צבע", "")
            model = vehicle.get("דגם", "")
            plate = vehicle.get("מספר_רישוי")

            vehicle_desc = " ".join(filter(None, [color, model]))
            if vehicle_desc:
                parts.append(f"רכב: {vehicle_desc}.")
            if plate:
                plate_spoken = " ".join(str(plate).replace("-", " "))
                parts.append(f"מספר רכב: {plate_spoken}.")

        parts.append("קו דיווח 13.")

        text = " ".join(parts)
        return self.generate(text, filename=f"emergency_{uuid.uuid4().hex[:8]}")

    def generate_end_incident(self) -> str:
        """Generate end of incident announcement."""
        text = "חדל. האיום נוטרל. סוף אירוע."
        return self.generate(text, filename=f"end_incident_{uuid.uuid4().hex[:8]}")

    def generate_code_broadcast(self, code_word: str, repeat: int = 3) -> str:
        """Generate code word broadcast."""
        words = [code_word] * repeat
        text = ". ".join(words) + "."
        return self.generate(text, filename=f"code_{code_word}_{uuid.uuid4().hex[:8]}")

    def generate_drone_dispatch(self, location: Optional[str] = None) -> str:
        """Generate drone dispatch announcement."""
        if location:
            text = f"רחפן הוקפץ לנקודה: {location}."
        else:
            text = "רחפן הוקפץ לנקודת האירוע."
        return self.generate(text, filename=f"drone_{uuid.uuid4().hex[:8]}")


# Type alias for any TTS service
TTSService = GoogleTTSService | SimpleTTSService

# Global singleton
_tts_service: Optional[TTSService] = None


def get_tts_service(use_google_cloud: bool = True, output_dir: str = "audio_output") -> Optional[TTSService]:
    """Get or create TTS service.

    Args:
        use_google_cloud: If True, try Google Cloud TTS first
        output_dir: Directory for audio output

    Returns:
        TTS service instance
    """
    global _tts_service

    if _tts_service is not None:
        return _tts_service

    if use_google_cloud and GOOGLE_TTS_AVAILABLE:
        try:
            _tts_service = GoogleTTSService(output_dir=output_dir)
            return _tts_service
        except Exception as e:
            logger.warning(f"Google Cloud TTS failed, falling back to gTTS: {e}")

    # Fallback to simple gTTS
    if GTTS_AVAILABLE:
        try:
            _tts_service = SimpleTTSService(output_dir=output_dir)
            return _tts_service
        except Exception as e:
            logger.error(f"Failed to initialize gTTS: {e}")

    logger.error("No TTS service available")
    return None


def reset_tts_service():
    """Reset global TTS service."""
    global _tts_service
    _tts_service = None
