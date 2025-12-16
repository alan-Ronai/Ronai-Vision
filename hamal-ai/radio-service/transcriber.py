"""
Radio Transcription Service - Upgraded

Integrates SimpleRTPReceiver with multiple STT backends:
1. Google Cloud Speech-to-Text (highest accuracy for Hebrew)
2. Gemini Audio Transcription (unified with other analysis)
3. Whisper (offline fallback)

Sends transcriptions to the backend via HTTP and detects voice commands.
"""

import os
import sys
import asyncio
import numpy as np
import httpx
import logging
import tempfile
import wave
from pathlib import Path
from datetime import datetime
from threading import Thread
from queue import Queue
from typing import Optional, Generator

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.audio.simple_rtp_receiver import SimpleRTPReceiver

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Environment configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:3000")
RTP_PORT = int(os.getenv("RTP_PORT", "5004"))
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "8000"))
BUFFER_DURATION = float(os.getenv("BUFFER_DURATION", "3.0"))  # seconds
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")  # tiny, base, small, medium, large
STT_ENGINE = os.getenv("STT_ENGINE", "auto")  # auto, google_cloud, gemini, whisper

# Try to import STT backends
GOOGLE_CLOUD_STT_AVAILABLE = False
GEMINI_AVAILABLE = False
WHISPER_AVAILABLE = False

try:
    from google.cloud import speech
    GOOGLE_CLOUD_STT_AVAILABLE = True
except ImportError:
    pass

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    pass

try:
    # Try faster-whisper first
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
    WHISPER_TYPE = "faster"
except ImportError:
    try:
        import whisper
        WHISPER_AVAILABLE = True
        WHISPER_TYPE = "openai"
    except ImportError:
        pass

# Audio buffer queue
audio_queue = Queue()


class RadioTranscriber:
    """
    Radio transcription service with multiple STT backend support.

    Priority order:
    1. Google Cloud STT (best Hebrew accuracy)
    2. Gemini (good accuracy, unified with vision analysis)
    3. Whisper (offline fallback)
    """

    def __init__(self):
        self.stt_engine = None
        self.engine_type = None
        self.running = False
        self.receiver = None

        self._init_stt_engine()

    def _init_stt_engine(self):
        """Initialize the best available STT engine"""
        engine_preference = STT_ENGINE

        # Auto-detect best engine
        if engine_preference == "auto":
            if GOOGLE_CLOUD_STT_AVAILABLE and os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
                engine_preference = "google_cloud"
            elif GEMINI_AVAILABLE and os.getenv("GEMINI_API_KEY"):
                engine_preference = "gemini"
            elif WHISPER_AVAILABLE:
                engine_preference = "whisper"
            else:
                logger.error("No STT engine available!")
                return

        # Initialize selected engine
        if engine_preference == "google_cloud" and GOOGLE_CLOUD_STT_AVAILABLE:
            self._init_google_cloud()
        elif engine_preference == "gemini" and GEMINI_AVAILABLE:
            self._init_gemini()
        elif engine_preference == "whisper" and WHISPER_AVAILABLE:
            self._init_whisper()
        else:
            logger.error(f"Requested STT engine '{engine_preference}' not available")

    def _init_google_cloud(self):
        """Initialize Google Cloud Speech-to-Text"""
        try:
            self.stt_engine = speech.SpeechClient()
            self.stt_config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=SAMPLE_RATE,
                language_code="he-IL",
                enable_automatic_punctuation=True,
                model="latest_long",  # Better for longer audio
            )
            self.engine_type = "google_cloud"
            logger.info("✅ STT: Using Google Cloud Speech-to-Text")
        except Exception as e:
            logger.error(f"Google Cloud STT init failed: {e}")

    def _init_gemini(self):
        """Initialize Gemini for audio transcription"""
        try:
            api_key = os.getenv("GEMINI_API_KEY")
            genai.configure(api_key=api_key)
            self.stt_engine = genai.GenerativeModel("gemini-1.5-flash")
            self.engine_type = "gemini"
            logger.info("✅ STT: Using Gemini Audio Transcription")
        except Exception as e:
            logger.error(f"Gemini init failed: {e}")

    def _init_whisper(self):
        """Initialize Whisper for audio transcription"""
        try:
            logger.info(f"Loading Whisper model: {WHISPER_MODEL}")
            if WHISPER_TYPE == "faster":
                self.stt_engine = WhisperModel(
                    WHISPER_MODEL,
                    device="cpu",
                    compute_type="int8"
                )
            else:
                import whisper
                self.stt_engine = whisper.load_model(WHISPER_MODEL)
            self.engine_type = "whisper"
            logger.info(f"✅ STT: Using {WHISPER_TYPE}-whisper ({WHISPER_MODEL})")
        except Exception as e:
            logger.error(f"Whisper init failed: {e}")

    def audio_callback(self, audio_data: bytes, sample_rate: int):
        """Called by RTP receiver with each audio chunk"""
        audio_queue.put({
            "data": audio_data,
            "sample_rate": sample_rate,
            "timestamp": datetime.now()
        })

    async def transcribe(self, audio_data: np.ndarray) -> str:
        """
        Transcribe audio using the configured STT engine.

        Args:
            audio_data: Audio samples as numpy array (int16)

        Returns:
            Transcribed Hebrew text
        """
        if self.stt_engine is None:
            return ""

        if self.engine_type == "google_cloud":
            return await self._transcribe_google_cloud(audio_data)
        elif self.engine_type == "gemini":
            return await self._transcribe_gemini(audio_data)
        elif self.engine_type == "whisper":
            return await self._transcribe_whisper(audio_data)

        return ""

    async def _transcribe_google_cloud(self, audio_data: np.ndarray) -> str:
        """Transcribe using Google Cloud STT"""
        try:
            # Convert to bytes
            audio_bytes = audio_data.astype(np.int16).tobytes()

            audio = speech.RecognitionAudio(content=audio_bytes)
            response = self.stt_engine.recognize(
                config=self.stt_config,
                audio=audio
            )

            transcript = ""
            for result in response.results:
                transcript += result.alternatives[0].transcript + " "

            return transcript.strip()

        except Exception as e:
            logger.error(f"Google Cloud transcription error: {e}")
            return ""

    async def _transcribe_gemini(self, audio_data: np.ndarray) -> str:
        """Transcribe using Gemini"""
        try:
            # Save audio to temp file (Gemini needs file upload)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
                with wave.open(f, 'wb') as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)  # 16-bit
                    wav.setframerate(SAMPLE_RATE)
                    wav.writeframes(audio_data.astype(np.int16).tobytes())

            # Upload to Gemini
            audio_file = genai.upload_file(temp_path)

            prompt = """
            תמלל את האודיו הזה לעברית.
            החזר רק את הטקסט המתומלל, בלי הערות או תוספות.
            אם יש מילים לא ברורות, נסה לנחש לפי הקונטקסט הצבאי/ביטחוני.
            """

            response = await self.stt_engine.generate_content_async([prompt, audio_file])
            transcript = response.text.strip()

            # Clean up temp file
            os.unlink(temp_path)

            return transcript

        except Exception as e:
            logger.error(f"Gemini transcription error: {e}")
            return ""

    async def _transcribe_whisper(self, audio_data: np.ndarray) -> str:
        """Transcribe using Whisper"""
        try:
            # Normalize audio
            audio = audio_data.astype(np.float32) / 32768.0

            # Resample to 16kHz if needed
            if SAMPLE_RATE != 16000:
                from scipy import signal
                audio = signal.resample(audio, int(len(audio) * 16000 / SAMPLE_RATE))

            if WHISPER_TYPE == "faster":
                segments, info = self.stt_engine.transcribe(
                    audio,
                    language="he",
                    beam_size=5,
                    vad_filter=True
                )
                transcript = " ".join([seg.text for seg in segments]).strip()
            else:
                result = self.stt_engine.transcribe(
                    audio,
                    language="he",
                    fp16=False
                )
                transcript = result["text"].strip()

            return transcript

        except Exception as e:
            logger.error(f"Whisper transcription error: {e}")
            return ""

    async def send_transcription(self, text: str):
        """Send transcription to backend and check for commands"""
        if not text:
            return

        logger.info(f"📻 Transcription: {text}")

        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{BACKEND_URL}/api/radio/transcription",
                    json={
                        "text": text,
                        "timestamp": datetime.now().isoformat(),
                        "source": "radio",
                        "engine": self.engine_type
                    },
                    timeout=5.0
                )
        except Exception as e:
            logger.error(f"Failed to send transcription: {e}")

        # Check for voice commands
        await self.check_commands(text)

    async def check_commands(self, text: str):
        """Check for voice commands and trigger simulations"""
        text_lower = text.lower()
        commands = []

        # Drone dispatch
        if "רחפן" in text or "חוזי" in text or "הקפיצו" in text:
            commands.append(("drone_dispatch", "🚁 Drone dispatch"))

        # Code broadcast (צפרדע)
        if "צפרדע" in text:
            commands.append(("code_broadcast", "📻 Code broadcast"))

        # Phone call
        if "התקשרו" in text or "חייגו" in text or "טלפון" in text:
            commands.append(("phone_call", "📞 Phone call"))

        # PA announcement
        if "כריזה" in text or "כרזו" in text:
            commands.append(("pa_announcement", "📢 PA announcement"))

        # End of event
        if "חדל" in text or "סיום" in text or "נוטרל" in text:
            commands.append(("threat_neutralized", "✅ Threat neutralized"))

        # Send detected commands
        for cmd_type, description in commands:
            logger.info(f"Command detected: {description}")
            try:
                async with httpx.AsyncClient() as client:
                    await client.post(
                        f"{BACKEND_URL}/api/radio/command",
                        json={"command": cmd_type},
                        timeout=5.0
                    )
            except Exception as e:
                logger.error(f"Failed to send command: {e}")

    async def process_queue(self):
        """Process audio queue and transcribe when buffer is full"""
        buffer = []
        buffer_samples = int(SAMPLE_RATE * BUFFER_DURATION)

        while self.running:
            try:
                # Get audio from queue (non-blocking with timeout)
                try:
                    chunk = audio_queue.get(timeout=0.1)
                    samples = np.frombuffer(chunk["data"], dtype=np.int16)
                    buffer.extend(samples)
                except:
                    pass

                # When buffer is full, transcribe
                if len(buffer) >= buffer_samples:
                    audio_array = np.array(buffer[:buffer_samples], dtype=np.int16)
                    buffer = buffer[buffer_samples:]

                    # Transcribe and send
                    transcript = await self.transcribe(audio_array)
                    if transcript:
                        await self.send_transcription(transcript)

            except Exception as e:
                logger.error(f"Queue processing error: {e}")
                await asyncio.sleep(0.1)

    def start(self):
        """Start RTP receiver with transcription"""
        if self.stt_engine is None:
            logger.error("Cannot start - no STT engine available")
            return

        logger.info(f"Starting radio transcriber on port {RTP_PORT}")

        # Create RTP receiver
        self.receiver = SimpleRTPReceiver(
            listen_host="0.0.0.0",
            listen_port=RTP_PORT,
            target_sample_rate=SAMPLE_RATE,
            audio_callback=self.audio_callback,
            inactivity_timeout=3.0
        )

        self.running = True

        # Start receiver
        self.receiver.start()

        logger.info(f"""
╔════════════════════════════════════════════════════════╗
║                                                        ║
║   📻 Radio Transcription Service                       ║
║   שירות תמלול קשר                                       ║
║                                                        ║
║   Listening on UDP port {RTP_PORT}                       ║
║   STT Engine: {self.engine_type}
║   Backend: {BACKEND_URL}
║                                                        ║
╚════════════════════════════════════════════════════════╝
        """)

        # Run async processing
        asyncio.run(self.process_queue())

    def stop(self):
        """Stop the transcriber"""
        self.running = False
        if self.receiver:
            self.receiver.stop()
        logger.info("Radio transcriber stopped")

    def get_status(self):
        """Get transcriber status"""
        return {
            "running": self.running,
            "engine": self.engine_type,
            "port": RTP_PORT,
            "sample_rate": SAMPLE_RATE,
            "buffer_duration": BUFFER_DURATION,
            "google_cloud_available": GOOGLE_CLOUD_STT_AVAILABLE,
            "gemini_available": GEMINI_AVAILABLE,
            "whisper_available": WHISPER_AVAILABLE
        }


def main():
    """Main entry point"""
    transcriber = RadioTranscriber()

    try:
        transcriber.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        transcriber.stop()


if __name__ == "__main__":
    main()
