"""Audio file writer for saving received audio."""

import wave
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


class AudioWriter:
    """Writes audio data to WAV files."""

    def __init__(
        self,
        output_dir: str = "audio_storage/recordings",
        session_id: Optional[str] = None,
        codec: str = "raw_pcm_16bit",
        sample_rate: int = 16000,
        channels: int = 1,
        sample_width: int = 2  # 16-bit = 2 bytes
    ):
        self.output_dir = Path(output_dir)
        self.session_id = session_id or f"session_{int(datetime.now().timestamp())}"
        self.codec = codec
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width = sample_width

        self._wav_file: Optional[wave.Wave_write] = None
        self._filepath: Optional[str] = None

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def filepath(self) -> Optional[str]:
        return self._filepath

    def open(self):
        """Open WAV file for writing."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.session_id}_{timestamp}.wav"
        self._filepath = str(self.output_dir / filename)

        self._wav_file = wave.open(self._filepath, 'wb')
        self._wav_file.setnchannels(self.channels)
        self._wav_file.setsampwidth(self.sample_width)
        self._wav_file.setframerate(self.sample_rate)

        logger.info(f"Opened WAV file: {self._filepath}")

    def write(self, audio_data: np.ndarray):
        """Write audio samples to file.

        Args:
            audio_data: Numpy array of int16 samples
        """
        if self._wav_file is None:
            self.open()

        self._wav_file.writeframes(audio_data.tobytes())

    def write_bytes(self, audio_bytes: bytes):
        """Write raw audio bytes to file."""
        if self._wav_file is None:
            self.open()

        self._wav_file.writeframes(audio_bytes)

    def close(self):
        """Close WAV file."""
        if self._wav_file:
            self._wav_file.close()
            self._wav_file = None
            logger.info(f"Closed WAV file: {self._filepath}")
