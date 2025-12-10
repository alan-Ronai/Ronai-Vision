"""Audio file writer for storing decoded audio streams.

Writes audio to WAV files with metadata JSON.
"""

import os
import json
import time
import wave
import numpy as np
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, asdict


@dataclass
class AudioMetadata:
    """Metadata for audio recording."""
    session_id: str
    codec: str
    sample_rate: int
    channels: int
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    total_samples: int = 0
    total_bytes: int = 0
    packets_received: int = 0
    packets_lost: int = 0


class AudioWriter:
    """Writes decoded audio to WAV files."""

    def __init__(
        self,
        output_dir: str,
        session_id: str,
        codec: str,
        sample_rate: int,
        channels: int = 1,
        max_file_size_mb: int = 100
    ):
        """Initialize audio writer.

        Args:
            output_dir: Output directory for audio files
            session_id: Session identifier
            codec: Codec name
            sample_rate: Sample rate in Hz
            channels: Number of audio channels (default 1)
            max_file_size_mb: Maximum file size in MB before rotation (default 100)
        """
        self.output_dir = output_dir
        self.session_id = session_id
        self.codec = codec
        self.sample_rate = sample_rate
        self.channels = channels
        self.max_file_size_mb = max_file_size_mb

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"{timestamp}_{session_id}_{codec}.wav"
        self.filepath = os.path.join(output_dir, self.filename)
        self.metadata_filepath = self.filepath.replace(".wav", ".json")

        # WAV file handle
        self._wav_file: Optional[wave.Wave_write] = None
        self._is_open = False

        # Metadata
        self.metadata = AudioMetadata(
            session_id=session_id,
            codec=codec,
            sample_rate=sample_rate,
            channels=channels,
            start_time=time.time()
        )

        print(f"[AudioWriter] Initialized: {self.filepath}")

    def open(self):
        """Open WAV file for writing."""
        if self._is_open:
            return

        try:
            self._wav_file = wave.open(self.filepath, 'wb')
            self._wav_file.setnchannels(self.channels)
            self._wav_file.setsampwidth(2)  # 16-bit PCM (2 bytes per sample)
            self._wav_file.setframerate(self.sample_rate)
            self._is_open = True
            print(f"[AudioWriter] Opened: {self.filepath}")

        except Exception as e:
            print(f"[AudioWriter] Failed to open file: {e}")
            raise

    def write(self, pcm_samples: np.ndarray):
        """Write PCM samples to WAV file.

        Args:
            pcm_samples: Numpy array of int16 PCM samples
        """
        if not self._is_open:
            self.open()

        if self._wav_file is None:
            return

        try:
            # Convert numpy array to bytes
            pcm_bytes = pcm_samples.astype(np.int16).tobytes()

            # Write to WAV file
            self._wav_file.writeframes(pcm_bytes)

            # Update metadata
            self.metadata.total_samples += len(pcm_samples)
            self.metadata.total_bytes += len(pcm_bytes)

            # Check file size for rotation
            if self._should_rotate():
                self._rotate_file()

        except Exception as e:
            print(f"[AudioWriter] Write error: {e}")

    def close(self):
        """Close WAV file and write metadata."""
        if not self._is_open:
            return

        try:
            if self._wav_file:
                self._wav_file.close()
                self._wav_file = None

            # Finalize metadata
            self.metadata.end_time = time.time()
            self.metadata.duration = self.metadata.end_time - self.metadata.start_time

            # Write metadata JSON
            self._write_metadata()

            self._is_open = False
            print(f"[AudioWriter] Closed: {self.filepath}")
            print(f"  Duration: {self.metadata.duration:.1f}s")
            print(f"  Samples: {self.metadata.total_samples}")
            print(f"  Size: {self.metadata.total_bytes / 1024:.1f} KB")

        except Exception as e:
            print(f"[AudioWriter] Close error: {e}")

    def _should_rotate(self) -> bool:
        """Check if file should be rotated based on size.

        Returns:
            True if file should be rotated
        """
        if not os.path.exists(self.filepath):
            return False

        file_size_mb = os.path.getsize(self.filepath) / (1024 * 1024)
        return file_size_mb >= self.max_file_size_mb

    def _rotate_file(self):
        """Rotate to a new file."""
        print(f"[AudioWriter] Rotating file (size limit reached)")

        # Close current file
        if self._wav_file:
            self._wav_file.close()
            self._write_metadata()

        # Generate new filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_index = 1
        while True:
            filename = f"{timestamp}_{self.session_id}_{self.codec}_{file_index}.wav"
            filepath = os.path.join(self.output_dir, filename)
            if not os.path.exists(filepath):
                break
            file_index += 1

        self.filename = filename
        self.filepath = filepath
        self.metadata_filepath = filepath.replace(".wav", ".json")

        # Reset metadata for new file
        self.metadata = AudioMetadata(
            session_id=self.session_id,
            codec=self.codec,
            sample_rate=self.sample_rate,
            channels=self.channels,
            start_time=time.time(),
            packets_received=self.metadata.packets_received,
            packets_lost=self.metadata.packets_lost
        )

        # Open new file
        self._wav_file = wave.open(self.filepath, 'wb')
        self._wav_file.setnchannels(self.channels)
        self._wav_file.setsampwidth(2)
        self._wav_file.setframerate(self.sample_rate)

        print(f"[AudioWriter] Rotated to: {self.filepath}")

    def _write_metadata(self):
        """Write metadata JSON file."""
        try:
            metadata_dict = asdict(self.metadata)
            with open(self.metadata_filepath, 'w') as f:
                json.dump(metadata_dict, f, indent=2)

        except Exception as e:
            print(f"[AudioWriter] Failed to write metadata: {e}")

    def update_stats(self, packets_received: int, packets_lost: int):
        """Update packet statistics.

        Args:
            packets_received: Total packets received
            packets_lost: Total packets lost
        """
        self.metadata.packets_received = packets_received
        self.metadata.packets_lost = packets_lost

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
