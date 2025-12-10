"""Audio codec decoders for military voice codecs.

Supports G.711 (μ-law and A-law), Opus, AMR, and MELPe (external binary).
"""

import audioop
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional


class AudioDecoder(ABC):
    """Abstract base class for audio decoders."""

    @abstractmethod
    def decode(self, rtp_payload: bytes) -> np.ndarray:
        """Decode RTP payload to PCM samples.

        Args:
            rtp_payload: Raw RTP payload bytes

        Returns:
            Numpy array of PCM samples (int16)
        """
        pass

    @abstractmethod
    def get_sample_rate(self) -> int:
        """Get decoder sample rate in Hz."""
        pass

    @abstractmethod
    def get_channels(self) -> int:
        """Get number of audio channels."""
        pass


class G711Decoder(AudioDecoder):
    """G.711 codec decoder (μ-law and A-law)."""

    def __init__(self, law: str = "ulaw"):
        """Initialize G.711 decoder.

        Args:
            law: Either "ulaw" (μ-law) or "alaw" (A-law)
        """
        if law not in ("ulaw", "alaw"):
            raise ValueError(f"Invalid law: {law}, must be 'ulaw' or 'alaw'")

        self.law = law
        self._sample_rate = 8000
        self._channels = 1

    def decode(self, rtp_payload: bytes) -> np.ndarray:
        """Decode G.711 payload to PCM.

        Args:
            rtp_payload: G.711 encoded bytes

        Returns:
            PCM samples as int16 numpy array
        """
        try:
            if self.law == "ulaw":
                # μ-law decoding
                pcm_bytes = audioop.ulaw2lin(rtp_payload, 2)  # 2 bytes per sample (int16)
            else:
                # A-law decoding
                pcm_bytes = audioop.alaw2lin(rtp_payload, 2)

            # Convert bytes to numpy array
            pcm_array = np.frombuffer(pcm_bytes, dtype=np.int16)
            return pcm_array

        except Exception as e:
            print(f"[G711Decoder] Decode error: {e}")
            # Return silence on error
            return np.zeros(len(rtp_payload), dtype=np.int16)

    def get_sample_rate(self) -> int:
        return self._sample_rate

    def get_channels(self) -> int:
        return self._channels


class OpusDecoder(AudioDecoder):
    """Opus codec decoder."""

    def __init__(self, sample_rate: int = 48000, channels: int = 1):
        """Initialize Opus decoder.

        Args:
            sample_rate: Sample rate in Hz (8000, 12000, 16000, 24000, or 48000)
            channels: Number of channels (1 or 2)
        """
        self._sample_rate = sample_rate
        self._channels = channels
        self._decoder = None

        # Try to import opuslib
        try:
            import opuslib
            self._decoder = opuslib.Decoder(sample_rate, channels)
            print(f"[OpusDecoder] Initialized: {sample_rate}Hz, {channels} channel(s)")
        except ImportError:
            print("[OpusDecoder] WARNING: opuslib not available, Opus decoding disabled")
            print("  Install with: pip install opuslib")

    def decode(self, rtp_payload: bytes) -> np.ndarray:
        """Decode Opus payload to PCM.

        Args:
            rtp_payload: Opus encoded bytes

        Returns:
            PCM samples as int16 numpy array
        """
        if self._decoder is None:
            # Return silence if decoder not available
            print("[OpusDecoder] Decoder not available")
            return np.zeros(960, dtype=np.int16)  # 20ms at 48kHz

        try:
            # Decode Opus frame
            pcm_bytes = self._decoder.decode(rtp_payload, 960)  # 20ms frame
            pcm_array = np.frombuffer(pcm_bytes, dtype=np.int16)
            return pcm_array

        except Exception as e:
            print(f"[OpusDecoder] Decode error: {e}")
            return np.zeros(960, dtype=np.int16)

    def get_sample_rate(self) -> int:
        return self._sample_rate

    def get_channels(self) -> int:
        return self._channels


class AMRDecoder(AudioDecoder):
    """AMR (Adaptive Multi-Rate) codec decoder."""

    def __init__(self):
        """Initialize AMR decoder."""
        self._sample_rate = 8000
        self._channels = 1

        # Check if pydub and FFmpeg are available
        self._available = False
        try:
            import pydub
            self._available = True
            print("[AMRDecoder] Initialized (via pydub + FFmpeg)")
        except ImportError:
            print("[AMRDecoder] WARNING: pydub not available, AMR decoding disabled")
            print("  Install with: pip install pydub")
            print("  Requires FFmpeg with AMR support")

    def decode(self, rtp_payload: bytes) -> np.ndarray:
        """Decode AMR payload to PCM.

        Args:
            rtp_payload: AMR encoded bytes

        Returns:
            PCM samples as int16 numpy array
        """
        if not self._available:
            return np.zeros(160, dtype=np.int16)  # 20ms at 8kHz

        try:
            from pydub import AudioSegment
            import io

            # AMR header
            amr_data = b'#!AMR\n' + rtp_payload

            # Decode using FFmpeg
            audio = AudioSegment.from_file(io.BytesIO(amr_data), format="amr")
            pcm_array = np.array(audio.get_array_of_samples(), dtype=np.int16)
            return pcm_array

        except Exception as e:
            print(f"[AMRDecoder] Decode error: {e}")
            return np.zeros(160, dtype=np.int16)

    def get_sample_rate(self) -> int:
        return self._sample_rate

    def get_channels(self) -> int:
        return self._channels


class MELPeDecoder(AudioDecoder):
    """MELPe (Mixed Excitation Linear Prediction enhanced) decoder.

    NATO standard military codec. Requires external binary as no Python implementation exists.
    """

    def __init__(self, melpe_binary_path: Optional[str] = None):
        """Initialize MELPe decoder.

        Args:
            melpe_binary_path: Path to MELPe decoder binary (optional)
        """
        self._sample_rate = 8000
        self._channels = 1
        self._binary_path = melpe_binary_path or "/usr/local/bin/melpe_decoder"

        # Check if binary exists
        import os
        self._available = os.path.exists(self._binary_path)

        if self._available:
            print(f"[MELPeDecoder] Initialized: {self._binary_path}")
        else:
            print(f"[MELPeDecoder] WARNING: MELPe binary not found at {self._binary_path}")
            print("  MELPe decoding disabled. Install MELPe decoder binary to enable.")

    def decode(self, rtp_payload: bytes) -> np.ndarray:
        """Decode MELPe payload to PCM.

        Args:
            rtp_payload: MELPe encoded bytes

        Returns:
            PCM samples as int16 numpy array
        """
        if not self._available:
            return np.zeros(180, dtype=np.int16)  # 22.5ms at 8kHz

        try:
            import subprocess
            import tempfile

            # Write payload to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".melpe") as f_in:
                f_in.write(rtp_payload)
                input_path = f_in.name

            # Output file
            output_path = input_path + ".pcm"

            # Call external decoder
            subprocess.run(
                [self._binary_path, input_path, output_path],
                check=True,
                capture_output=True
            )

            # Read decoded PCM
            with open(output_path, "rb") as f_out:
                pcm_bytes = f_out.read()

            # Cleanup
            import os
            os.unlink(input_path)
            os.unlink(output_path)

            # Convert to numpy array
            pcm_array = np.frombuffer(pcm_bytes, dtype=np.int16)
            return pcm_array

        except Exception as e:
            print(f"[MELPeDecoder] Decode error: {e}")
            return np.zeros(180, dtype=np.int16)

    def get_sample_rate(self) -> int:
        return self._sample_rate

    def get_channels(self) -> int:
        return self._channels


def get_decoder(codec: str, **kwargs) -> Optional[AudioDecoder]:
    """Factory function to get decoder by codec name.

    Args:
        codec: Codec name ("g711_ulaw", "g711_alaw", "opus", "amr", "melpe")
        **kwargs: Additional arguments passed to decoder constructor

    Returns:
        AudioDecoder instance or None if codec not supported
    """
    codec = codec.lower()

    if codec in ("g711", "g711_ulaw", "pcmu"):
        return G711Decoder(law="ulaw")
    elif codec in ("g711_alaw", "pcma"):
        return G711Decoder(law="alaw")
    elif codec == "opus":
        return OpusDecoder(**kwargs)
    elif codec == "amr":
        return AMRDecoder()
    elif codec == "melpe":
        return MELPeDecoder(**kwargs)
    else:
        print(f"[AudioDecoders] Unknown codec: {codec}")
        return None
