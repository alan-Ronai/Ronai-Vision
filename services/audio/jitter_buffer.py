"""Jitter buffer for RTP packet reordering and timing.

Handles out-of-order packets, maintains timing, and provides statistics.
"""

import time
import threading
from typing import Optional, List, Tuple
from dataclasses import dataclass
from collections import deque


@dataclass
class RTPPacket:
    """RTP packet data."""
    sequence_number: int
    timestamp: int
    payload_type: int
    payload: bytes
    received_at: float


class JitterBuffer:
    """Buffer for reordering and timing RTP packets."""

    def __init__(self, buffer_ms: int = 100, max_size: int = 200):
        """Initialize jitter buffer.

        Args:
            buffer_ms: Buffer depth in milliseconds (default 100ms)
            max_size: Maximum number of packets to buffer (default 200)
        """
        self.buffer_ms = buffer_ms
        self.max_size = max_size

        self._buffer: deque = deque(maxlen=max_size)
        self._lock = threading.Lock()

        # Sequence tracking
        self._last_sequence: Optional[int] = None
        self._expected_sequence: Optional[int] = None

        # Statistics
        self.packets_received = 0
        self.packets_dropped = 0
        self.packets_reordered = 0
        self.packets_late = 0
        self.packets_duplicate = 0

    def insert(self, packet: RTPPacket) -> bool:
        """Insert packet into buffer.

        Args:
            packet: RTP packet to insert

        Returns:
            True if packet was inserted, False if dropped
        """
        with self._lock:
            self.packets_received += 1

            # Initialize sequence tracking
            if self._expected_sequence is None:
                self._expected_sequence = packet.sequence_number
                self._last_sequence = packet.sequence_number

            # Check for duplicate
            if self._is_duplicate(packet.sequence_number):
                self.packets_duplicate += 1
                return False

            # Check if packet is late (older than all buffered packets)
            if len(self._buffer) > 0 and packet.sequence_number < self._buffer[0].sequence_number:
                # Allow some tolerance for reordering
                if self._sequence_diff(self._buffer[0].sequence_number, packet.sequence_number) > 100:
                    self.packets_late += 1
                    return False

            # Insert packet in sorted order by sequence number
            inserted_at = len(self._buffer)
            for i, buffered_packet in enumerate(self._buffer):
                if packet.sequence_number < buffered_packet.sequence_number:
                    self._buffer.insert(i, packet)
                    inserted_at = i
                    if i > 0:
                        self.packets_reordered += 1
                    break
            else:
                self._buffer.append(packet)

            # Trim buffer if too large
            if len(self._buffer) > self.max_size:
                dropped = self._buffer.popleft()
                self.packets_dropped += 1
                print(f"[JitterBuffer] Buffer full, dropped seq {dropped.sequence_number}")

            return True

    def pop(self) -> Optional[RTPPacket]:
        """Pop the next packet ready for decoding.

        Returns:
            Next RTP packet or None if buffer is empty or not ready
        """
        with self._lock:
            if len(self._buffer) == 0:
                return None

            # Wait for buffer to fill to minimum threshold (buffer_ms / 2)
            min_packets = max(1, int(self.buffer_ms / 20 / 2))  # Assume ~20ms per packet
            if len(self._buffer) < min_packets:
                return None

            # Pop oldest packet
            packet = self._buffer.popleft()
            self._last_sequence = packet.sequence_number
            self._expected_sequence = (packet.sequence_number + 1) & 0xFFFF

            return packet

    def flush(self) -> List[RTPPacket]:
        """Flush all packets from buffer.

        Returns:
            List of all buffered packets
        """
        with self._lock:
            packets = list(self._buffer)
            self._buffer.clear()
            return packets

    def get_stats(self) -> dict:
        """Get buffer statistics.

        Returns:
            Dictionary of statistics
        """
        with self._lock:
            loss_rate = 0.0
            if self.packets_received > 0:
                loss_rate = self.packets_dropped / self.packets_received

            return {
                "buffer_size": len(self._buffer),
                "max_size": self.max_size,
                "packets_received": self.packets_received,
                "packets_dropped": self.packets_dropped,
                "packets_reordered": self.packets_reordered,
                "packets_late": self.packets_late,
                "packets_duplicate": self.packets_duplicate,
                "loss_rate": loss_rate,
            }

    def reset(self):
        """Reset buffer and statistics."""
        with self._lock:
            self._buffer.clear()
            self._last_sequence = None
            self._expected_sequence = None

            self.packets_received = 0
            self.packets_dropped = 0
            self.packets_reordered = 0
            self.packets_late = 0
            self.packets_duplicate = 0

    def _is_duplicate(self, sequence_number: int) -> bool:
        """Check if sequence number is a duplicate.

        Args:
            sequence_number: Sequence number to check

        Returns:
            True if duplicate
        """
        if self._last_sequence is None:
            return False

        for packet in self._buffer:
            if packet.sequence_number == sequence_number:
                return True

        return False

    @staticmethod
    def _sequence_diff(seq1: int, seq2: int) -> int:
        """Calculate difference between sequence numbers (handles wraparound).

        Args:
            seq1: First sequence number
            seq2: Second sequence number

        Returns:
            Absolute difference
        """
        diff = (seq1 - seq2) & 0xFFFF
        if diff > 32768:
            diff = 65536 - diff
        return diff
