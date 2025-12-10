"""RTSP session lifecycle management.

Handles session creation, state tracking, timeout detection, and cleanup.
"""

import time
import threading
import uuid
from enum import Enum
from typing import Dict, Optional
from dataclasses import dataclass, field


class RTSPState(Enum):
    """RTSP session states."""
    INIT = "init"
    READY = "ready"
    PLAYING = "playing"
    RECORDING = "recording"


@dataclass
class RTSPSession:
    """RTSP session metadata."""
    session_id: str
    client_address: tuple
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    state: RTSPState = RTSPState.INIT

    # Transport details
    client_rtp_port: Optional[int] = None
    client_rtcp_port: Optional[int] = None
    server_rtp_port: Optional[int] = None
    server_rtcp_port: Optional[int] = None

    # Audio details
    codec: Optional[str] = None
    sample_rate: Optional[int] = None
    channels: int = 1

    # Statistics
    packets_received: int = 0
    packets_lost: int = 0
    bytes_received: int = 0

    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = time.time()

    def is_expired(self, timeout: int = 60) -> bool:
        """Check if session has expired."""
        return (time.time() - self.last_activity) > timeout


class RTSPSessionManager:
    """Manages RTSP session lifecycle."""

    def __init__(self, session_timeout: int = 60):
        """Initialize session manager.

        Args:
            session_timeout: Session timeout in seconds (default 60)
        """
        self.session_timeout = session_timeout
        self.sessions: Dict[str, RTSPSession] = {}
        self._lock = threading.Lock()
        self._cleanup_thread: Optional[threading.Thread] = None
        self._running = False

    def start(self):
        """Start session manager and cleanup thread."""
        if self._running:
            return

        self._running = True
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True
        )
        self._cleanup_thread.start()

    def stop(self):
        """Stop session manager and cleanup thread."""
        self._running = False
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=2.0)

    def create_session(self, client_address: tuple) -> RTSPSession:
        """Create a new RTSP session.

        Args:
            client_address: (host, port) tuple of client

        Returns:
            New RTSPSession object
        """
        session_id = str(uuid.uuid4()).replace('-', '')[:16]
        session = RTSPSession(
            session_id=session_id,
            client_address=client_address
        )

        with self._lock:
            self.sessions[session_id] = session

        print(f"[SessionManager] Created session {session_id} for {client_address}")
        return session

    def get_session(self, session_id: str) -> Optional[RTSPSession]:
        """Get session by ID.

        Args:
            session_id: Session identifier

        Returns:
            RTSPSession or None if not found
        """
        with self._lock:
            return self.sessions.get(session_id)

    def update_session(self, session_id: str, **kwargs):
        """Update session attributes.

        Args:
            session_id: Session identifier
            **kwargs: Attributes to update
        """
        with self._lock:
            session = self.sessions.get(session_id)
            if session:
                for key, value in kwargs.items():
                    if hasattr(session, key):
                        setattr(session, key, value)
                session.update_activity()

    def remove_session(self, session_id: str) -> bool:
        """Remove a session.

        Args:
            session_id: Session identifier

        Returns:
            True if session was removed, False if not found
        """
        with self._lock:
            if session_id in self.sessions:
                session = self.sessions.pop(session_id)
                print(f"[SessionManager] Removed session {session_id}")
                print(f"  Duration: {time.time() - session.created_at:.1f}s")
                print(f"  Packets: {session.packets_received} received, "
                      f"{session.packets_lost} lost")
                print(f"  Bytes: {session.bytes_received}")
                return True
        return False

    def get_active_sessions(self) -> Dict[str, RTSPSession]:
        """Get all active sessions.

        Returns:
            Dictionary of session_id -> RTSPSession
        """
        with self._lock:
            return self.sessions.copy()

    def get_session_count(self) -> int:
        """Get number of active sessions."""
        with self._lock:
            return len(self.sessions)

    def _cleanup_loop(self):
        """Background thread to cleanup expired sessions."""
        while self._running:
            try:
                self._cleanup_expired_sessions()
            except Exception as e:
                print(f"[SessionManager] Cleanup error: {e}")

            # Check every 10 seconds
            for _ in range(10):
                if not self._running:
                    break
                time.sleep(1)

    def _cleanup_expired_sessions(self):
        """Remove expired sessions."""
        expired = []

        with self._lock:
            for session_id, session in self.sessions.items():
                if session.is_expired(self.session_timeout):
                    expired.append(session_id)

        for session_id in expired:
            print(f"[SessionManager] Session {session_id} expired (timeout)")
            self.remove_session(session_id)
