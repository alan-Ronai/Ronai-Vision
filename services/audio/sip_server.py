"""SIP + RTP server for VoIP communication.

Handles SIP signaling and RTP media streams for voice communication.
Supports INVITE, ACK, BYE for basic call setup and teardown.
"""

import socket
import threading
import time
import re
import logging
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime

from .raw_rtp_receiver import RawRTPReceiver

logger = logging.getLogger(__name__)


@dataclass
class SIPCall:
    """Represents an active SIP call."""

    call_id: str
    from_uri: str
    to_uri: str
    remote_addr: Tuple[str, int]
    rtp_port: int
    rtp_receiver: Optional[RawRTPReceiver] = None
    state: str = "ringing"  # ringing, active, terminated
    created_at: datetime = None


class SIPServer:
    """SIP server for VoIP call signaling.

    Handles basic SIP methods: INVITE, ACK, BYE, CANCEL.
    Creates RTP receivers for media streams.
    """

    def __init__(
        self,
        sip_host: str = "0.0.0.0",
        sip_port: int = 5060,
        rtp_base_port: int = 10000,
        storage_path: str = "audio_storage/recordings",
    ):
        """Initialize SIP server.

        Args:
            sip_host: SIP server bind address
            sip_port: SIP server port (default 5060, standard SIP port)
            rtp_base_port: Base port for RTP media streams
            storage_path: Path to store audio recordings
        """
        self.sip_host = sip_host
        self.sip_port = sip_port
        self.rtp_base_port = rtp_base_port
        self.storage_path = storage_path

        # SIP socket
        self._socket: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None

        # Active calls
        self._calls: Dict[str, SIPCall] = {}
        self._next_rtp_port = rtp_base_port

        # State
        self._running = False

        logger.info(f"SIPServer initialized")
        logger.info(f"  SIP: {sip_host}:{sip_port}")
        logger.info(f"  RTP base port: {rtp_base_port}")

    def start(self):
        """Start SIP server."""
        if self._running:
            logger.warning("SIP server already running")
            return

        logger.info("Starting SIP server...")

        try:
            # Create UDP socket for SIP
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._socket.bind((self.sip_host, self.sip_port))
            self._socket.settimeout(1.0)

            logger.info(f"SIP server listening on {self.sip_host}:{self.sip_port}")

        except Exception as e:
            logger.error(f"Failed to bind SIP socket: {e}")
            raise

        # Start SIP thread
        self._running = True
        self._thread = threading.Thread(
            target=self._sip_loop, daemon=True, name="SIP-Server"
        )
        self._thread.start()

        logger.info("SIP server started")

    def stop(self):
        """Stop SIP server."""
        if not self._running:
            return

        logger.info("Stopping SIP server...")
        self._running = False

        # Stop all active calls
        for call_id in list(self._calls.keys()):
            self._terminate_call(call_id)

        # Close socket
        if self._socket:
            try:
                self._socket.close()
            except:
                pass

        # Wait for thread
        if self._thread:
            self._thread.join(timeout=2.0)

        logger.info("SIP server stopped")

    def get_stats(self) -> dict:
        """Get server statistics."""
        return {
            "running": self._running,
            "active_calls": len(self._calls),
            "calls": {
                call_id: {
                    "state": call.state,
                    "from": call.from_uri,
                    "to": call.to_uri,
                    "rtp_port": call.rtp_port,
                }
                for call_id, call in self._calls.items()
            },
        }

    def _sip_loop(self):
        """Main SIP message handling loop."""
        logger.info("SIP loop started")

        while self._running:
            try:
                # Receive SIP message
                data, addr = self._socket.recvfrom(4096)
                message = data.decode("utf-8", errors="ignore")

                logger.debug(f"Received SIP message from {addr}")

                # Handle SIP message
                self._handle_sip_message(message, addr)

            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    logger.error(f"SIP loop error: {e}", exc_info=True)

        logger.info("SIP loop stopped")

    def _handle_sip_message(self, message: str, addr: Tuple[str, int]):
        """Parse and handle SIP message.

        Args:
            message: SIP message text
            addr: Source address (host, port)
        """
        lines = message.strip().split("\r\n")
        if not lines:
            return

        # Parse request line
        request_line = lines[0]

        # Parse headers
        headers = {}
        for line in lines[1:]:
            if ":" in line:
                key, value = line.split(":", 1)
                headers[key.strip()] = value.strip()

        # Dispatch based on method
        if request_line.startswith("INVITE"):
            self._handle_invite(request_line, headers, addr)
        elif request_line.startswith("ACK"):
            self._handle_ack(request_line, headers, addr)
        elif request_line.startswith("BYE"):
            self._handle_bye(request_line, headers, addr)
        elif request_line.startswith("CANCEL"):
            self._handle_cancel(request_line, headers, addr)
        elif request_line.startswith("OPTIONS"):
            self._handle_options(request_line, headers, addr)

    def _handle_invite(
        self, request_line: str, headers: Dict[str, str], addr: Tuple[str, int]
    ):
        """Handle SIP INVITE request."""
        call_id = headers.get("Call-ID", f"call_{int(time.time())}")
        from_uri = headers.get("From", "Unknown")
        to_uri = headers.get("To", "Unknown")

        logger.info(f"INVITE from {from_uri} to {to_uri} (Call-ID: {call_id})")

        # Allocate RTP port
        rtp_port = self._allocate_rtp_port()

        # Create call
        call = SIPCall(
            call_id=call_id,
            from_uri=from_uri,
            to_uri=to_uri,
            remote_addr=addr,
            rtp_port=rtp_port,
            state="ringing",
            created_at=datetime.now(),
        )
        self._calls[call_id] = call

        # Send 180 Ringing
        self._send_response(addr, "180 Ringing", headers)

        # Send 200 OK with SDP
        self._send_ok_with_sdp(addr, headers, rtp_port)

        # Start RTP receiver
        self._start_rtp_receiver(call)

    def _handle_ack(
        self, request_line: str, headers: Dict[str, str], addr: Tuple[str, int]
    ):
        """Handle SIP ACK request."""
        call_id = headers.get("Call-ID")
        if call_id in self._calls:
            self._calls[call_id].state = "active"
            logger.info(f"Call {call_id} is now active")

    def _handle_bye(
        self, request_line: str, headers: Dict[str, str], addr: Tuple[str, int]
    ):
        """Handle SIP BYE request."""
        call_id = headers.get("Call-ID")

        logger.info(f"BYE for call {call_id}")

        # Send 200 OK
        self._send_response(addr, "200 OK", headers)

        # Terminate call
        if call_id:
            self._terminate_call(call_id)

    def _handle_cancel(
        self, request_line: str, headers: Dict[str, str], addr: Tuple[str, int]
    ):
        """Handle SIP CANCEL request."""
        call_id = headers.get("Call-ID")

        logger.info(f"CANCEL for call {call_id}")

        # Send 200 OK
        self._send_response(addr, "200 OK", headers)

        # Terminate call
        if call_id:
            self._terminate_call(call_id)

    def _handle_options(
        self, request_line: str, headers: Dict[str, str], addr: Tuple[str, int]
    ):
        """Handle SIP OPTIONS request."""
        self._send_response(
            addr, "200 OK", headers, allow="INVITE, ACK, BYE, CANCEL, OPTIONS"
        )

    def _send_response(
        self,
        addr: Tuple[str, int],
        status: str,
        request_headers: Dict[str, str],
        allow: str = None,
    ):
        """Send SIP response.

        Args:
            addr: Destination address
            status: Status line (e.g., "200 OK")
            request_headers: Headers from request (for Via, Call-ID, etc.)
            allow: Optional Allow header value
        """
        response = f"SIP/2.0 {status}\r\n"
        response += f"Via: {request_headers.get('Via', 'SIP/2.0/UDP unknown')}\r\n"
        response += f"From: {request_headers.get('From', 'Unknown')}\r\n"
        response += f"To: {request_headers.get('To', 'Unknown')}\r\n"
        response += f"Call-ID: {request_headers.get('Call-ID', 'unknown')}\r\n"
        response += f"CSeq: {request_headers.get('CSeq', '1 INVITE')}\r\n"

        if allow:
            response += f"Allow: {allow}\r\n"

        response += f"Content-Length: 0\r\n"
        response += "\r\n"

        try:
            self._socket.sendto(response.encode("utf-8"), addr)
            logger.debug(f"Sent SIP response: {status} to {addr}")
        except Exception as e:
            logger.error(f"Failed to send SIP response: {e}")

    def _send_ok_with_sdp(
        self, addr: Tuple[str, int], request_headers: Dict[str, str], rtp_port: int
    ):
        """Send 200 OK with SDP for media negotiation.

        Args:
            addr: Destination address
            request_headers: Headers from INVITE request
            rtp_port: RTP port allocated for this call
        """
        # Create SDP
        sdp = (
            f"v=0\r\n"
            f"o=ronai {int(time.time())} {int(time.time())} IN IP4 {self.sip_host}\r\n"
            f"s=Ronai Voice\r\n"
            f"c=IN IP4 {self.sip_host}\r\n"
            f"t=0 0\r\n"
            f"m=audio {rtp_port} RTP/AVP 0 8\r\n"
            f"a=rtpmap:0 PCMU/8000\r\n"
            f"a=rtpmap:8 PCMA/8000\r\n"
        )

        response = f"SIP/2.0 200 OK\r\n"
        response += f"Via: {request_headers.get('Via', 'SIP/2.0/UDP unknown')}\r\n"
        response += f"From: {request_headers.get('From', 'Unknown')}\r\n"
        response += f"To: {request_headers.get('To', 'Unknown')}\r\n"
        response += f"Call-ID: {request_headers.get('Call-ID', 'unknown')}\r\n"
        response += f"CSeq: {request_headers.get('CSeq', '1 INVITE')}\r\n"
        response += f"Content-Type: application/sdp\r\n"
        response += f"Content-Length: {len(sdp)}\r\n"
        response += "\r\n"
        response += sdp

        try:
            self._socket.sendto(response.encode("utf-8"), addr)
            logger.info(f"Sent 200 OK with SDP (RTP port {rtp_port}) to {addr}")
        except Exception as e:
            logger.error(f"Failed to send 200 OK: {e}")

    def _allocate_rtp_port(self) -> int:
        """Allocate next available RTP port."""
        port = self._next_rtp_port
        self._next_rtp_port += 2  # RTP uses even ports, RTCP uses odd
        return port

    def _start_rtp_receiver(self, call: SIPCall):
        """Start RTP receiver for call.

        Args:
            call: SIPCall object
        """
        try:
            receiver = RawRTPReceiver(
                listen_host="0.0.0.0",
                listen_port=call.rtp_port,
                storage_path=self.storage_path,
                auto_detect_codec=True,
            )
            receiver.start()
            call.rtp_receiver = receiver

            logger.info(
                f"Started RTP receiver for call {call.call_id} on port {call.rtp_port}"
            )

        except Exception as e:
            logger.error(f"Failed to start RTP receiver: {e}")

    def _terminate_call(self, call_id: str):
        """Terminate call and cleanup resources.

        Args:
            call_id: Call identifier
        """
        call = self._calls.get(call_id)
        if not call:
            return

        logger.info(f"Terminating call {call_id}")

        # Stop RTP receiver
        if call.rtp_receiver:
            call.rtp_receiver.stop()

        # Update state
        call.state = "terminated"

        # Remove from active calls
        del self._calls[call_id]

        logger.info(f"Call {call_id} terminated")
