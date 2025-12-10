"""RTP/RTSP Audio Server implementation.

Receives audio streams via RTSP/RTP protocol, decodes them, and saves to disk.
Supports military codecs: G.711, Opus, AMR, MELPe.
"""

import socket
import struct
import threading
import time
from typing import Optional, Dict
from urllib.parse import urlparse

from .session_manager import RTSPSessionManager, RTSPState
from .jitter_buffer import JitterBuffer, RTPPacket
from .audio_decoders import get_decoder, AudioDecoder
from .audio_writer import AudioWriter


class RTPAudioServer:
    """RTP/RTSP audio server for receiving military audio streams."""

    def __init__(
        self,
        rtsp_host: str = "0.0.0.0",
        rtsp_port: int = 8554,  # Use 8554 instead of 554 to avoid needing root
        rtp_base_port: int = 5004,
        storage_path: str = "audio_storage/recordings",
        session_timeout: int = 60,
        jitter_buffer_ms: int = 100
    ):
        """Initialize RTP/RTSP audio server.

        Args:
            rtsp_host: RTSP server bind address (default "0.0.0.0")
            rtsp_port: RTSP server port (default 8554, standard is 554 but requires root)
            rtp_base_port: Base port for RTP (default 5004)
            storage_path: Path to store audio recordings
            session_timeout: Session timeout in seconds (default 60)
            jitter_buffer_ms: Jitter buffer depth in ms (default 100)
        """
        self.rtsp_host = rtsp_host
        self.rtsp_port = rtsp_port
        self.rtp_base_port = rtp_base_port
        self.storage_path = storage_path
        self.session_timeout = session_timeout
        self.jitter_buffer_ms = jitter_buffer_ms

        # Session manager
        self.session_manager = RTSPSessionManager(session_timeout=session_timeout)

        # RTSP server socket
        self._rtsp_socket: Optional[socket.socket] = None
        self._rtsp_thread: Optional[threading.Thread] = None

        # RTP receivers (per session)
        self._rtp_receivers: Dict[str, RTPReceiver] = {}

        # Server state
        self._running = False

        print(f"[RTPAudioServer] Initialized")
        print(f"  RTSP: {rtsp_host}:{rtsp_port}")
        print(f"  RTP base port: {rtp_base_port}")
        print(f"  Storage: {storage_path}")

    def start(self):
        """Start RTSP server."""
        if self._running:
            print("[RTPAudioServer] Already running")
            return

        print("[RTPAudioServer] Starting...")

        # Start session manager
        self.session_manager.start()

        # Create RTSP socket
        try:
            self._rtsp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._rtsp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._rtsp_socket.bind((self.rtsp_host, self.rtsp_port))
            self._rtsp_socket.listen(5)
            print(f"[RTPAudioServer] RTSP server listening on {self.rtsp_host}:{self.rtsp_port}")

        except Exception as e:
            print(f"[RTPAudioServer] Failed to bind RTSP socket: {e}")
            raise

        # Start RTSP server thread
        self._running = True
        self._rtsp_thread = threading.Thread(target=self._rtsp_server_loop, daemon=True)
        self._rtsp_thread.start()

        print("[RTPAudioServer] Started")

    def stop(self):
        """Stop RTSP server."""
        if not self._running:
            return

        print("[RTPAudioServer] Stopping...")
        self._running = False

        # Stop all RTP receivers
        for receiver in list(self._rtp_receivers.values()):
            receiver.stop()

        # Close RTSP socket
        if self._rtsp_socket:
            try:
                self._rtsp_socket.close()
            except:
                pass

        # Stop session manager
        self.session_manager.stop()

        # Wait for thread
        if self._rtsp_thread:
            self._rtsp_thread.join(timeout=2.0)

        print("[RTPAudioServer] Stopped")

    def get_status(self) -> dict:
        """Get server status.

        Returns:
            Status dictionary
        """
        return {
            "running": self._running,
            "rtsp_host": self.rtsp_host,
            "rtsp_port": self.rtsp_port,
            "rtp_base_port": self.rtp_base_port,
            "active_sessions": self.session_manager.get_session_count(),
            "active_receivers": len(self._rtp_receivers)
        }

    def _rtsp_server_loop(self):
        """Main RTSP server loop (accepts connections)."""
        while self._running:
            try:
                # Accept connection with timeout
                self._rtsp_socket.settimeout(1.0)
                client_socket, client_address = self._rtsp_socket.accept()

                print(f"[RTPAudioServer] RTSP connection from {client_address}")

                # Handle client in separate thread
                client_thread = threading.Thread(
                    target=self._handle_rtsp_client,
                    args=(client_socket, client_address),
                    daemon=True
                )
                client_thread.start()

            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    print(f"[RTPAudioServer] RTSP accept error: {e}")
                break

    def _handle_rtsp_client(self, client_socket: socket.socket, client_address: tuple):
        """Handle RTSP client connection.

        Args:
            client_socket: Client socket
            client_address: (host, port) tuple
        """
        session_id = None

        try:
            client_socket.settimeout(30.0)  # 30 second timeout for requests

            while self._running:
                # Read RTSP request
                request_data = client_socket.recv(4096)
                if not request_data:
                    break

                request = request_data.decode('utf-8', errors='ignore')
                print(f"[RTSP] Request from {client_address}:\n{request[:200]}")

                # Parse and handle request
                response, session_id = self._handle_rtsp_request(request, client_address, session_id)

                # Send response
                client_socket.sendall(response.encode('utf-8'))

        except socket.timeout:
            print(f"[RTSP] Client {client_address} timeout")
        except Exception as e:
            print(f"[RTSP] Client {client_address} error: {e}")
        finally:
            client_socket.close()

            # Cleanup session if exists
            if session_id:
                self._cleanup_session(session_id)

    def _handle_rtsp_request(self, request: str, client_address: tuple, session_id: Optional[str]) -> tuple:
        """Parse and handle RTSP request.

        Args:
            request: RTSP request string
            client_address: Client address tuple
            session_id: Existing session ID (if any)

        Returns:
            (response_string, session_id) tuple
        """
        lines = request.strip().split('\r\n')
        if not lines:
            return self._rtsp_error(400, "Bad Request"), session_id

        # Parse request line
        request_line = lines[0].split()
        if len(request_line) < 3:
            return self._rtsp_error(400, "Bad Request"), session_id

        method = request_line[0]
        url = request_line[1]
        version = request_line[2]

        # Parse headers
        headers = {}
        for line in lines[1:]:
            if ':' in line:
                key, value = line.split(':', 1)
                headers[key.strip()] = value.strip()

        # Get CSeq (required)
        cseq = headers.get('CSeq', '1')

        # Handle methods
        if method == "OPTIONS":
            return self._handle_options(cseq), session_id

        elif method == "SETUP":
            response, new_session_id = self._handle_setup(cseq, url, headers, client_address)
            return response, new_session_id

        elif method == "PLAY":
            return self._handle_play(cseq, headers, session_id), session_id

        elif method == "TEARDOWN":
            return self._handle_teardown(cseq, headers, session_id), None

        else:
            return self._rtsp_error(501, "Not Implemented", cseq), session_id

    def _handle_options(self, cseq: str) -> str:
        """Handle OPTIONS request.

        Args:
            cseq: CSeq header value

        Returns:
            RTSP response string
        """
        return (
            f"RTSP/1.0 200 OK\r\n"
            f"CSeq: {cseq}\r\n"
            f"Public: OPTIONS, SETUP, PLAY, TEARDOWN\r\n"
            f"\r\n"
        )

    def _handle_setup(self, cseq: str, url: str, headers: dict, client_address: tuple) -> tuple:
        """Handle SETUP request.

        Args:
            cseq: CSeq header value
            url: Request URL
            headers: Request headers
            client_address: Client address

        Returns:
            (response_string, session_id) tuple
        """
        # Parse Transport header
        transport = headers.get('Transport', '')
        if 'RTP/AVP' not in transport:
            return self._rtsp_error(461, "Unsupported Transport", cseq), None

        # Extract client ports
        client_rtp_port = None
        client_rtcp_port = None

        if 'client_port=' in transport:
            port_str = transport.split('client_port=')[1].split(';')[0]
            ports = port_str.split('-')
            client_rtp_port = int(ports[0])
            client_rtcp_port = int(ports[1]) if len(ports) > 1 else client_rtp_port + 1

        # Create session
        session = self.session_manager.create_session(client_address)

        # Assign server ports (use base port + session offset)
        server_rtp_port = self.rtp_base_port
        server_rtcp_port = server_rtp_port + 1

        # Update session
        self.session_manager.update_session(
            session.session_id,
            state=RTSPState.READY,
            client_rtp_port=client_rtp_port,
            client_rtcp_port=client_rtcp_port,
            server_rtp_port=server_rtp_port,
            server_rtcp_port=server_rtcp_port,
            codec="g711"  # Default to G.711, will be updated when RTP packets arrive
        )

        # Build response
        response = (
            f"RTSP/1.0 200 OK\r\n"
            f"CSeq: {cseq}\r\n"
            f"Session: {session.session_id}\r\n"
            f"Transport: RTP/AVP/UDP;unicast;client_port={client_rtp_port}-{client_rtcp_port};"
            f"server_port={server_rtp_port}-{server_rtcp_port}\r\n"
            f"\r\n"
        )

        return response, session.session_id

    def _handle_play(self, cseq: str, headers: dict, session_id: Optional[str]) -> str:
        """Handle PLAY request.

        Args:
            cseq: CSeq header value
            headers: Request headers
            session_id: Session ID

        Returns:
            RTSP response string
        """
        if not session_id:
            return self._rtsp_error(454, "Session Not Found", cseq)

        session = self.session_manager.get_session(session_id)
        if not session:
            return self._rtsp_error(454, "Session Not Found", cseq)

        # Update session state
        self.session_manager.update_session(session_id, state=RTSPState.PLAYING)

        # Start RTP receiver for this session
        self._start_rtp_receiver(session)

        return (
            f"RTSP/1.0 200 OK\r\n"
            f"CSeq: {cseq}\r\n"
            f"Session: {session_id}\r\n"
            f"\r\n"
        )

    def _handle_teardown(self, cseq: str, headers: dict, session_id: Optional[str]) -> str:
        """Handle TEARDOWN request.

        Args:
            cseq: CSeq header value
            headers: Request headers
            session_id: Session ID

        Returns:
            RTSP response string
        """
        if session_id:
            self._cleanup_session(session_id)

        return (
            f"RTSP/1.0 200 OK\r\n"
            f"CSeq: {cseq}\r\n"
            f"\r\n"
        )

    def _rtsp_error(self, code: int, message: str, cseq: str = "1") -> str:
        """Generate RTSP error response.

        Args:
            code: Error code
            message: Error message
            cseq: CSeq value

        Returns:
            RTSP error response
        """
        return (
            f"RTSP/1.0 {code} {message}\r\n"
            f"CSeq: {cseq}\r\n"
            f"\r\n"
        )

    def _start_rtp_receiver(self, session):
        """Start RTP receiver for session.

        Args:
            session: RTSPSession object
        """
        if session.session_id in self._rtp_receivers:
            print(f"[RTPAudioServer] RTP receiver already exists for session {session.session_id}")
            return

        # Create RTP receiver
        receiver = RTPReceiver(
            session=session,
            storage_path=self.storage_path,
            jitter_buffer_ms=self.jitter_buffer_ms
        )

        self._rtp_receivers[session.session_id] = receiver
        receiver.start()

        print(f"[RTPAudioServer] Started RTP receiver for session {session.session_id}")

    def _cleanup_session(self, session_id: str):
        """Cleanup session and RTP receiver.

        Args:
            session_id: Session ID
        """
        # Stop RTP receiver
        if session_id in self._rtp_receivers:
            receiver = self._rtp_receivers.pop(session_id)
            receiver.stop()

        # Remove session
        self.session_manager.remove_session(session_id)


class RTPReceiver:
    """RTP packet receiver and audio decoder."""

    def __init__(self, session, storage_path: str, jitter_buffer_ms: int = 100):
        """Initialize RTP receiver.

        Args:
            session: RTSPSession object
            storage_path: Path to store audio files
            jitter_buffer_ms: Jitter buffer depth in ms
        """
        self.session = session
        self.storage_path = storage_path
        self.jitter_buffer_ms = jitter_buffer_ms

        # RTP socket
        self._rtp_socket: Optional[socket.socket] = None
        self._rtp_thread: Optional[threading.Thread] = None
        self._decoder_thread: Optional[threading.Thread] = None

        # Jitter buffer
        self.jitter_buffer = JitterBuffer(buffer_ms=jitter_buffer_ms)

        # Decoder
        self.decoder: Optional[AudioDecoder] = None

        # Audio writer
        self.audio_writer: Optional[AudioWriter] = None

        # State
        self._running = False

    def start(self):
        """Start RTP receiver."""
        if self._running:
            return

        try:
            # Create UDP socket for RTP
            self._rtp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._rtp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._rtp_socket.bind(("0.0.0.0", self.session.server_rtp_port))
            self._rtp_socket.settimeout(1.0)

            print(f"[RTPReceiver] Listening on UDP port {self.session.server_rtp_port}")

            # Initialize decoder (default G.711)
            self.decoder = get_decoder(self.session.codec or "g711")

            if self.decoder:
                # Initialize audio writer
                self.audio_writer = AudioWriter(
                    output_dir=self.storage_path,
                    session_id=self.session.session_id,
                    codec=self.session.codec or "g711",
                    sample_rate=self.decoder.get_sample_rate(),
                    channels=self.decoder.get_channels()
                )
                self.audio_writer.open()

            # Start threads
            self._running = True
            self._rtp_thread = threading.Thread(target=self._rtp_receiver_loop, daemon=True)
            self._rtp_thread.start()

            self._decoder_thread = threading.Thread(target=self._decoder_loop, daemon=True)
            self._decoder_thread.start()

            print(f"[RTPReceiver] Started for session {self.session.session_id}")

        except Exception as e:
            print(f"[RTPReceiver] Failed to start: {e}")
            raise

    def stop(self):
        """Stop RTP receiver."""
        if not self._running:
            return

        print(f"[RTPReceiver] Stopping session {self.session.session_id}")
        self._running = False

        # Close socket
        if self._rtp_socket:
            self._rtp_socket.close()

        # Wait for threads
        if self._rtp_thread:
            self._rtp_thread.join(timeout=2.0)
        if self._decoder_thread:
            self._decoder_thread.join(timeout=2.0)

        # Close audio writer
        if self.audio_writer:
            # Update stats
            stats = self.jitter_buffer.get_stats()
            self.audio_writer.update_stats(
                packets_received=stats["packets_received"],
                packets_lost=stats["packets_dropped"]
            )
            self.audio_writer.close()

        print(f"[RTPReceiver] Stopped session {self.session.session_id}")

    def _rtp_receiver_loop(self):
        """Receive RTP packets."""
        while self._running:
            try:
                data, addr = self._rtp_socket.recvfrom(2048)

                # Parse RTP header
                packet = self._parse_rtp_packet(data)
                if packet:
                    # Insert into jitter buffer
                    self.jitter_buffer.insert(packet)

                    # Update session stats
                    self.session.packets_received += 1
                    self.session.bytes_received += len(data)
                    self.session.update_activity()

            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    print(f"[RTPReceiver] Receive error: {e}")

    def _decoder_loop(self):
        """Decode and write audio."""
        while self._running:
            try:
                # Pop packet from jitter buffer
                packet = self.jitter_buffer.pop()
                if packet is None:
                    time.sleep(0.01)
                    continue

                # Decode packet
                if self.decoder and self.audio_writer:
                    pcm_samples = self.decoder.decode(packet.payload)

                    # Write to file
                    self.audio_writer.write(pcm_samples)

            except Exception as e:
                if self._running:
                    print(f"[RTPReceiver] Decode error: {e}")

    @staticmethod
    def _parse_rtp_packet(data: bytes) -> Optional[RTPPacket]:
        """Parse RTP packet.

        Args:
            data: Raw UDP packet data

        Returns:
            RTPPacket or None if invalid
        """
        if len(data) < 12:
            return None

        try:
            # Parse RTP header (RFC 3550)
            # 0-1: V(2), P(1), X(1), CC(4), M(1), PT(7)
            # 2-3: Sequence number
            # 4-7: Timestamp
            # 8-11: SSRC
            byte0, byte1, seq, timestamp, ssrc = struct.unpack('!BBHII', data[:12])

            version = (byte0 >> 6) & 0x03
            padding = (byte0 >> 5) & 0x01
            extension = (byte0 >> 4) & 0x01
            csrc_count = byte0 & 0x0F

            marker = (byte1 >> 7) & 0x01
            payload_type = byte1 & 0x7F

            # Skip CSRC identifiers
            header_len = 12 + (csrc_count * 4)

            # Skip extension if present
            if extension:
                if len(data) < header_len + 4:
                    return None
                ext_len = struct.unpack('!H', data[header_len+2:header_len+4])[0]
                header_len += 4 + (ext_len * 4)

            # Extract payload
            payload = data[header_len:]

            # Remove padding if present
            if padding and len(payload) > 0:
                padding_len = payload[-1]
                payload = payload[:-padding_len]

            return RTPPacket(
                sequence_number=seq,
                timestamp=timestamp,
                payload_type=payload_type,
                payload=payload,
                received_at=time.time()
            )

        except Exception as e:
            print(f"[RTPReceiver] Parse error: {e}")
            return None
