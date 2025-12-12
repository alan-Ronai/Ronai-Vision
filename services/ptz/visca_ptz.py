"""VISCA over IP PTZ controller.

VISCA is a professional camera control protocol used by Sony, Canon, and other
professional camera manufacturers. It typically operates on TCP port 5678 or UDP port 1259.
"""

from typing import Optional, Dict, Any
import socket
import struct
import logging
import time

logger = logging.getLogger(__name__)


class VISCAPTZController:
    """VISCA over IP PTZ controller.

    Supports both TCP (port 5678) and UDP (port 1259) transports.
    """

    def __init__(
        self,
        host: str,
        port: int = 5678,
        transport: str = "tcp",  # 'tcp' or 'udp'
        camera_id: int = 1,
    ):
        """Initialize VISCA PTZ controller.

        Args:
            host: Camera IP address
            port: VISCA port (default: 5678 for TCP, 1259 for UDP)
            transport: Transport protocol ('tcp' or 'udp')
            camera_id: Camera ID (1-7, default: 1)
        """
        self.host = host
        self.port = port
        self.transport = transport.lower()
        self.camera_id = camera_id
        self.sequence_number = 0
        self.socket = None

        logger.info(f"VISCA PTZ controller initialized for {host}:{port} ({transport.upper()})")

    def _connect(self):
        """Establish connection to camera."""
        if self.socket:
            return True

        try:
            if self.transport == "tcp":
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.settimeout(5.0)
                self.socket.connect((self.host, self.port))
                logger.info(f"Connected to VISCA camera via TCP")
            else:  # UDP
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.socket.settimeout(5.0)
                logger.info(f"Connected to VISCA camera via UDP")

            return True

        except Exception as e:
            logger.error(f"Failed to connect to VISCA camera: {e}")
            self.socket = None
            return False

    def _send_command(self, command: bytes) -> Dict[str, Any]:
        """Send VISCA command to camera.

        Args:
            command: VISCA command bytes

        Returns:
            Response dictionary
        """
        if not self._connect():
            return {"ok": False, "error": "Failed to connect"}

        try:
            # VISCA over IP header (for TCP)
            if self.transport == "tcp":
                # Header format: [type][length][sequence]
                # Type: 0x0100 for command, 0x0200 for inquiry
                header = struct.pack(">HHI", 0x0100, len(command), self.sequence_number)
                packet = header + command
                self.sequence_number += 1
            else:
                # UDP uses raw VISCA packets
                packet = command

            # Send command
            if self.transport == "tcp":
                self.socket.send(packet)
            else:
                self.socket.sendto(packet, (self.host, self.port))

            # Try to receive response (may timeout for UDP)
            try:
                if self.transport == "tcp":
                    response = self.socket.recv(1024)
                else:
                    response, _ = self.socket.recvfrom(1024)

                logger.debug(f"VISCA response: {response.hex()}")
                return {"ok": True, "response": response}

            except socket.timeout:
                # Some commands don't send responses
                return {"ok": True, "response": None}

        except Exception as e:
            logger.error(f"VISCA command failed: {e}")
            return {"ok": False, "error": str(e)}

    def _build_visca_command(self, command_bytes: list) -> bytes:
        """Build VISCA command packet.

        Args:
            command_bytes: List of command bytes (without address and terminator)

        Returns:
            Complete VISCA packet
        """
        # VISCA format: [address][command][terminator]
        # Address: 0x80 + camera_id (0-7)
        address = 0x80 + (self.camera_id - 1)
        packet = bytes([address] + command_bytes + [0xFF])
        return packet

    def continuous_move(
        self,
        pan_velocity: float = 0.0,
        tilt_velocity: float = 0.0,
        zoom_velocity: float = 0.0,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Move PTZ continuously.

        Args:
            pan_velocity: Pan velocity (-1.0 to 1.0)
            tilt_velocity: Tilt velocity (-1.0 to 1.0)
            zoom_velocity: Zoom velocity (-1.0 to 1.0)
            timeout: Movement duration (ignored for VISCA)

        Returns:
            Response dictionary
        """
        # Convert velocities to VISCA speed (0x01-0x18, max 24)
        pan_speed = int(abs(pan_velocity) * 24)
        tilt_speed = int(abs(tilt_velocity) * 24)

        pan_speed = max(1, min(24, pan_speed)) if pan_speed > 0 else 0
        tilt_speed = max(1, min(24, tilt_speed)) if tilt_speed > 0 else 0

        # Determine direction
        # Pan: 0x01=left, 0x02=right, 0x03=stop
        # Tilt: 0x01=up, 0x02=down, 0x03=stop
        if pan_velocity < 0:
            pan_dir = 0x01  # Left
        elif pan_velocity > 0:
            pan_dir = 0x02  # Right
        else:
            pan_dir = 0x03  # Stop

        if tilt_velocity > 0:
            tilt_dir = 0x01  # Up
        elif tilt_velocity < 0:
            tilt_dir = 0x02  # Down
        else:
            tilt_dir = 0x03  # Stop

        # Build pan/tilt command
        # Command: 0x01 0x06 0x01 [pan_speed] [tilt_speed] [pan_dir] [tilt_dir]
        command = [0x01, 0x06, 0x01, pan_speed, tilt_speed, pan_dir, tilt_dir]
        packet = self._build_visca_command(command)

        result = self._send_command(packet)

        # Handle zoom separately if needed
        if zoom_velocity != 0:
            zoom_speed = int(abs(zoom_velocity) * 7)  # VISCA zoom: 0-7
            zoom_speed = max(0, min(7, zoom_speed))

            if zoom_velocity > 0:
                # Zoom in: 0x01 0x04 0x07 0x2p (p=speed)
                zoom_cmd = [0x01, 0x04, 0x07, 0x20 + zoom_speed]
            else:
                # Zoom out: 0x01 0x04 0x07 0x3p (p=speed)
                zoom_cmd = [0x01, 0x04, 0x07, 0x30 + zoom_speed]

            zoom_packet = self._build_visca_command(zoom_cmd)
            self._send_command(zoom_packet)

        if result.get("ok"):
            result.update({
                "pan_velocity": pan_velocity,
                "tilt_velocity": tilt_velocity,
                "zoom_velocity": zoom_velocity,
                "pan_speed": pan_speed,
                "tilt_speed": tilt_speed,
            })

        return result

    def stop(self) -> Dict[str, Any]:
        """Stop all PTZ movement.

        Returns:
            Response dictionary
        """
        # Stop pan/tilt: 0x01 0x06 0x01 [pan_speed] [tilt_speed] 0x03 0x03
        command = [0x01, 0x06, 0x01, 0x00, 0x00, 0x03, 0x03]
        packet = self._build_visca_command(command)

        result = self._send_command(packet)

        # Stop zoom: 0x01 0x04 0x07 0x00
        zoom_cmd = [0x01, 0x04, 0x07, 0x00]
        zoom_packet = self._build_visca_command(zoom_cmd)
        self._send_command(zoom_packet)

        return result

    def get_status(self) -> Dict[str, Any]:
        """Get current PTZ status.

        Returns:
            Response dictionary
        """
        # VISCA inquiry for pan/tilt position: 0x09 0x06 0x12
        command = [0x09, 0x06, 0x12]
        packet = self._build_visca_command(command)

        result = self._send_command(packet)

        if result.get("ok") and result.get("response"):
            # Parse response (format varies by camera)
            return {
                "ok": True,
                "position": {
                    "pan": None,  # Would need to parse from response
                    "tilt": None,
                    "zoom": None,
                },
                "raw_response": result["response"].hex(),
            }

        return {"ok": True, "note": "Position feedback not available"}

    def goto_home_position(self, speed: Optional[float] = None) -> Dict[str, Any]:
        """Move to home position.

        Args:
            speed: Movement speed (ignored)

        Returns:
            Response dictionary
        """
        # Home command: 0x01 0x06 0x04
        command = [0x01, 0x06, 0x04]
        packet = self._build_visca_command(command)

        return self._send_command(packet)

    def goto_preset(self, preset_number: int) -> Dict[str, Any]:
        """Go to preset position.

        Args:
            preset_number: Preset number (0-255)

        Returns:
            Response dictionary
        """
        # Recall preset: 0x01 0x04 0x3F 0x02 [preset]
        command = [0x01, 0x04, 0x3F, 0x02, preset_number]
        packet = self._build_visca_command(command)

        return self._send_command(packet)

    def set_preset(self, preset_number: int, preset_name: str = "") -> Dict[str, Any]:
        """Save current position as preset.

        Args:
            preset_number: Preset number (0-255)
            preset_name: Preset name (ignored by VISCA)

        Returns:
            Response dictionary
        """
        # Set preset: 0x01 0x04 0x3F 0x01 [preset]
        command = [0x01, 0x04, 0x3F, 0x01, preset_number]
        packet = self._build_visca_command(command)

        return self._send_command(packet)

    def absolute_move(
        self,
        pan: Optional[float] = None,
        tilt: Optional[float] = None,
        zoom: Optional[float] = None,
        pan_speed: Optional[float] = None,
        tilt_speed: Optional[float] = None,
        zoom_speed: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Move PTZ to absolute position.

        Args:
            pan: Pan position (-1.0 to 1.0)
            tilt: Tilt position (-1.0 to 1.0)
            zoom: Zoom position (0.0 to 1.0)
            pan_speed: Movement speed
            tilt_speed: Movement speed
            zoom_speed: Movement speed

        Returns:
            Response dictionary
        """
        if pan is None or tilt is None:
            return {"ok": False, "error": "Both pan and tilt required for absolute positioning"}

        # Convert to VISCA position values
        # Pan: -170° to +170° → 0x0000 to 0xFFFF (typically ±1440 for ±100%)
        # Tilt: -30° to +90° → 0x0000 to 0xFFFF (typically ±360 for ±100%)

        pan_pos = int(pan * 1440)  # Scale to ±1440
        tilt_pos = int(tilt * 360)  # Scale to ±360

        # Convert to VISCA format (split into nibbles)
        pan_pos = pan_pos + 0x8000  # Offset to unsigned
        tilt_pos = tilt_pos + 0x8000

        p1 = (pan_pos >> 12) & 0x0F
        p2 = (pan_pos >> 8) & 0x0F
        p3 = (pan_pos >> 4) & 0x0F
        p4 = pan_pos & 0x0F

        t1 = (tilt_pos >> 12) & 0x0F
        t2 = (tilt_pos >> 8) & 0x0F
        t3 = (tilt_pos >> 4) & 0x0F
        t4 = tilt_pos & 0x0F

        speed_pan = int((pan_speed or 0.5) * 24)
        speed_tilt = int((tilt_speed or 0.5) * 24)

        # Absolute position command: 0x01 0x06 0x02 [pan_speed] [tilt_speed] [pan_pos] [tilt_pos]
        command = [0x01, 0x06, 0x02, speed_pan, speed_tilt, p1, p2, p3, p4, t1, t2, t3, t4]
        packet = self._build_visca_command(command)

        return self._send_command(packet)

    def relative_move(
        self,
        pan_delta: float = 0.0,
        tilt_delta: float = 0.0,
        zoom_delta: float = 0.0,
        pan_speed: Optional[float] = None,
        tilt_speed: Optional[float] = None,
        zoom_speed: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Move PTZ relative to current position.

        Args:
            pan_delta: Pan change
            tilt_delta: Tilt change
            zoom_delta: Zoom change
            pan_speed: Movement speed
            tilt_speed: Movement speed
            zoom_speed: Movement speed

        Returns:
            Response dictionary
        """
        # VISCA doesn't have native relative positioning
        # Would need to query current position first, then do absolute move
        return {
            "ok": False,
            "error": "Relative positioning requires querying current position first"
        }

    def get_presets(self) -> Dict[str, Any]:
        """Get list of presets.

        Returns:
            Response dictionary
        """
        return {
            "ok": True,
            "presets": [],
            "note": "VISCA cameras support presets 0-255. Try calling them directly."
        }

    def close(self):
        """Close connection."""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
