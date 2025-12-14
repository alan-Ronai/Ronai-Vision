"""HTTP-based PTZ controller for cameras without ONVIF support.

This module provides PTZ control for cameras that use HTTP CGI commands
instead of ONVIF protocol (common with Hikvision, Dahua, and similar brands).
"""

from typing import Optional, Dict, Any
import logging
import requests
from requests.auth import HTTPDigestAuth, HTTPBasicAuth
from urllib.parse import urljoin

logger = logging.getLogger(__name__)


class HTTPPTZController:
    """HTTP CGI-based PTZ controller.

    Supports multiple camera brands:
    - Hikvision
    - Dahua
    - Generic ONVIF-less IP cameras
    """

    def __init__(
        self,
        host: str,
        port: int = 80,
        username: str = "admin",
        password: str = "",
        brand: str = "hikvision",
        use_https: bool = False,
    ):
        """Initialize HTTP PTZ controller.

        Args:
            host: Camera IP address or hostname
            port: HTTP port (default: 80)
            username: Camera username
            password: Camera password
            brand: Camera brand ('hikvision', 'dahua', 'generic')
            use_https: Use HTTPS instead of HTTP
        """
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.brand = brand.lower()

        protocol = "https" if use_https else "http"
        self.base_url = f"{protocol}://{host}:{port}"

        # Try digest auth first, fall back to basic
        self.auth_digest = HTTPDigestAuth(username, password)
        self.auth_basic = HTTPBasicAuth(username, password)

        logger.info(
            f"HTTP PTZ controller initialized for {brand} camera at {self.base_url}"
        )

    def _request(
        self, path: str, params: Optional[Dict] = None, method: str = "GET"
    ) -> Dict[str, Any]:
        """Make authenticated HTTP request to camera.

        Args:
            path: URL path
            params: Query parameters
            method: HTTP method

        Returns:
            Response dictionary
        """
        url = urljoin(self.base_url, path)

        try:
            # Try digest auth first
            if method == "GET":
                response = requests.get(
                    url, params=params, auth=self.auth_digest, timeout=5, verify=False
                )
            else:
                response = requests.post(
                    url, data=params, auth=self.auth_digest, timeout=5, verify=False
                )

            # If digest fails with 401, try basic auth
            if response.status_code == 401:
                if method == "GET":
                    response = requests.get(
                        url,
                        params=params,
                        auth=self.auth_basic,
                        timeout=5,
                        verify=False,
                    )
                else:
                    response = requests.post(
                        url, data=params, auth=self.auth_basic, timeout=5, verify=False
                    )

            if response.status_code == 200:
                return {
                    "ok": True,
                    "status_code": response.status_code,
                    "text": response.text,
                }
            else:
                return {
                    "ok": False,
                    "status_code": response.status_code,
                    "error": response.text,
                }

        except Exception as e:
            logger.error(f"HTTP request failed: {e}")
            return {"ok": False, "error": str(e)}

    def _hikvision_ptz(self, command: str, speed: int = 50) -> Dict[str, Any]:
        """Execute Hikvision PTZ command.

        Args:
            command: PTZ command (e.g., 'UP', 'DOWN', 'LEFT', 'RIGHT', 'ZOOM_IN', 'ZOOM_OUT')
            speed: Movement speed (1-100)

        Returns:
            Response dictionary
        """
        # Map commands to pan/tilt/zoom values
        if command == "RIGHT":
            pan, tilt, zoom = speed, 0, 0
        elif command == "LEFT":
            pan, tilt, zoom = -speed, 0, 0
        elif command == "UP":
            pan, tilt, zoom = 0, speed, 0
        elif command == "DOWN":
            pan, tilt, zoom = 0, -speed, 0
        elif command == "ZOOM_IN":
            pan, tilt, zoom = 0, 0, speed
        elif command == "ZOOM_OUT":
            pan, tilt, zoom = 0, 0, -speed
        else:  # STOP
            pan, tilt, zoom = 0, 0, 0

        # Try multiple ISAPI endpoints (different channels, momentary vs continuous)
        endpoints = [
            "/ISAPI/PTZCtrl/channels/1/continuous",
            "/ISAPI/PTZCtrl/channels/1/momentary",
            "/ISAPI/System/Video/inputs/channels/1/PTZ/Continuous",
        ]

        last_error = None

        for path in endpoints:
            # Hikvision uses XML payload
            xml_data = f"""<?xml version="1.0" encoding="UTF-8"?>
<PTZData>
    <pan>{pan}</pan>
    <tilt>{tilt}</tilt>
    <zoom>{zoom}</zoom>
</PTZData>"""

            try:
                url = urljoin(self.base_url, path)

                # Try digest auth
                response = requests.put(
                    url, data=xml_data, auth=self.auth_digest, timeout=5, verify=False
                )

                # Try basic auth if digest fails
                if response.status_code == 401:
                    response = requests.put(
                        url,
                        data=xml_data,
                        auth=self.auth_basic,
                        timeout=5,
                        verify=False,
                    )

                if response.status_code in [200, 204]:
                    logger.info(
                        f"Hikvision PTZ command sent via {path}: {command} (speed={speed})"
                    )
                    return {
                        "ok": True,
                        "action": command,
                        "status_code": response.status_code,
                        "endpoint": path,
                    }
                else:
                    last_error = f"Status {response.status_code}: {response.text[:100]}"
                    logger.debug(f"Failed endpoint {path}: {last_error}")

            except Exception as e:
                last_error = str(e)
                logger.debug(f"Exception on endpoint {path}: {e}")
                continue

        # All endpoints failed
        return {
            "ok": False,
            "error": last_error or "All ISAPI endpoints failed",
            "tried_endpoints": endpoints,
        }

    def _dahua_ptz(self, command: str, speed: int = 50) -> Dict[str, Any]:
        """Execute Dahua PTZ command.

        Args:
            command: PTZ command
            speed: Movement speed (1-100)

        Returns:
            Response dictionary
        """
        # Dahua PTZ commands
        cmd_map = {
            "UP": 0,
            "DOWN": 1,
            "LEFT": 2,
            "RIGHT": 3,
            "ZOOM_IN": 10,
            "ZOOM_OUT": 11,
            "STOP": 8,
        }

        code = cmd_map.get(command, 8)
        path = "/cgi-bin/ptz.cgi"
        params = {"action": "start", "code": code, "arg1": 0, "arg2": speed, "arg3": 0}

        return self._request(path, params)

    def _generic_ptz(self, command: str, speed: int = 50) -> Dict[str, Any]:
        """Execute generic PTZ command.

        Args:
            command: PTZ command
            speed: Movement speed (1-100)

        Returns:
            Response dictionary
        """
        # Try common CGI patterns
        paths = [
            f"/cgi-bin/ptz.cgi?move={command.lower()}&speed={speed}",
            f"/axis-cgi/com/ptz.cgi?move={command.lower()}&speed={speed}",
            f"/ptz?action={command.lower()}&speed={speed}",
        ]

        for path in paths:
            result = self._request(path)
            if result.get("ok"):
                return result

        return {"ok": False, "error": "No working PTZ endpoint found"}

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
            timeout: Movement duration (ignored for HTTP PTZ)

        Returns:
            Response dictionary
        """
        # Determine primary direction
        if abs(pan_velocity) > abs(tilt_velocity) and abs(pan_velocity) > abs(
            zoom_velocity
        ):
            command = "RIGHT" if pan_velocity > 0 else "LEFT"
            speed = int(abs(pan_velocity) * 100)
        elif abs(tilt_velocity) > abs(zoom_velocity):
            command = "UP" if tilt_velocity > 0 else "DOWN"
            speed = int(abs(tilt_velocity) * 100)
        elif abs(zoom_velocity) > 0:
            command = "ZOOM_IN" if zoom_velocity > 0 else "ZOOM_OUT"
            speed = int(abs(zoom_velocity) * 100)
        else:
            command = "STOP"
            speed = 0

        # Execute command based on brand
        if self.brand == "hikvision":
            result = self._hikvision_ptz(command, speed)
        elif self.brand == "dahua":
            result = self._dahua_ptz(command, speed)
        else:
            result = self._generic_ptz(command, speed)

        if result.get("ok"):
            result.update(
                {
                    "pan_velocity": pan_velocity,
                    "tilt_velocity": tilt_velocity,
                    "zoom_velocity": zoom_velocity,
                }
            )

        return result

    def stop(self) -> Dict[str, Any]:
        """Stop all PTZ movement.

        Returns:
            Response dictionary
        """
        if self.brand == "hikvision":
            return self._hikvision_ptz("STOP", 0)
        elif self.brand == "dahua":
            return self._dahua_ptz("STOP", 0)
        else:
            return self._generic_ptz("STOP", 0)

    def get_status(self) -> Dict[str, Any]:
        """Get current PTZ status.

        Note: HTTP-based cameras often don't provide position feedback.

        Returns:
            Response dictionary
        """
        return {
            "ok": True,
            "note": "HTTP PTZ cameras typically don't provide position feedback",
            "position": {
                "pan": None,
                "tilt": None,
                "zoom": None,
            },
        }

    def goto_preset(self, preset_number: int) -> Dict[str, Any]:
        """Go to preset position.

        Args:
            preset_number: Preset number (1-255)

        Returns:
            Response dictionary
        """
        if self.brand == "hikvision":
            path = f"/ISAPI/PTZCtrl/channels/1/presets/{preset_number}/goto"
            return self._request(path, method="PUT")
        elif self.brand == "dahua":
            path = "/cgi-bin/ptz.cgi"
            params = {
                "action": "start",
                "code": "GotoPreset",
                "arg1": 0,
                "arg2": preset_number,
                "arg3": 0,
            }
            return self._request(path, params)
        else:
            return {"ok": False, "error": "Preset not supported for generic cameras"}

    def set_preset(self, preset_number: int, preset_name: str = "") -> Dict[str, Any]:
        """Save current position as preset.

        Args:
            preset_number: Preset number (1-255)
            preset_name: Preset name (optional)

        Returns:
            Response dictionary
        """
        if self.brand == "hikvision":
            path = f"/ISAPI/PTZCtrl/channels/1/presets/{preset_number}"
            xml_data = f"""<?xml version="1.0" encoding="UTF-8"?>
<PTZPreset>
    <id>{preset_number}</id>
    <presetName>{preset_name or f"Preset_{preset_number}"}</presetName>
</PTZPreset>"""

            try:
                url = urljoin(self.base_url, path)
                response = requests.put(
                    url, data=xml_data, auth=self.auth_digest, timeout=5, verify=False
                )
                if response.status_code in [200, 201, 204]:
                    return {
                        "ok": True,
                        "preset_number": preset_number,
                        "preset_name": preset_name,
                    }
                else:
                    return {"ok": False, "error": f"Status {response.status_code}"}
            except Exception as e:
                return {"ok": False, "error": str(e)}

        elif self.brand == "dahua":
            path = "/cgi-bin/ptz.cgi"
            params = {
                "action": "start",
                "code": "SetPreset",
                "arg1": 0,
                "arg2": preset_number,
                "arg3": 0,
            }
            return self._request(path, params)
        else:
            return {"ok": False, "error": "Preset not supported for generic cameras"}

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

        Note: Most HTTP PTZ cameras don't support absolute positioning.
        This method is provided for API compatibility but may not work.

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
        return {
            "ok": False,
            "error": "Absolute positioning not supported for HTTP PTZ cameras. Use continuous_move() or presets instead.",
        }

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

        Note: Most HTTP PTZ cameras don't support relative positioning.
        This method is provided for API compatibility but may not work.

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
        return {
            "ok": False,
            "error": "Relative positioning not supported for HTTP PTZ cameras. Use continuous_move() or presets instead.",
        }

    def goto_home_position(self, speed: Optional[float] = None) -> Dict[str, Any]:
        """Move to home position.

        Args:
            speed: Movement speed (ignored)

        Returns:
            Response dictionary
        """
        # Home position is typically preset 0 or 1
        return self.goto_preset(0)

    def get_presets(self) -> Dict[str, Any]:
        """Get list of presets.

        Note: Most HTTP PTZ cameras don't provide preset listing via API.

        Returns:
            Response dictionary
        """
        return {
            "ok": True,
            "presets": [],
            "note": "HTTP PTZ cameras typically don't provide preset listing. Try preset numbers 0-99.",
        }
