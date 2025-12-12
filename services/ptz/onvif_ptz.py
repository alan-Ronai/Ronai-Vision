"""ONVIF PTZ controller for real RTSP cameras.

This module provides PTZ control for ONVIF-compatible cameras using
continuous and absolute positioning commands.
"""

from typing import Optional, Dict, Any
import logging

try:
    from onvif import ONVIFCamera
    from onvif.exceptions import ONVIFError
    ONVIF_AVAILABLE = True
except ImportError:
    ONVIF_AVAILABLE = False
    ONVIFCamera = None
    ONVIFError = Exception

logger = logging.getLogger(__name__)


class ONVIFPTZController:
    """ONVIF PTZ controller for real cameras.

    Supports:
    - Continuous movement (move with velocity)
    - Absolute positioning
    - Relative movement
    - Preset positions
    - PTZ status queries
    """

    def __init__(
        self,
        host: str,
        port: int = 80,
        username: str = "admin",
        password: str = "",
        wsdl_dir: Optional[str] = None,
        no_ssl_verify: bool = True,
    ):
        """Initialize ONVIF PTZ controller.

        Args:
            host: Camera IP address or hostname
            port: ONVIF port (default: 80)
            username: ONVIF username
            password: ONVIF password
            wsdl_dir: Path to WSDL files (optional, uses python-onvif default if None)
            no_ssl_verify: Disable SSL certificate verification (default: True for self-signed certs)

        Raises:
            ImportError: If python-onvif-zeep is not installed
            ONVIFError: If camera connection fails
        """
        if not ONVIF_AVAILABLE:
            raise ImportError(
                "python-onvif-zeep is required for ONVIF PTZ control. "
                "Install with: pip install onvif-zeep"
            )

        self.host = host
        self.port = port
        self.username = username
        self.password = password

        try:
            # Disable SSL verification for self-signed certificates
            if no_ssl_verify:
                import ssl
                ssl._create_default_https_context = ssl._create_unverified_context

            # Connect to camera
            if wsdl_dir:
                self.camera = ONVIFCamera(host, port, username, password, wsdl_dir)
            else:
                self.camera = ONVIFCamera(host, port, username, password)

            # Get PTZ service
            self.ptz_service = self.camera.create_ptz_service()

            # Get media service (for profile tokens)
            self.media_service = self.camera.create_media_service()

            # Get first profile token (most cameras use profile 0)
            profiles = self.media_service.GetProfiles()
            if not profiles:
                raise ONVIFError("No media profiles found on camera")

            self.profile_token = profiles[0].token
            logger.info(f"Connected to ONVIF camera at {host}:{port}, profile: {self.profile_token}")

            # Get PTZ configuration
            self.ptz_config = self.ptz_service.GetConfigurationOptions({
                'ConfigurationToken': profiles[0].PTZConfiguration.token
            })

            # Cache movement speeds (normalized to -1.0 to 1.0 range)
            self.default_pan_speed = 0.5
            self.default_tilt_speed = 0.5
            self.default_zoom_speed = 0.5

        except Exception as e:
            logger.error(f"Failed to connect to ONVIF camera: {e}")
            raise

    def continuous_move(
        self,
        pan_velocity: float = 0.0,
        tilt_velocity: float = 0.0,
        zoom_velocity: float = 0.0,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Move PTZ continuously with specified velocities.

        Args:
            pan_velocity: Pan velocity (-1.0 to 1.0, negative=left, positive=right)
            tilt_velocity: Tilt velocity (-1.0 to 1.0, negative=down, positive=up)
            zoom_velocity: Zoom velocity (-1.0 to 1.0, negative=zoom out, positive=zoom in)
            timeout: Movement duration in seconds (None=continuous until stop())

        Returns:
            Status dictionary
        """
        try:
            # Create velocity request
            request = self.ptz_service.create_type('ContinuousMove')
            request.ProfileToken = self.profile_token

            # Set velocities
            if request.Velocity is None:
                request.Velocity = self.ptz_service.GetStatus({'ProfileToken': self.profile_token}).Position

            request.Velocity.PanTilt.x = max(-1.0, min(1.0, pan_velocity))
            request.Velocity.PanTilt.y = max(-1.0, min(1.0, tilt_velocity))
            request.Velocity.Zoom.x = max(-1.0, min(1.0, zoom_velocity))

            if timeout is not None:
                request.Timeout = f"PT{timeout}S"

            self.ptz_service.ContinuousMove(request)
            logger.info(f"PTZ continuous move: pan={pan_velocity:.2f}, tilt={tilt_velocity:.2f}, zoom={zoom_velocity:.2f}")

            return {
                "ok": True,
                "action": "continuous_move",
                "pan_velocity": pan_velocity,
                "tilt_velocity": tilt_velocity,
                "zoom_velocity": zoom_velocity,
            }

        except Exception as e:
            logger.error(f"PTZ continuous move failed: {e}")
            return {"ok": False, "error": str(e)}

    def stop(self) -> Dict[str, Any]:
        """Stop all PTZ movement.

        Returns:
            Status dictionary
        """
        try:
            request = self.ptz_service.create_type('Stop')
            request.ProfileToken = self.profile_token
            request.PanTilt = True
            request.Zoom = True

            self.ptz_service.Stop(request)
            logger.info("PTZ stopped")

            return {"ok": True, "action": "stop"}

        except Exception as e:
            logger.error(f"PTZ stop failed: {e}")
            return {"ok": False, "error": str(e)}

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
            pan: Pan position (-1.0 to 1.0, None=don't change)
            tilt: Tilt position (-1.0 to 1.0, None=don't change)
            zoom: Zoom position (0.0 to 1.0, None=don't change)
            pan_speed: Pan movement speed (0.0 to 1.0, None=use default)
            tilt_speed: Tilt movement speed (0.0 to 1.0, None=use default)
            zoom_speed: Zoom movement speed (0.0 to 1.0, None=use default)

        Returns:
            Status dictionary
        """
        try:
            request = self.ptz_service.create_type('AbsoluteMove')
            request.ProfileToken = self.profile_token

            # Get current position
            status = self.ptz_service.GetStatus({'ProfileToken': self.profile_token})
            current_pos = status.Position

            # Set position (use current if not specified)
            request.Position.PanTilt.x = pan if pan is not None else current_pos.PanTilt.x
            request.Position.PanTilt.y = tilt if tilt is not None else current_pos.PanTilt.y
            request.Position.Zoom.x = zoom if zoom is not None else current_pos.Zoom.x

            # Set speeds
            if any([pan_speed, tilt_speed, zoom_speed]):
                request.Speed = self.ptz_service.create_type('PTZSpeed')
                request.Speed.PanTilt.x = pan_speed or self.default_pan_speed
                request.Speed.PanTilt.y = tilt_speed or self.default_tilt_speed
                request.Speed.Zoom.x = zoom_speed or self.default_zoom_speed

            self.ptz_service.AbsoluteMove(request)
            logger.info(f"PTZ absolute move: pan={pan}, tilt={tilt}, zoom={zoom}")

            return {
                "ok": True,
                "action": "absolute_move",
                "pan": request.Position.PanTilt.x,
                "tilt": request.Position.PanTilt.y,
                "zoom": request.Position.Zoom.x,
            }

        except Exception as e:
            logger.error(f"PTZ absolute move failed: {e}")
            return {"ok": False, "error": str(e)}

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
            pan_delta: Pan change (-1.0 to 1.0)
            tilt_delta: Tilt change (-1.0 to 1.0)
            zoom_delta: Zoom change (-1.0 to 1.0)
            pan_speed: Pan movement speed (0.0 to 1.0, None=use default)
            tilt_speed: Tilt movement speed (0.0 to 1.0, None=use default)
            zoom_speed: Zoom movement speed (0.0 to 1.0, None=use default)

        Returns:
            Status dictionary
        """
        try:
            request = self.ptz_service.create_type('RelativeMove')
            request.ProfileToken = self.profile_token

            # Set translation
            request.Translation.PanTilt.x = pan_delta
            request.Translation.PanTilt.y = tilt_delta
            request.Translation.Zoom.x = zoom_delta

            # Set speeds
            if any([pan_speed, tilt_speed, zoom_speed]):
                request.Speed = self.ptz_service.create_type('PTZSpeed')
                request.Speed.PanTilt.x = pan_speed or self.default_pan_speed
                request.Speed.PanTilt.y = tilt_speed or self.default_tilt_speed
                request.Speed.Zoom.x = zoom_speed or self.default_zoom_speed

            self.ptz_service.RelativeMove(request)
            logger.info(f"PTZ relative move: pan_delta={pan_delta:.2f}, tilt_delta={tilt_delta:.2f}, zoom_delta={zoom_delta:.2f}")

            return {
                "ok": True,
                "action": "relative_move",
                "pan_delta": pan_delta,
                "tilt_delta": tilt_delta,
                "zoom_delta": zoom_delta,
            }

        except Exception as e:
            logger.error(f"PTZ relative move failed: {e}")
            return {"ok": False, "error": str(e)}

    def get_status(self) -> Dict[str, Any]:
        """Get current PTZ status.

        Returns:
            Status dictionary with position and movement state
        """
        try:
            status = self.ptz_service.GetStatus({'ProfileToken': self.profile_token})

            return {
                "ok": True,
                "position": {
                    "pan": float(status.Position.PanTilt.x),
                    "tilt": float(status.Position.PanTilt.y),
                    "zoom": float(status.Position.Zoom.x),
                },
                "move_status": str(status.MoveStatus.PanTilt) if status.MoveStatus else "IDLE",
                "error": status.Error if hasattr(status, 'Error') and status.Error else None,
            }

        except Exception as e:
            logger.error(f"PTZ get status failed: {e}")
            return {"ok": False, "error": str(e)}

    def goto_home_position(self, speed: Optional[float] = None) -> Dict[str, Any]:
        """Move PTZ to home position (preset 0).

        Args:
            speed: Movement speed (0.0 to 1.0, None=use default)

        Returns:
            Status dictionary
        """
        try:
            request = self.ptz_service.create_type('GotoHomePosition')
            request.ProfileToken = self.profile_token

            if speed is not None:
                request.Speed = self.ptz_service.create_type('PTZSpeed')
                request.Speed.PanTilt.x = speed
                request.Speed.PanTilt.y = speed
                request.Speed.Zoom.x = speed

            self.ptz_service.GotoHomePosition(request)
            logger.info("PTZ moving to home position")

            return {"ok": True, "action": "goto_home"}

        except Exception as e:
            logger.error(f"PTZ goto home failed: {e}")
            return {"ok": False, "error": str(e)}

    def set_preset(self, preset_name: str, preset_token: Optional[str] = None) -> Dict[str, Any]:
        """Save current PTZ position as a preset.

        Args:
            preset_name: Name for the preset
            preset_token: Preset token (None=auto-generate)

        Returns:
            Status dictionary with preset token
        """
        try:
            request = self.ptz_service.create_type('SetPreset')
            request.ProfileToken = self.profile_token
            request.PresetName = preset_name
            if preset_token:
                request.PresetToken = preset_token

            response = self.ptz_service.SetPreset(request)
            logger.info(f"PTZ preset saved: {preset_name} (token: {response})")

            return {"ok": True, "action": "set_preset", "preset_name": preset_name, "preset_token": str(response)}

        except Exception as e:
            logger.error(f"PTZ set preset failed: {e}")
            return {"ok": False, "error": str(e)}

    def goto_preset(self, preset_token: str, speed: Optional[float] = None) -> Dict[str, Any]:
        """Move PTZ to a saved preset position.

        Args:
            preset_token: Preset token to move to
            speed: Movement speed (0.0 to 1.0, None=use default)

        Returns:
            Status dictionary
        """
        try:
            request = self.ptz_service.create_type('GotoPreset')
            request.ProfileToken = self.profile_token
            request.PresetToken = preset_token

            if speed is not None:
                request.Speed = self.ptz_service.create_type('PTZSpeed')
                request.Speed.PanTilt.x = speed
                request.Speed.PanTilt.y = speed
                request.Speed.Zoom.x = speed

            self.ptz_service.GotoPreset(request)
            logger.info(f"PTZ moving to preset: {preset_token}")

            return {"ok": True, "action": "goto_preset", "preset_token": preset_token}

        except Exception as e:
            logger.error(f"PTZ goto preset failed: {e}")
            return {"ok": False, "error": str(e)}

    def get_presets(self) -> Dict[str, Any]:
        """Get list of saved presets.

        Returns:
            Status dictionary with preset list
        """
        try:
            presets = self.ptz_service.GetPresets({'ProfileToken': self.profile_token})

            preset_list = [
                {
                    "token": preset.token,
                    "name": preset.Name if hasattr(preset, 'Name') else None,
                    "pan": float(preset.PTZPosition.PanTilt.x) if preset.PTZPosition else None,
                    "tilt": float(preset.PTZPosition.PanTilt.y) if preset.PTZPosition else None,
                    "zoom": float(preset.PTZPosition.Zoom.x) if preset.PTZPosition else None,
                }
                for preset in presets
            ]

            return {"ok": True, "presets": preset_list}

        except Exception as e:
            logger.error(f"PTZ get presets failed: {e}")
            return {"ok": False, "error": str(e)}
