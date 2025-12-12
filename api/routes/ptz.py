"""PTZ control API endpoints.

Provides REST API for controlling PTZ cameras via ONVIF protocol.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Global PTZ controller instance (initialized per camera)
_ptz_controllers: Dict[str, Any] = {}


class PTZConfig(BaseModel):
    """PTZ camera configuration."""
    camera_id: str
    host: str
    port: int = 80
    username: str = "admin"
    password: str = ""
    protocol: str = "http"  # 'onvif' or 'http'
    brand: str = "hikvision"  # For HTTP: 'hikvision', 'dahua', 'generic'
    no_ssl_verify: bool = True
    use_https: bool = False


class ContinuousMoveRequest(BaseModel):
    """Continuous movement request."""
    camera_id: str
    pan_velocity: float = 0.0
    tilt_velocity: float = 0.0
    zoom_velocity: float = 0.0
    timeout: Optional[float] = None


class AbsoluteMoveRequest(BaseModel):
    """Absolute position movement request."""
    camera_id: str
    pan: Optional[float] = None
    tilt: Optional[float] = None
    zoom: Optional[float] = None
    pan_speed: Optional[float] = None
    tilt_speed: Optional[float] = None
    zoom_speed: Optional[float] = None


class RelativeMoveRequest(BaseModel):
    """Relative movement request."""
    camera_id: str
    pan_delta: float = 0.0
    tilt_delta: float = 0.0
    zoom_delta: float = 0.0
    pan_speed: Optional[float] = None
    tilt_speed: Optional[float] = None
    zoom_speed: Optional[float] = None


class PresetRequest(BaseModel):
    """Preset operation request."""
    camera_id: str
    preset_name: Optional[str] = None
    preset_token: Optional[str] = None
    speed: Optional[float] = None


def _get_controller(camera_id: str):
    """Get PTZ controller for camera or raise error."""
    if camera_id not in _ptz_controllers:
        raise HTTPException(
            status_code=404,
            detail=f"PTZ controller not configured for camera '{camera_id}'. Use POST /api/ptz/connect first."
        )
    return _ptz_controllers[camera_id]


@router.post("/connect")
def connect_ptz(config: PTZConfig):
    """Connect to PTZ camera via ONVIF or HTTP.

    This initializes the PTZ controller for the specified camera.
    Set protocol='onvif' for ONVIF cameras or protocol='http' for HTTP CGI cameras.
    """
    try:
        if config.protocol.lower() == "onvif":
            # ONVIF protocol
            from services.ptz.onvif_ptz import ONVIFPTZController

            controller = ONVIFPTZController(
                host=config.host,
                port=config.port,
                username=config.username,
                password=config.password,
                no_ssl_verify=config.no_ssl_verify,
            )
        else:
            # HTTP CGI protocol
            from services.ptz.http_ptz import HTTPPTZController

            controller = HTTPPTZController(
                host=config.host,
                port=config.port,
                username=config.username,
                password=config.password,
                brand=config.brand,
                use_https=config.use_https,
            )

        _ptz_controllers[config.camera_id] = controller

        return {
            "ok": True,
            "message": f"Connected to PTZ camera '{config.camera_id}' at {config.host}:{config.port} using {config.protocol.upper()} protocol",
            "camera_id": config.camera_id,
            "protocol": config.protocol,
            "brand": config.brand if config.protocol == "http" else "N/A",
        }

    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail=f"PTZ support not available: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Failed to connect to PTZ camera: {e}")
        raise HTTPException(status_code=500, detail=f"PTZ connection failed: {str(e)}")


@router.post("/disconnect")
def disconnect_ptz(camera_id: str):
    """Disconnect from PTZ camera."""
    if camera_id in _ptz_controllers:
        del _ptz_controllers[camera_id]
        return {"ok": True, "message": f"Disconnected from PTZ camera '{camera_id}'"}
    else:
        raise HTTPException(status_code=404, detail=f"PTZ camera '{camera_id}' not connected")


@router.get("/status")
def get_ptz_status(camera_id: str):
    """Get PTZ status for camera."""
    controller = _get_controller(camera_id)
    return controller.get_status()


@router.post("/continuous_move")
def continuous_move(request: ContinuousMoveRequest):
    """Move PTZ continuously with specified velocities.

    Velocities range from -1.0 to 1.0:
    - pan_velocity: negative=left, positive=right
    - tilt_velocity: negative=down, positive=up
    - zoom_velocity: negative=zoom out, positive=zoom in
    - timeout: movement duration in seconds (None=continuous until stop)
    """
    controller = _get_controller(request.camera_id)
    return controller.continuous_move(
        pan_velocity=request.pan_velocity,
        tilt_velocity=request.tilt_velocity,
        zoom_velocity=request.zoom_velocity,
        timeout=request.timeout,
    )


@router.post("/stop")
def stop_ptz(camera_id: str):
    """Stop all PTZ movement."""
    controller = _get_controller(camera_id)
    return controller.stop()


@router.post("/absolute_move")
def absolute_move(request: AbsoluteMoveRequest):
    """Move PTZ to absolute position.

    Positions range from -1.0 to 1.0 for pan/tilt, 0.0 to 1.0 for zoom.
    Speeds range from 0.0 to 1.0.
    """
    controller = _get_controller(request.camera_id)
    return controller.absolute_move(
        pan=request.pan,
        tilt=request.tilt,
        zoom=request.zoom,
        pan_speed=request.pan_speed,
        tilt_speed=request.tilt_speed,
        zoom_speed=request.zoom_speed,
    )


@router.post("/relative_move")
def relative_move(request: RelativeMoveRequest):
    """Move PTZ relative to current position.

    Deltas range from -1.0 to 1.0.
    Speeds range from 0.0 to 1.0.
    """
    controller = _get_controller(request.camera_id)
    return controller.relative_move(
        pan_delta=request.pan_delta,
        tilt_delta=request.tilt_delta,
        zoom_delta=request.zoom_delta,
        pan_speed=request.pan_speed,
        tilt_speed=request.tilt_speed,
        zoom_speed=request.zoom_speed,
    )


@router.post("/goto_home")
def goto_home(camera_id: str, speed: Optional[float] = None):
    """Move PTZ to home position."""
    controller = _get_controller(camera_id)
    return controller.goto_home_position(speed=speed)


@router.post("/set_preset")
def set_preset(request: PresetRequest):
    """Save current PTZ position as a preset."""
    if not request.preset_name:
        raise HTTPException(status_code=400, detail="preset_name is required")

    controller = _get_controller(request.camera_id)
    return controller.set_preset(
        preset_name=request.preset_name,
        preset_token=request.preset_token,
    )


@router.post("/goto_preset")
def goto_preset(request: PresetRequest):
    """Move PTZ to a saved preset."""
    if not request.preset_token:
        raise HTTPException(status_code=400, detail="preset_token is required")

    controller = _get_controller(request.camera_id)
    return controller.goto_preset(
        preset_token=request.preset_token,
        speed=request.speed,
    )


@router.get("/presets")
def get_presets(camera_id: str):
    """Get list of saved presets for camera."""
    controller = _get_controller(camera_id)
    return controller.get_presets()


@router.get("/connected")
def list_connected_cameras():
    """List all connected PTZ cameras."""
    return {
        "ok": True,
        "cameras": list(_ptz_controllers.keys()),
        "count": len(_ptz_controllers),
    }
