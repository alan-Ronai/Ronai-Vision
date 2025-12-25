"""Webcam utilities for detecting available webcam devices.

Webcams are streamed through go2rtc using FFmpeg device capture.
This module provides utilities for listing available devices.
"""

import cv2
import platform
import logging
import subprocess
import re
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def get_webcam_capabilities(device_index: int = 0) -> Dict:
    """Get webcam capabilities including supported resolutions and framerates.

    Args:
        device_index: Webcam device index

    Returns:
        Dict with capabilities: {
            "resolutions": [(1280, 720), (1920, 1080), ...],
            "framerates": [15, 30],
            "best_resolution": (1280, 720),
            "best_framerate": 30
        }
    """
    system = platform.system()
    capabilities = {
        "resolutions": [],
        "framerates": [],
        "best_resolution": (1280, 720),  # Default
        "best_framerate": 30  # Default
    }

    if system == "Darwin":
        capabilities = _get_capabilities_macos(device_index)
    elif system == "Linux":
        capabilities = _get_capabilities_linux(device_index)
    else:
        capabilities = _get_capabilities_opencv(device_index)

    return capabilities


def _get_capabilities_macos(device_index: int) -> Dict:
    """Get webcam capabilities on macOS using FFmpeg."""
    capabilities = {
        "resolutions": [],
        "framerates": [],
        "best_resolution": (1280, 720),
        "best_framerate": 30
    }

    try:
        # Use FFmpeg to list device capabilities
        result = subprocess.run(
            ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            capture_output=True,
            text=True,
            timeout=5
        )

        # Now get the actual formats for this device
        result = subprocess.run(
            ["ffmpeg", "-f", "avfoundation", "-framerate", "30", "-video_size", "1280x720",
             "-i", f"{device_index}:none", "-t", "0.001", "-f", "null", "-"],
            capture_output=True,
            text=True,
            timeout=10
        )

        # Parse supported modes from error output
        # Format: "1280x720@[15.000000 30.000000]fps"
        modes = re.findall(r'(\d+)x(\d+)@\[([\d.]+)\s+([\d.]+)\]fps', result.stderr)

        resolutions = set()
        framerates = set()

        for mode in modes:
            width, height, min_fps, max_fps = mode
            resolutions.add((int(width), int(height)))
            framerates.add(int(float(min_fps)))
            framerates.add(int(float(max_fps)))

        if resolutions:
            capabilities["resolutions"] = sorted(list(resolutions), key=lambda x: x[0] * x[1], reverse=True)
            # Prefer 1280x720 if available, otherwise highest resolution
            if (1280, 720) in resolutions:
                capabilities["best_resolution"] = (1280, 720)
            else:
                capabilities["best_resolution"] = capabilities["resolutions"][0]

        if framerates:
            capabilities["framerates"] = sorted(list(framerates), reverse=True)
            # Prefer 30fps if available, otherwise highest
            if 30 in framerates:
                capabilities["best_framerate"] = 30
            else:
                capabilities["best_framerate"] = capabilities["framerates"][0]

        logger.info(f"Webcam {device_index} capabilities: {capabilities}")

    except Exception as e:
        logger.warning(f"Failed to get webcam capabilities: {e}, using defaults")

    return capabilities


def _get_capabilities_linux(device_index: int) -> Dict:
    """Get webcam capabilities on Linux using v4l2-ctl."""
    capabilities = {
        "resolutions": [],
        "framerates": [],
        "best_resolution": (1280, 720),
        "best_framerate": 30
    }

    try:
        device_path = f"/dev/video{device_index}"

        # Get supported formats
        result = subprocess.run(
            ["v4l2-ctl", "--device", device_path, "--list-formats-ext"],
            capture_output=True,
            text=True,
            timeout=5
        )

        # Parse resolutions and framerates
        # Format: "Size: Discrete 1280x720" and "Interval: Discrete 0.033s (30.000 fps)"
        resolutions = set()
        framerates = set()

        current_resolution = None
        for line in result.stdout.split('\n'):
            size_match = re.search(r'Size:\s+Discrete\s+(\d+)x(\d+)', line)
            if size_match:
                current_resolution = (int(size_match.group(1)), int(size_match.group(2)))
                resolutions.add(current_resolution)

            fps_match = re.search(r'\((\d+\.?\d*)\s*fps\)', line)
            if fps_match:
                framerates.add(int(float(fps_match.group(1))))

        if resolutions:
            capabilities["resolutions"] = sorted(list(resolutions), key=lambda x: x[0] * x[1], reverse=True)
            if (1280, 720) in resolutions:
                capabilities["best_resolution"] = (1280, 720)
            else:
                capabilities["best_resolution"] = capabilities["resolutions"][0]

        if framerates:
            capabilities["framerates"] = sorted(list(framerates), reverse=True)
            if 30 in framerates:
                capabilities["best_framerate"] = 30
            else:
                capabilities["best_framerate"] = capabilities["framerates"][0]

        logger.info(f"Webcam {device_index} capabilities: {capabilities}")

    except Exception as e:
        logger.warning(f"Failed to get webcam capabilities: {e}, using defaults")

    return capabilities


def _get_capabilities_opencv(device_index: int) -> Dict:
    """Get webcam capabilities using OpenCV (fallback for all platforms)."""
    capabilities = {
        "resolutions": [],
        "framerates": [],
        "best_resolution": (1280, 720),
        "best_framerate": 30
    }

    try:
        cap = cv2.VideoCapture(device_index)
        if cap.isOpened():
            # Try to get current/default values
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            if width > 0 and height > 0:
                capabilities["best_resolution"] = (width, height)
                capabilities["resolutions"] = [(width, height)]

            if fps > 0:
                capabilities["best_framerate"] = fps
                capabilities["framerates"] = [fps]

            cap.release()

            logger.info(f"Webcam {device_index} capabilities (OpenCV): {capabilities}")

    except Exception as e:
        logger.warning(f"Failed to get webcam capabilities via OpenCV: {e}")

    return capabilities


def get_best_webcam_settings(device_index: int = 0) -> Tuple[int, int, int]:
    """Get the best resolution and framerate for a webcam.

    Args:
        device_index: Webcam device index

    Returns:
        Tuple of (width, height, fps)
    """
    caps = get_webcam_capabilities(device_index)
    width, height = caps["best_resolution"]
    fps = caps["best_framerate"]
    return width, height, fps


def list_available_webcams(max_devices: int = 10) -> List[Dict]:
    """List available webcam devices.

    Returns:
        List of dicts with device info: [{"index": 0, "name": "Device 0", "available": True}, ...]
    """
    devices = []
    system = platform.system()

    if system == "Darwin":
        # macOS - try to get device names from system_profiler or use generic names
        devices = _list_webcams_macos(max_devices)
    elif system == "Linux":
        # Linux - check /dev/video* devices
        devices = _list_webcams_linux(max_devices)
    elif system == "Windows":
        # Windows - use OpenCV probe
        devices = _list_webcams_opencv(max_devices)
    else:
        # Fallback to OpenCV probe
        devices = _list_webcams_opencv(max_devices)

    return devices


def _list_webcams_opencv(max_devices: int) -> List[Dict]:
    """List webcams using OpenCV probe (works on all platforms)."""
    devices = []

    for i in range(max_devices):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Try to read a frame to verify it works
            ret, _ = cap.read()
            devices.append({
                "index": i,
                "name": f"Webcam {i}",
                "available": ret,
                "device_path": str(i)
            })
            cap.release()
        else:
            # Stop at first unavailable device
            break

    return devices


def _list_webcams_macos(max_devices: int) -> List[Dict]:
    """List webcams on macOS."""
    devices = []

    # First try OpenCV to see what's available
    for i in range(max_devices):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            # On macOS, device names are like "FaceTime HD Camera"
            # but we access them by index
            name = f"Camera {i}"

            # Try to get actual device name using ffmpeg
            try:
                result = subprocess.run(
                    ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                # Parse output to find device names
                lines = result.stderr.split('\n')
                video_section = False
                device_idx = 0
                for line in lines:
                    if "AVFoundation video devices" in line:
                        video_section = True
                        continue
                    if "AVFoundation audio devices" in line:
                        video_section = False
                        continue
                    if video_section and "[" in line and "]" in line:
                        if device_idx == i:
                            # Extract device name
                            start = line.find("]") + 2
                            name = line[start:].strip()
                            break
                        device_idx += 1
            except Exception:
                pass

            devices.append({
                "index": i,
                "name": name,
                "available": ret,
                "device_path": str(i)
            })
            cap.release()
        else:
            break

    return devices


def _list_webcams_linux(max_devices: int) -> List[Dict]:
    """List webcams on Linux by checking /dev/video* devices."""
    import os
    devices = []

    for i in range(max_devices):
        device_path = f"/dev/video{i}"
        if os.path.exists(device_path):
            # Try to open with OpenCV
            cap = cv2.VideoCapture(i)
            available = False
            if cap.isOpened():
                ret, _ = cap.read()
                available = ret
                cap.release()

            # Try to get device name from v4l2
            name = f"Video Device {i}"
            try:
                result = subprocess.run(
                    ["v4l2-ctl", "--device", device_path, "--info"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                for line in result.stdout.split('\n'):
                    if "Card type" in line:
                        name = line.split(":")[-1].strip()
                        break
            except Exception:
                pass

            devices.append({
                "index": i,
                "name": name,
                "available": available,
                "device_path": device_path
            })

    return devices


def get_ffmpeg_device_source(device_index: int = 0) -> str:
    """Get the FFmpeg device source string for go2rtc.

    Args:
        device_index: Webcam device index (0 = first webcam)

    Returns:
        FFmpeg source string for go2rtc (e.g., "ffmpeg:-f avfoundation -i 0")
    """
    system = platform.system()

    if system == "Darwin":
        # macOS uses AVFoundation
        # Format: ffmpeg:-f avfoundation -framerate 30 -video_size 1280x720 -i "0"
        return f'ffmpeg:-f avfoundation -framerate 15 -video_size 1280x720 -i "{device_index}"#video=h264'

    elif system == "Linux":
        # Linux uses v4l2
        # Format: ffmpeg:-f v4l2 -input_format mjpeg -video_size 1280x720 -i /dev/video0
        return f'ffmpeg:-f v4l2 -input_format mjpeg -framerate 15 -video_size 1280x720 -i /dev/video{device_index}#video=h264'

    elif system == "Windows":
        # Windows uses dshow
        # Note: On Windows, you often need the device name, not index
        # We'll try with "video=0" format first
        return f'ffmpeg:-f dshow -framerate 15 -video_size 1280x720 -i video="USB Video Device"#video=h264'

    else:
        # Fallback - try v4l2 style
        return f'ffmpeg:-f v4l2 -i /dev/video{device_index}#video=h264'


def get_webcam_source_for_go2rtc(device_index: int = 0, width: int = 1280, height: int = 720, fps: int = 15) -> str:
    """Get the complete go2rtc source configuration for a webcam.

    go2rtc uses a special device syntax: ffmpeg:device?{params}#{output}
    This is NOT a raw FFmpeg command line - go2rtc handles platform-specific
    FFmpeg commands internally (avfoundation on macOS, v4l2 on Linux, dshow on Windows).

    See: https://github.com/AlexxIT/go2rtc

    Args:
        device_index: Webcam device index
        width: Video width
        height: Video height
        fps: Frames per second

    Returns:
        go2rtc source string for webcam device capture
    """
    # go2rtc unified device syntax - works on all platforms
    # Format: ffmpeg:device?video={index}&video_size={w}x{h}&framerate={fps}#video=h264
    return f'ffmpeg:device?video={device_index}&video_size={width}x{height}&framerate={fps}#video=h264'
