"""Test script for PTZ camera control.

This script tests PTZ functionality with your RTSP camera using ONVIF protocol.
"""

import requests
import time
import json
from typing import Dict, Any

# Camera configuration from camera_settings.json
# Since ONVIF isn't supported, we'll try HTTP PTZ commands
CAMERA_CONFIG = {
    "camera_id": "cam1",
    "host": "IP_ADDRESS_OF_YOUR_CAMERA",
    "port": 80,
    "username": "admin",
    "password": "PASSWORD",
    "protocol": "http",  # Use 'http' instead of 'onvif'
    "brand": "hikvision",  # Try 'hikvision', 'dahua', or 'generic'
    "no_ssl_verify": True,
    "use_https": False,
}

API_BASE = "http://localhost:8000/api/ptz"


def print_response(name: str, response: requests.Response):
    """Pretty print API response."""
    print(f"\n{'=' * 60}")
    print(f"{name}")
    print(f"{'=' * 60}")
    print(f"Status: {response.status_code}")
    try:
        data = response.json()
        print(json.dumps(data, indent=2))
    except:
        print(response.text)


def test_connect(port=None):
    """Test connecting to PTZ camera."""
    config = CAMERA_CONFIG.copy()
    if port:
        config["port"] = port
        print(f"\n🔌 Testing PTZ Connection on port {port}...")
    else:
        print(f"\n🔌 Testing PTZ Connection on port {config['port']}...")

    response = requests.post(f"{API_BASE}/connect", json=config)
    print_response("Connect to PTZ Camera", response)
    return response.status_code == 200


def test_get_status():
    """Test getting PTZ status."""
    print("\n📊 Getting PTZ Status...")
    response = requests.get(
        f"{API_BASE}/status", params={"camera_id": CAMERA_CONFIG["camera_id"]}
    )
    print_response("PTZ Status", response)
    return response.json() if response.status_code == 200 else None


def test_continuous_move(
    pan_vel: float, tilt_vel: float, zoom_vel: float, duration: float = 2.0
):
    """Test continuous movement."""
    print(
        f"\n🎮 Testing Continuous Move (pan={pan_vel}, tilt={tilt_vel}, zoom={zoom_vel})..."
    )

    # Start movement
    response = requests.post(
        f"{API_BASE}/continuous_move",
        json={
            "camera_id": CAMERA_CONFIG["camera_id"],
            "pan_velocity": pan_vel,
            "tilt_velocity": tilt_vel,
            "zoom_velocity": zoom_vel,
            "timeout": duration,
        },
    )
    print_response("Continuous Move", response)

    # Wait for movement to complete
    time.sleep(duration + 0.5)

    return response.status_code == 200


def test_stop():
    """Test stopping movement."""
    print("\n🛑 Testing Stop...")
    response = requests.post(
        f"{API_BASE}/stop", params={"camera_id": CAMERA_CONFIG["camera_id"]}
    )
    print_response("Stop Movement", response)
    return response.status_code == 200


def test_absolute_move(pan: float = 0.0, tilt: float = 0.0, zoom: float = 0.5):
    """Test absolute positioning."""
    print(f"\n📍 Testing Absolute Move (pan={pan}, tilt={tilt}, zoom={zoom})...")
    response = requests.post(
        f"{API_BASE}/absolute_move",
        json={
            "camera_id": CAMERA_CONFIG["camera_id"],
            "pan": pan,
            "tilt": tilt,
            "zoom": zoom,
        },
    )
    print_response("Absolute Move", response)
    time.sleep(2)
    return response.status_code == 200


def test_relative_move(pan_delta: float, tilt_delta: float, zoom_delta: float = 0.0):
    """Test relative movement."""
    print(
        f"\n↔️ Testing Relative Move (Δpan={pan_delta}, Δtilt={tilt_delta}, Δzoom={zoom_delta})..."
    )
    response = requests.post(
        f"{API_BASE}/relative_move",
        json={
            "camera_id": CAMERA_CONFIG["camera_id"],
            "pan_delta": pan_delta,
            "tilt_delta": tilt_delta,
            "zoom_delta": zoom_delta,
        },
    )
    print_response("Relative Move", response)
    time.sleep(2)
    return response.status_code == 200


def test_home():
    """Test going to home position."""
    print("\n🏠 Testing Go Home...")
    response = requests.post(
        f"{API_BASE}/goto_home", params={"camera_id": CAMERA_CONFIG["camera_id"]}
    )
    print_response("Go Home", response)
    time.sleep(3)
    return response.status_code == 200


def test_presets():
    """Test preset operations."""
    print("\n💾 Testing Presets...")

    # Get existing presets
    response = requests.get(
        f"{API_BASE}/presets", params={"camera_id": CAMERA_CONFIG["camera_id"]}
    )
    print_response("Get Presets", response)

    presets = response.json().get("presets", []) if response.status_code == 200 else []

    # Set a new preset
    print("\n💾 Setting Preset 'test_position'...")
    response = requests.post(
        f"{API_BASE}/set_preset",
        json={
            "camera_id": CAMERA_CONFIG["camera_id"],
            "preset_name": "test_position",
        },
    )
    print_response("Set Preset", response)

    preset_token = (
        response.json().get("preset_token") if response.status_code == 200 else None
    )

    if preset_token:
        # Move away
        test_relative_move(0.2, 0.1)

        # Go back to preset
        print(f"\n📌 Going to Preset '{preset_token}'...")
        response = requests.post(
            f"{API_BASE}/goto_preset",
            json={
                "camera_id": CAMERA_CONFIG["camera_id"],
                "preset_token": preset_token,
            },
        )
        print_response("Go to Preset", response)
        time.sleep(2)


def test_scan_pattern():
    """Test a scanning pattern (left-right sweep)."""
    print("\n🔍 Testing Scan Pattern (left-right sweep)...")

    # Pan left
    print("  → Panning left...")
    test_continuous_move(pan_vel=-0.3, tilt_vel=0.0, zoom_vel=0.0, duration=2.0)

    # Pan right
    print("  → Panning right...")
    test_continuous_move(pan_vel=0.3, tilt_vel=0.0, zoom_vel=0.0, duration=4.0)

    # Pan back to center
    print("  → Returning to center...")
    test_continuous_move(pan_vel=-0.3, tilt_vel=0.0, zoom_vel=0.0, duration=2.0)

    test_stop()


def main():
    """Run PTZ tests."""
    print("=" * 60)
    print("PTZ Camera Test Script")
    print("=" * 60)
    print(f"\nCamera: {CAMERA_CONFIG['camera_id']}")
    print(f"Host: {CAMERA_CONFIG['host']}:{CAMERA_CONFIG['port']}")
    print(f"Username: {CAMERA_CONFIG['username']}")
    print("\nMake sure the API server is running:")
    print("  uvicorn api.server:app --host 0.0.0.0 --port 8000")
    print("\n" + "=" * 60)

    input("\nPress Enter to start tests...")

    # Test 1: Connect - try HTTP PTZ with multiple brands
    brands_to_try = ["hikvision", "dahua", "generic"]
    connected = False

    print("\nSince ONVIF isn't supported, trying HTTP PTZ commands...")

    for brand in brands_to_try:
        print(f"\nTrying {brand.upper()} PTZ commands...")
        CAMERA_CONFIG["brand"] = brand
        if test_connect():
            connected = True
            break

    if not connected:
        print("\n❌ Failed to connect to PTZ camera with HTTP commands.")
        print("\nPossible reasons:")
        print("  1. Camera doesn't support PTZ via HTTP CGI")
        print("  2. PTZ is disabled in camera settings")
        print("  3. Camera uses a proprietary PTZ protocol")
        print("\n📖 Next steps:")
        print("  1. Log into camera web UI: http://IP")
        print("  2. Check if PTZ controls work in the web interface")
        print("  3. Look for API documentation or PTZ control URLs")
        print("  4. Check if camera requires PTZ license/activation")
        return

    print("\n✅ PTZ connected successfully!")

    # Test 2: Get Status
    status = test_get_status()
    if status:
        print(f"\n✅ Current Position:")
        print(f"   Pan: {status.get('position', {}).get('pan', 'N/A')}")
        print(f"   Tilt: {status.get('position', {}).get('tilt', 'N/A')}")
        print(f"   Zoom: {status.get('position', {}).get('zoom', 'N/A')}")

    input("\nPress Enter to test continuous movement (pan right)...")
    test_continuous_move(pan_vel=0.3, tilt_vel=0.0, zoom_vel=0.0, duration=3.0)
    test_get_status()

    input("\nPress Enter to test continuous movement (tilt up)...")
    test_continuous_move(pan_vel=0.0, tilt_vel=0.3, zoom_vel=0.0, duration=2.0)
    test_get_status()

    input("\nPress Enter to test zoom in...")
    test_continuous_move(pan_vel=0.0, tilt_vel=0.0, zoom_vel=0.3, duration=2.0)
    test_get_status()

    input("\nPress Enter to test zoom out...")
    test_continuous_move(pan_vel=0.0, tilt_vel=0.0, zoom_vel=-0.3, duration=2.0)
    test_get_status()

    input("\nPress Enter to return to home position...")
    test_home()
    test_get_status()

    input("\nPress Enter to test absolute positioning (center)...")
    test_absolute_move(pan=0.0, tilt=0.0, zoom=0.5)
    test_get_status()

    input("\nPress Enter to test relative movement...")
    test_relative_move(pan_delta=0.1, tilt_delta=0.05)
    test_get_status()

    input("\nPress Enter to test presets...")
    test_presets()

    input("\nPress Enter to run scan pattern...")
    test_scan_pattern()

    # Return to home
    print("\n🏠 Returning to home position...")
    test_home()

    print("\n" + "=" * 60)
    print("✅ PTZ Tests Complete!")
    print("=" * 60)

    # List connected cameras
    response = requests.get(f"{API_BASE}/connected")
    if response.status_code == 200:
        print(f"\nConnected PTZ cameras: {response.json().get('cameras', [])}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Tests interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
