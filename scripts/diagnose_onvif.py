"""Diagnostic script to detect ONVIF port and capabilities.

This script helps identify if a camera supports ONVIF and which port it uses.
"""

import socket
import sys
from urllib.parse import urlparse

# Camera details
CAMERA_IP = "IP_ADDRESS_OF_YOUR_CAMERA"  # Replace with your camera's IP address
RTSP_URL = "URL_to_your_RTSP_stream_1"
USERNAME = "admin"
PASSWORD = "PASSWORD"

# Common ONVIF ports to test
ONVIF_PORTS = [80, 8080, 8899, 8000, 554, 8081, 9000, 5000, 8888]


def test_port(host: str, port: int, timeout: float = 2.0) -> bool:
    """Test if a port is open."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception as e:
        return False


def test_http_endpoint(
    host: str, port: int, path: str = "/onvif/device_service"
) -> tuple:
    """Test if HTTP endpoint responds."""
    try:
        import requests

        url = f"http://{host}:{port}{path}"
        response = requests.get(url, timeout=3)
        return True, response.status_code, response.text[:200]
    except Exception as e:
        return False, None, str(e)


def test_onvif_connection(host: str, port: int, username: str, password: str) -> tuple:
    """Test actual ONVIF connection."""
    try:
        from onvif import ONVIFCamera
        import ssl

        # Disable SSL verification
        ssl._create_default_https_context = ssl._create_unverified_context

        camera = ONVIFCamera(host, port, username, password)
        device_service = camera.create_devicemgmt_service()
        device_info = device_service.GetDeviceInformation()

        return True, {
            "manufacturer": device_info.Manufacturer,
            "model": device_info.Model,
            "firmware": device_info.FirmwareVersion,
            "serial": device_info.SerialNumber,
        }
    except Exception as e:
        return False, str(e)


def main():
    print("=" * 70)
    print("ONVIF Camera Diagnostic Tool")
    print("=" * 70)
    print(f"\nCamera IP: {CAMERA_IP}")
    print(f"RTSP URL: {RTSP_URL}")
    print(f"Username: {USERNAME}")
    print(f"Password: {'*' * len(PASSWORD)}")

    # Step 1: Test basic connectivity
    print("\n" + "=" * 70)
    print("Step 1: Testing basic connectivity")
    print("=" * 70)

    try:
        import subprocess

        result = subprocess.run(
            ["ping", "-c", "1", CAMERA_IP], capture_output=True, timeout=5
        )
        if result.returncode == 0:
            print(f"✅ Camera is reachable at {CAMERA_IP}")
        else:
            print(f"❌ Camera is NOT reachable at {CAMERA_IP}")
            print("   Check network connection and camera IP address")
            return
    except Exception as e:
        print(f"⚠️  Could not test ping: {e}")

    # Step 2: Test RTSP port (554)
    print("\n" + "=" * 70)
    print("Step 2: Testing RTSP port (554)")
    print("=" * 70)

    if test_port(CAMERA_IP, 554):
        print("✅ RTSP port 554 is OPEN")
        print("   This confirms the camera is online and streaming")
    else:
        print("❌ RTSP port 554 is CLOSED or filtered")
        print("   Camera may be offline or behind a firewall")

    # Step 3: Scan for open ports
    print("\n" + "=" * 70)
    print("Step 3: Scanning common ONVIF ports")
    print("=" * 70)

    open_ports = []
    for port in ONVIF_PORTS:
        sys.stdout.write(f"\rTesting port {port}...      ")
        sys.stdout.flush()
        if test_port(CAMERA_IP, port, timeout=1.0):
            print(f"\r✅ Port {port} is OPEN")
            open_ports.append(port)
        else:
            print(f"\r   Port {port} is closed", end="")

    print("\n")

    if not open_ports:
        print("❌ No common ONVIF ports are open!")
        print("\nPossible reasons:")
        print("  1. Camera doesn't support ONVIF")
        print("  2. ONVIF is disabled in camera settings")
        print("  3. Camera uses a non-standard port")
        print("  4. Firewall is blocking ONVIF ports")
        print("\n📖 Next steps:")
        print("  1. Log into camera web interface")
        print("  2. Look for 'ONVIF' settings")
        print("  3. Enable ONVIF if disabled")
        print("  4. Check which port ONVIF uses")
        return

    print(f"\n✅ Found {len(open_ports)} open port(s): {open_ports}")

    # Step 4: Test HTTP endpoints
    print("\n" + "=" * 70)
    print("Step 4: Testing HTTP ONVIF endpoints")
    print("=" * 70)

    onvif_paths = [
        "/onvif/device_service",
        "/onvif/services",
        "/onvif-http/snapshot",
        "/cgi-bin/snapshot.cgi",
    ]

    for port in open_ports:
        print(f"\nTesting port {port}:")
        for path in onvif_paths:
            success, status, text = test_http_endpoint(CAMERA_IP, port, path)
            if success:
                print(f"  ✅ {path} - Status {status}")
            else:
                print(f"     {path} - Not accessible")

    # Step 5: Test actual ONVIF connection
    print("\n" + "=" * 70)
    print("Step 5: Testing ONVIF authentication")
    print("=" * 70)

    try:
        from onvif import ONVIFCamera
    except ImportError:
        print("❌ python-onvif-zeep not installed")
        print("   Install with: pip install onvif-zeep")
        return

    for port in open_ports:
        print(f"\nTesting ONVIF on port {port}...")
        success, result = test_onvif_connection(CAMERA_IP, port, USERNAME, PASSWORD)

        if success:
            print(f"✅ ONVIF connection successful on port {port}!")
            print(f"\n📷 Camera Information:")
            print(f"   Manufacturer: {result['manufacturer']}")
            print(f"   Model: {result['model']}")
            print(f"   Firmware: {result['firmware']}")
            print(f"   Serial: {result['serial']}")
            print(f"\n✅ Use this configuration:")
            print(f"   Host: {CAMERA_IP}")
            print(f"   Port: {port}")
            print(f"   Username: {USERNAME}")
            print(f"   Password: {PASSWORD}")
            return
        else:
            print(f"   ❌ ONVIF failed on port {port}: {result[:100]}")

    # Summary
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)
    print("\n❌ ONVIF connection failed on all tested ports")
    print("\nPossible issues:")
    print("  1. Camera doesn't support ONVIF protocol")
    print("  2. ONVIF authentication credentials are different from RTSP")
    print("  3. ONVIF is disabled in camera settings")
    print("  4. Camera uses a proprietary API instead of ONVIF")

    print("\n📖 Recommended actions:")
    print("  1. Access camera web interface: http://IP")
    print("  2. Check camera documentation for ONVIF support")
    print("  3. Look for 'ONVIF' or 'Network Settings' in web UI")
    print("  4. Try creating a separate ONVIF user account")
    print("  5. Check if camera requires ONVIF license/activation")

    print("\n💡 Alternative: Use RTSP camera controls")
    print("   Some cameras support PTZ via RTSP commands instead of ONVIF")
    print("   Check camera documentation for RTSP PTZ control URLs")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Diagnostic interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
