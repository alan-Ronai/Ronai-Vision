"""Scan all ports to find PTZ control interface.

This script tests PTZ commands on multiple ports to find the correct one.
"""

import requests
from requests.auth import HTTPDigestAuth, HTTPBasicAuth
import socket
import sys

CAMERA_IP = "IP_ADDRESS_OF_YOUR_CAMERA"  # Replace with your camera's IP address
USERNAME = "admin"
PASSWORD = "PASSWORD_OF_YOUR_CAMERA"

# Extended list of ports to try
PORTS_TO_SCAN = [
    80,  # Standard HTTP
    8080,  # Alternative HTTP
    8081,  # Alternative HTTP
    8000,  # Alternative HTTP
    8888,  # Common camera port
    8899,  # Common camera port
    9000,  # Alternative
    554,  # RTSP (sometimes has HTTP interface)
    5000,  # Alternative
    8443,  # HTTPS alternative
    443,  # HTTPS
]


def test_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """Check if port is open."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except:
        return False


def test_ptz_on_port(host: str, port: int, use_https: bool = False) -> dict:
    """Test PTZ functionality on a specific port."""
    protocol = "https" if use_https else "http"
    auth_digest = HTTPDigestAuth(USERNAME, PASSWORD)
    auth_basic = HTTPBasicAuth(USERNAME, PASSWORD)

    results = {
        "port": port,
        "open": False,
        "http_works": False,
        "device_info": None,
        "ptz_status": None,
        "ptz_move_works": None,
    }

    # Test if port is open
    results["open"] = test_port_open(host, port)
    if not results["open"]:
        return results

    print(f"  Port {port} is OPEN, testing HTTP...")

    # Test basic HTTP connectivity
    try:
        url = f"{protocol}://{host}:{port}/"
        response = requests.get(url, auth=auth_digest, timeout=3, verify=False)
        if response.status_code in [200, 401, 403]:
            results["http_works"] = True
            print(f"  Port {port} has HTTP service")
    except:
        pass

    # Test Hikvision device info
    try:
        url = f"{protocol}://{host}:{port}/ISAPI/System/deviceInfo"
        for auth in [auth_digest, auth_basic]:
            response = requests.get(url, auth=auth, timeout=3, verify=False)
            if response.status_code == 200:
                results["device_info"] = response.text[:300]
                print(f"  ✅ Port {port}: Device info found!")
                break
    except:
        pass

    # Test PTZ status
    try:
        url = f"{protocol}://{host}:{port}/ISAPI/PTZCtrl/channels/1/status"
        for auth in [auth_digest, auth_basic]:
            response = requests.get(url, auth=auth, timeout=3, verify=False)
            if response.status_code == 200:
                results["ptz_status"] = response.text[:300]
                print(f"  ✅ Port {port}: PTZ status endpoint found!")
                break
    except:
        pass

    # Test actual PTZ movement (Hikvision)
    try:
        url = f"{protocol}://{host}:{port}/ISAPI/PTZCtrl/channels/1/continuous"
        data = "<?xml version='1.0'?><PTZData><pan>50</pan><tilt>0</tilt><zoom>0</zoom></PTZData>"
        for auth in [auth_digest, auth_basic]:
            response = requests.put(url, data=data, auth=auth, timeout=3, verify=False)
            if response.status_code in [200, 204]:
                results["ptz_move_works"] = True
                print(f"  ✅ Port {port}: PTZ move command accepted!")

                # Stop movement
                stop_data = "<?xml version='1.0'?><PTZData><pan>0</pan><tilt>0</tilt><zoom>0</zoom></PTZData>"
                requests.put(url, data=stop_data, auth=auth, timeout=3, verify=False)
                break
    except:
        pass

    # Test Dahua PTZ
    if not results["ptz_move_works"]:
        try:
            url = f"{protocol}://{host}:{port}/cgi-bin/ptz.cgi"
            params = {
                "action": "start",
                "code": "Right",
                "arg1": 0,
                "arg2": 5,
                "arg3": 0,
            }
            for auth in [auth_digest, auth_basic]:
                response = requests.get(
                    url, params=params, auth=auth, timeout=3, verify=False
                )
                if response.status_code in [200, 204]:
                    results["ptz_move_works"] = True
                    print(f"  ✅ Port {port}: Dahua PTZ command accepted!")

                    # Stop
                    stop_params = {"action": "stop"}
                    requests.get(
                        url, params=stop_params, auth=auth, timeout=3, verify=False
                    )
                    break
        except:
            pass

    return results


def main():
    print("=" * 70)
    print("PTZ Port Scanner")
    print("=" * 70)
    print(f"\nCamera: {CAMERA_IP}")
    print(f"Scanning {len(PORTS_TO_SCAN)} ports for PTZ control interface...")
    print(f"Username: {USERNAME}")
    print(f"Password: {'*' * len(PASSWORD)}")

    print("\n" + "=" * 70)
    print("Scanning Ports")
    print("=" * 70)

    findings = []

    for port in PORTS_TO_SCAN:
        sys.stdout.write(f"\rScanning port {port}...         ")
        sys.stdout.flush()

        # Try HTTP first
        result = test_ptz_on_port(CAMERA_IP, port, use_https=False)
        if (
            result["http_works"]
            or result["device_info"]
            or result["ptz_status"]
            or result["ptz_move_works"]
        ):
            findings.append(result)

        # If port is 443 or 8443, also try HTTPS
        if port in [443, 8443]:
            result_https = test_ptz_on_port(CAMERA_IP, port, use_https=True)
            if (
                result_https["http_works"]
                or result_https["device_info"]
                or result_https["ptz_status"]
                or result_https["ptz_move_works"]
            ):
                findings.append(result_https)

    print("\r" + " " * 50)  # Clear line

    # Summary
    print("\n" + "=" * 70)
    print("SCAN RESULTS")
    print("=" * 70)

    if not findings:
        print("\n❌ No PTZ interfaces found on any port")
        print("\nThis strongly indicates:")
        print("  1. Camera is a FIXED camera (no PTZ hardware)")
        print("  2. PTZ control is on a non-standard port we didn't scan")
        print("  3. PTZ requires different authentication")
        return

    print(f"\n✅ Found {len(findings)} potential PTZ interface(s):\n")

    for i, finding in enumerate(findings, 1):
        print(f"{i}. Port {finding['port']}")
        print(f"   HTTP Service: {'Yes' if finding['http_works'] else 'No'}")
        print(f"   Device Info: {'Found' if finding['device_info'] else 'Not found'}")
        print(f"   PTZ Status: {'Found' if finding['ptz_status'] else 'Not found'}")
        print(f"   PTZ Move: {'Works' if finding['ptz_move_works'] else 'Not tested'}")

        if finding["device_info"]:
            print(f"   Device Info Preview: {finding['device_info'][:100]}...")
        print()

    # Check if camera actually moved
    print("=" * 70)
    print("MOVEMENT TEST")
    print("=" * 70)
    print("\n❓ Did the camera physically move during the scan?")
    print("   If YES: Note which port worked (see above)")
    print("   If NO: This is likely a fixed camera")

    # Recommendations
    best_port = None
    for finding in findings:
        if finding["ptz_move_works"]:
            best_port = finding["port"]
            break

    if best_port:
        print(f"\n✅ RECOMMENDED: Use port {best_port} for PTZ control")
        print(f"\nUpdate your configuration:")
        print(f"  Host: {CAMERA_IP}")
        print(f"  Port: {best_port}")
        print(f"  Protocol: http")
        print(f"  Brand: hikvision")
    else:
        print("\n⚠️  Camera accepts PTZ commands but may not have PTZ hardware")
        print("   Commands return 200 OK but camera doesn't move")


if __name__ == "__main__":
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Scan interrupted")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
