"""Check if camera actually has PTZ capabilities.

This script determines if the camera is a PTZ camera or a fixed camera.
"""

import requests
from requests.auth import HTTPDigestAuth, HTTPBasicAuth
import xml.etree.ElementTree as ET

CAMERA_IP = "IP_ADDRESS_OF_YOUR_CAMERA"  # Replace with your camera's IP address
USERNAME = "admin"
PASSWORD = "YOUR_PASSWORD"

auth_digest = HTTPDigestAuth(USERNAME, PASSWORD)
auth_basic = HTTPBasicAuth(USERNAME, PASSWORD)


def test_url(url, description):
    """Test a URL with both auth methods."""
    print(f"\nTesting: {description}")
    print(f"URL: {url}")

    # Try digest auth
    try:
        response = requests.get(url, auth=auth_digest, timeout=5, verify=False)
        print(f"  Digest Auth - Status: {response.status_code}")
        if response.status_code == 200:
            print(f"  Content preview: {response.text[:200]}")
            return response
    except Exception as e:
        print(f"  Digest Auth - Error: {e}")

    # Try basic auth
    try:
        response = requests.get(url, auth=auth_basic, timeout=5, verify=False)
        print(f"  Basic Auth - Status: {response.status_code}")
        if response.status_code == 200:
            print(f"  Content preview: {response.text[:200]}")
            return response
    except Exception as e:
        print(f"  Basic Auth - Error: {e}")

    return None


def main():
    print("=" * 70)
    print("PTZ Capability Check")
    print("=" * 70)
    print(f"\nCamera: {CAMERA_IP}")
    print(f"Checking if this camera has PTZ capabilities...")

    # Try different PTZ capability detection URLs
    capability_urls = [
        # Hikvision
        (
            f"http://{CAMERA_IP}/ISAPI/PTZCtrl/channels/1/capabilities",
            "Hikvision PTZ Capabilities",
        ),
        (f"http://{CAMERA_IP}/ISAPI/System/deviceInfo", "Hikvision Device Info"),
        (f"http://{CAMERA_IP}/ISAPI/PTZCtrl/channels/1/status", "Hikvision PTZ Status"),
        # Dahua
        (
            f"http://{CAMERA_IP}/cgi-bin/devVideoInput.cgi?action=getCaps",
            "Dahua Video Capabilities",
        ),
        (
            f"http://{CAMERA_IP}/cgi-bin/magicBox.cgi?action=getDeviceType",
            "Dahua Device Type",
        ),
        (
            f"http://{CAMERA_IP}/cgi-bin/ptz.cgi?action=getCurrentProtocolCaps",
            "Dahua PTZ Caps",
        ),
        # Generic
        (
            f"http://{CAMERA_IP}/cgi-bin/param.cgi?action=list&group=PTZ",
            "Generic PTZ Params",
        ),
        (f"http://{CAMERA_IP}/api/system/deviceInfo", "Generic Device Info"),
    ]

    print("\n" + "=" * 70)
    print("Checking PTZ Capability URLs")
    print("=" * 70)

    found_info = False
    for url, desc in capability_urls:
        response = test_url(url, desc)
        if response and response.status_code == 200:
            found_info = True

            # Try to parse XML for Hikvision
            if "ISAPI" in url:
                try:
                    root = ET.fromstring(response.text)
                    print(f"\n  📋 Parsed XML:")
                    for child in root:
                        print(f"    {child.tag}: {child.text}")
                except:
                    pass

    if not found_info:
        print("\n❌ Could not retrieve device capabilities")

    # Try some actual PTZ movement commands
    print("\n" + "=" * 70)
    print("Testing Actual PTZ Commands")
    print("=" * 70)

    ptz_commands = [
        # Hikvision continuous move
        (
            f"http://{CAMERA_IP}/ISAPI/PTZCtrl/channels/1/continuous",
            "PUT",
            "<?xml version='1.0'?><PTZData><pan>50</pan><tilt>0</tilt><zoom>0</zoom></PTZData>",
            "Hikvision Continuous Move (Pan Right)",
        ),
        # Hikvision momentary
        (
            f"http://{CAMERA_IP}/ISAPI/PTZCtrl/channels/1/momentary",
            "PUT",
            "<?xml version='1.0'?><PTZData><pan>50</pan><tilt>0</tilt><zoom>0</zoom></PTZData>",
            "Hikvision Momentary Move (Pan Right)",
        ),
        # Dahua
        (
            f"http://{CAMERA_IP}/cgi-bin/ptz.cgi?action=start&code=Right&arg1=0&arg2=5&arg3=0",
            "GET",
            None,
            "Dahua Pan Right",
        ),
        # Generic
        (
            f"http://{CAMERA_IP}/cgi-bin/ptz.cgi?move=right&speed=50",
            "GET",
            None,
            "Generic Pan Right",
        ),
    ]

    print("\n⚠️  WATCH YOUR CAMERA - It may move during these tests!")
    input("Press Enter to test PTZ commands...")

    for url, method, data, desc in ptz_commands:
        print(f"\n{desc}")
        print(f"  {method} {url}")

        try:
            if method == "PUT":
                # Try digest
                response = requests.put(
                    url, data=data, auth=auth_digest, timeout=5, verify=False
                )
                print(f"  Digest - Status: {response.status_code}")

                if response.status_code not in [200, 204]:
                    # Try basic
                    response = requests.put(
                        url, data=data, auth=auth_basic, timeout=5, verify=False
                    )
                    print(f"  Basic - Status: {response.status_code}")
            else:
                # Try digest
                response = requests.get(url, auth=auth_digest, timeout=5, verify=False)
                print(f"  Digest - Status: {response.status_code}")

                if response.status_code not in [200, 204]:
                    # Try basic
                    response = requests.get(
                        url, auth=auth_basic, timeout=5, verify=False
                    )
                    print(f"  Basic - Status: {response.status_code}")

            if response.status_code in [200, 204]:
                print(f"  ✅ Command accepted!")
                print(f"  👀 Did the camera move?")
                import time

                time.sleep(2)

        except Exception as e:
            print(f"  Error: {e}")

    # Stop command
    print("\nSending STOP command...")
    stop_urls = [
        f"http://{CAMERA_IP}/ISAPI/PTZCtrl/channels/1/continuous",
        f"http://{CAMERA_IP}/cgi-bin/ptz.cgi?action=stop",
    ]

    for url in stop_urls:
        try:
            stop_data = "<?xml version='1.0'?><PTZData><pan>0</pan><tilt>0</tilt><zoom>0</zoom></PTZData>"
            requests.put(url, data=stop_data, auth=auth_digest, timeout=5, verify=False)
        except:
            pass

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nBased on the tests:")
    print("1. If the camera DID NOT move, this is likely a FIXED camera (no PTZ)")
    print("2. If the camera DID move, note which command worked and we can use that")
    print("3. If some commands returned 200 OK but nothing moved, PTZ may be disabled")

    print("\n📖 Next steps:")
    print("1. Log into camera web UI: http://IP")
    print("2. Look for PTZ controls or settings")
    print("3. Check camera model/documentation to confirm PTZ support")
    print("4. If camera is fixed, PTZ control is not possible")


if __name__ == "__main__":
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
