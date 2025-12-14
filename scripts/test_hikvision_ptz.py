#!/usr/bin/env python3
"""Quick PTZ test for Hikvision DS-MH6171.

Simple script to test PTZ commands without running full diagnostics.
"""

import sys
import requests
from requests.auth import HTTPDigestAuth
import time


def test_ptz(host, username="admin", password=""):
    """Test PTZ commands."""
    print("=" * 60)
    print("Hikvision DS-MH6171 PTZ Quick Test")
    print("=" * 60)
    print(f"Host: {host}")
    print(f"User: {username}")
    print()

    auth = HTTPDigestAuth(username, password)
    base_url = f"http://{host}"

    # Test device info
    print("[1/4] Testing connection...")
    try:
        response = requests.get(
            f"{base_url}/ISAPI/System/deviceInfo", auth=auth, timeout=5, verify=False
        )
        if response.status_code == 200:
            print("✅ Connection successful")
            # Try to parse model
            if "DS-MH6171" in response.text:
                print("✅ Confirmed DS-MH6171 Mobile DVR")
        else:
            print(f"❌ Connection failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

    # Test PTZ capabilities
    print("\n[2/4] Checking PTZ capabilities...")
    try:
        response = requests.get(
            f"{base_url}/ISAPI/PTZCtrl/channels/1/capabilities",
            auth=auth,
            timeout=5,
            verify=False,
        )
        if response.status_code == 200:
            print("✅ PTZ endpoint accessible")
            if "isSupportContinuous" in response.text:
                if ">true<" in response.text or ">True<" in response.text:
                    print("✅ Continuous movement supported")
                else:
                    print("⚠️  Continuous movement not supported")
        else:
            print(f"⚠️  PTZ capabilities: {response.status_code}")
    except Exception as e:
        print(f"⚠️  PTZ capabilities error: {e}")

    # Ask to proceed
    print("\n[3/4] Ready to test PTZ movement")
    print("⚠️  WARNING: Camera will move if connected properly!")
    print()
    proceed = input("Continue? (yes/no): ").strip().lower()

    if proceed != "yes":
        print("Test cancelled")
        return False

    # Test movement
    print("\n[4/4] Testing PTZ commands...")

    commands = [
        ("Pan RIGHT", "<PTZData><pan>50</pan><tilt>0</tilt><zoom>0</zoom></PTZData>"),
        ("STOP", "<PTZData><pan>0</pan><tilt>0</tilt><zoom>0</zoom></PTZData>"),
        ("Pan LEFT", "<PTZData><pan>-50</pan><tilt>0</tilt><zoom>0</zoom></PTZData>"),
        ("STOP", "<PTZData><pan>0</pan><tilt>0</tilt><zoom>0</zoom></PTZData>"),
        ("Tilt UP", "<PTZData><pan>0</pan><tilt>50</tilt><zoom>0</zoom></PTZData>"),
        ("STOP", "<PTZData><pan>0</pan><tilt>0</tilt><zoom>0</zoom></PTZData>"),
        ("Tilt DOWN", "<PTZData><pan>0</pan><tilt>-50</tilt><zoom>0</zoom></PTZData>"),
        ("STOP", "<PTZData><pan>0</pan><tilt>0</tilt><zoom>0</zoom></PTZData>"),
    ]

    success_count = 0

    for name, xml_data in commands:
        print(f"\n→ {name}...")

        xml_full = f'<?xml version="1.0" encoding="UTF-8"?>{xml_data}'

        try:
            response = requests.put(
                f"{base_url}/ISAPI/PTZCtrl/channels/1/continuous",
                data=xml_full,
                auth=auth,
                headers={"Content-Type": "application/xml"},
                timeout=5,
                verify=False,
            )

            if response.status_code in [200, 204]:
                print(f"   ✅ Command sent (status {response.status_code})")
                success_count += 1

                if "STOP" not in name:
                    time.sleep(2)  # Let it move
            else:
                print(f"   ❌ Failed: {response.status_code}")
                print(f"   Response: {response.text[:200]}")

        except Exception as e:
            print(f"   ❌ Error: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)
    print(f"Commands sent: {success_count}/{len(commands)}")

    if success_count == len(commands):
        print("\n✅ All commands sent successfully!")
        print("\n❓ Did the camera physically move?")
        moved = input("Enter 'yes' if camera moved: ").strip().lower()

        if moved == "yes":
            print("\n🎉 SUCCESS! PTZ is working!")
            print("\nYou can now use:")
            print("  - services/ptz/http_ptz.py (HTTPPTZController)")
            print("  - API server: uvicorn api.server:app")
        else:
            print("\n⚠️  Commands sent but camera didn't move")
            print("\nPossible reasons:")
            print("  1. PTZ disabled in DVR settings")
            print("  2. No external PTZ camera connected to RS-485")
            print("  3. Wrong protocol/baud/address settings")
            print("  4. RS-485 wiring incorrect")
            print("\nNext steps:")
            print("  1. Access web UI: http://{}".format(host))
            print("  2. Go to Configuration → PTZ")
            print("  3. Enable PTZ and configure protocol")
            print("  4. Check RS-485 wiring to external PTZ")
    else:
        print("\n❌ Some commands failed")
        print("\nTry:")
        print("  - Different port: --port 8000 or 443")
        print("  - HTTPS: --https")
        print("  - Different channel: --channel 2")
        print("  - Check credentials")

    print("=" * 60)

    return success_count == len(commands)


def main():
    """Main entry point."""
    import argparse
    import urllib3

    # Disable SSL warnings
    urllib3.disable_warnings()

    parser = argparse.ArgumentParser(
        description="Quick PTZ test for Hikvision DS-MH6171"
    )
    parser.add_argument("host", help="DVR IP address")
    parser.add_argument("--username", default="admin", help="Username (default: admin)")
    parser.add_argument("--password", default="", help="Password")

    args = parser.parse_args()

    try:
        test_ptz(args.host, args.username, args.password)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()
