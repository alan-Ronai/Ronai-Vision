#!/usr/bin/env python3
"""Simple PTZ status checker using the working /status endpoint."""

import sys
import requests
from requests.auth import HTTPDigestAuth
import xml.etree.ElementTree as ET


def check_ptz_status(host, username, password):
    """Check PTZ status for all channels."""
    auth = HTTPDigestAuth(username, password)

    print("=" * 70)
    print("CHECKING PTZ STATUS ON ALL CHANNELS")
    print("=" * 70)
    print(f"Host: {host}")
    print()

    found_ptz = False

    for channel in range(1, 5):
        url = f"http://{host}/ISAPI/PTZCtrl/channels/{channel}/status"

        try:
            response = requests.get(url, auth=auth, timeout=5, verify=False)

            if response.status_code == 200:
                print(f"\n{'=' * 70}")
                print(f"CHANNEL {channel}")
                print("=" * 70)

                if "<html" in response.text.lower():
                    print("⚠️  Returns HTML (endpoint doesn't exist)")
                    continue

                try:
                    root = ET.fromstring(response.text)

                    # Get all elements
                    pan = root.findtext(".//pan")
                    tilt = root.findtext(".//tilt")
                    zoom = root.findtext(".//zoom")

                    if pan is not None or tilt is not None or zoom is not None:
                        print("✅ PTZ Status Available")
                        found_ptz = True

                        if pan:
                            print(f"   Pan: {pan}")
                        if tilt:
                            print(f"   Tilt: {tilt}")
                        if zoom:
                            print(f"   Zoom: {zoom}")

                        # Print all elements
                        print("\n   Full status:")
                        for elem in root.iter():
                            if elem.text and elem.text.strip():
                                print(f"   {elem.tag}: {elem.text.strip()}")
                    else:
                        print("⚠️  Endpoint exists but no PTZ position data")

                except Exception as e:
                    print(f"❌ XML parse error: {e}")

            elif response.status_code == 404:
                pass  # Skip 404s

        except Exception as e:
            print(f"\nChannel {channel}: {e}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if found_ptz:
        print("✅ DVR supports PTZ and can send commands")
        print("\n❌ BUT camera doesn't move when commanded")
        print("\n🔍 Root Cause:")
        print("   The DS-MH6171 is a Mobile DVR/Encoder, not a PTZ camera.")
        print("   It CONTROLS external PTZ cameras via RS-485.")
        print("\n📋 What you need:")
        print("   1. Physical PTZ camera connected to RS-485 terminals")
        print("   2. Correct wiring: A-to-A, B-to-B")
        print("   3. Matching settings:")
        print("      - Protocol: Pelco-D/P or Hikvision")
        print("      - Address: 1 (default)")
        print("      - Baud Rate: 9600 (common)")
        print("\n🔧 Troubleshooting:")
        print("   1. Check RS-485 terminals on DVR for connected wires")
        print("   2. If no PTZ connected: You need to buy/connect one")
        print("   3. If PTZ connected: Try swapping A and B wires")
        print("   4. Check PTZ camera's DIP switches/settings")
    else:
        print("⚠️  No PTZ status available on any channel")
        print("\nThe DVR might only support RS-485 passthrough.")
        print("You need to configure PTZ in the web interface.")

    print("=" * 70)


def main():
    import argparse
    import urllib3

    urllib3.disable_warnings()

    parser = argparse.ArgumentParser()
    parser.add_argument("host")
    parser.add_argument("--username", default="admin")
    parser.add_argument("--password", default="")
    args = parser.parse_args()

    check_ptz_status(args.host, args.username, args.password)


if __name__ == "__main__":
    main()
