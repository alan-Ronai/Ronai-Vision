#!/usr/bin/env python3
"""Check and configure PTZ settings via ISAPI (no web UI needed).

This script checks PTZ configuration and can enable/configure it.
"""

import sys
import requests
from requests.auth import HTTPDigestAuth
import xml.etree.ElementTree as ET


def get_ptz_config(host, username, password, channel=1):
    """Get PTZ configuration for a channel."""
    auth = HTTPDigestAuth(username, password)

    print(f"\n[Channel {channel}] Checking PTZ Configuration...")
    print("-" * 60)

    # Try to get PTZ configuration
    endpoints = [
        f"/ISAPI/PTZCtrl/channels/{channel}/status",
        f"/ISAPI/PTZCtrl/channels/{channel}/capabilities",
        f"/ISAPI/System/Video/inputs/channels/{channel}/PTZ",
        f"/ISAPI/System/Video/inputs/channels/{channel}",
    ]

    for endpoint in endpoints:
        url = f"http://{host}{endpoint}"
        try:
            response = requests.get(url, auth=auth, timeout=5, verify=False)

            if response.status_code == 200:
                print(f"\n✅ {endpoint}")
                print("-" * 60)

                # Check if response is actually XML (not HTML redirect)
                if (
                    not response.text.strip().startswith("<?xml")
                    and "<html" in response.text.lower()
                ):
                    print(
                        "⚠️  Received HTML instead of XML (endpoint might not exist or requires different auth)"
                    )
                    continue

                try:
                    # Pretty print XML
                    root = ET.fromstring(response.text)
                    print_xml_tree(root, indent=0)

                    # Check for enabled status
                    enabled = root.findtext(".//enabled", "").lower()
                    if enabled == "false":
                        print("\n⚠️  PTZ is DISABLED in configuration!")
                        return False, endpoint
                    elif enabled == "true":
                        print("\n✅ PTZ is ENABLED")

                    # Check protocol
                    protocol = root.findtext(".//PTZProtocol", "") or root.findtext(
                        ".//protocolType", ""
                    )
                    if protocol:
                        print(f"📡 Protocol: {protocol}")

                    # Check address
                    address = root.findtext(".//address", "") or root.findtext(
                        ".//PTZAddress", ""
                    )
                    if address:
                        print(f"📍 Address: {address}")

                    # Check baud rate
                    baud = root.findtext(".//baudRate", "") or root.findtext(
                        ".//PTZBaudRate", ""
                    )
                    if baud:
                        print(f"⚡ Baud Rate: {baud}")

                except Exception as e:
                    print(f"Could not parse XML: {e}")
                    print("\nRaw response:")
                    print(response.text[:500])

                return True, endpoint

            elif response.status_code == 404:
                continue
            else:
                print(f"❌ {endpoint}: Status {response.status_code}")

        except Exception as e:
            print(f"⚠️  {endpoint}: {e}")
            continue

    print("\n❌ Could not retrieve PTZ configuration")
    return False, None


def print_xml_tree(element, indent=0):
    """Pretty print XML tree."""
    prefix = "  " * indent

    if element.text and element.text.strip():
        print(f"{prefix}{element.tag}: {element.text.strip()}")
    elif len(element) == 0:
        print(f"{prefix}{element.tag}")
    else:
        print(f"{prefix}{element.tag}:")

    for child in element:
        print_xml_tree(child, indent + 1)


def enable_ptz(
    host, username, password, channel=1, protocol="PELCOD", address=1, baudrate=9600
):
    """Try to enable PTZ via ISAPI."""
    auth = HTTPDigestAuth(username, password)

    print(f"\n[Channel {channel}] Attempting to Enable PTZ...")
    print("-" * 60)
    print(f"Protocol: {protocol}")
    print(f"Address: {address}")
    print(f"Baud Rate: {baudrate}")
    print()

    # Try to enable PTZ
    xml_config = f"""<?xml version="1.0" encoding="UTF-8"?>
<PTZ>
    <enabled>true</enabled>
    <PTZProtocol>{protocol}</PTZProtocol>
    <address>{address}</address>
    <baudRate>{baudrate}</baudRate>
</PTZ>"""

    endpoint = f"/ISAPI/System/Video/inputs/channels/{channel}/PTZ"
    url = f"http://{host}{endpoint}"

    try:
        response = requests.put(
            url, data=xml_config, auth=auth, timeout=5, verify=False
        )

        if response.status_code in [200, 204]:
            print(f"✅ PTZ configuration updated!")
            print(f"   Status: {response.status_code}")
            return True
        else:
            print(f"❌ Failed: Status {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def check_all_channels(host, username, password, max_channels=4):
    """Check PTZ configuration on all channels."""
    print("\n" + "=" * 60)
    print("CHECKING ALL CHANNELS")
    print("=" * 60)

    working_channels = []

    for channel in range(1, max_channels + 1):
        found, endpoint = get_ptz_config(host, username, password, channel)
        if found:
            working_channels.append(channel)
        print()

    if working_channels:
        print(f"✅ Found PTZ configuration on channels: {working_channels}")
    else:
        print("❌ No PTZ configuration found on any channel")

    return working_channels


def main():
    """Main entry point."""
    import argparse
    import urllib3

    urllib3.disable_warnings()

    parser = argparse.ArgumentParser(
        description="Check and configure Hikvision PTZ via ISAPI"
    )
    parser.add_argument("host", help="DVR IP address")
    parser.add_argument("--username", default="admin", help="Username")
    parser.add_argument("--password", default="", help="Password")
    parser.add_argument(
        "--channel", type=int, default=1, help="Channel number (default: 1)"
    )
    parser.add_argument("--enable", action="store_true", help="Try to enable PTZ")
    parser.add_argument(
        "--protocol", default="PELCOD", help="PTZ protocol (default: PELCOD)"
    )
    parser.add_argument(
        "--address", type=int, default=1, help="PTZ address (default: 1)"
    )
    parser.add_argument(
        "--baudrate", type=int, default=9600, help="Baud rate (default: 9600)"
    )
    parser.add_argument("--scan-all", action="store_true", help="Scan all channels")

    args = parser.parse_args()

    print("=" * 60)
    print("HIKVISION PTZ CONFIGURATION CHECK")
    print("=" * 60)
    print(f"Host: {args.host}")
    print(f"User: {args.username}")
    print("=" * 60)

    if args.scan_all:
        check_all_channels(args.host, args.username, args.password)
    else:
        found, endpoint = get_ptz_config(
            args.host, args.username, args.password, args.channel
        )

        if not found and args.enable:
            print("\n" + "=" * 60)
            input("Press Enter to try enabling PTZ...")
            enable_ptz(
                args.host,
                args.username,
                args.password,
                args.channel,
                args.protocol,
                args.address,
                args.baudrate,
            )

            # Check again
            print("\n" + "=" * 60)
            print("Verifying configuration...")
            get_ptz_config(args.host, args.username, args.password, args.channel)

        elif not found:
            print("\n" + "=" * 60)
            print("PTZ configuration not found or disabled")
            print("\nTo enable PTZ, run:")
            print(f"  python {sys.argv[0]} {args.host} --enable")
            print("\nCommon protocols:")
            print("  PELCOD (Pelco-D)")
            print("  PELCOP (Pelco-P)")
            print("  HIKVISION")
            print("  SAMSUNG")
            print("  PANASONIC")
            print("\nExample:")
            print(
                f"  python {sys.argv[0]} {args.host} --enable --protocol HIKVISION --address 1 --baudrate 9600"
            )

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
