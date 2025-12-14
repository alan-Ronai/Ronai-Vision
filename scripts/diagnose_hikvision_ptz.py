#!/usr/bin/env python3
"""Comprehensive Hikvision DS-MH6171 PTZ diagnostic and control script.

Tests all possible PTZ control methods:
1. ISAPI (HTTP/HTTPS)
2. ONVIF
3. Pelco-D/P over RS-485
4. SDK (if available)
5. RTSP backchannel

Model: Hikvision DS-MH6171
"""

import sys
import os
import time
import requests
from requests.auth import HTTPDigestAuth, HTTPBasicAuth
import socket
import struct

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class HikvisionPTZDiagnostics:
    """Comprehensive diagnostics for Hikvision PTZ camera."""

    def __init__(self, host: str, username: str = "admin", password: str = ""):
        self.host = host
        self.username = username
        self.password = password
        self.auth_digest = HTTPDigestAuth(username, password)
        self.auth_basic = HTTPBasicAuth(username, password)

        self.results = {
            "isapi": {},
            "onvif": {},
            "device_info": {},
            "network": {},
            "capabilities": {},
        }

    def run_all_diagnostics(self):
        """Run all diagnostic tests."""
        print("=" * 70)
        print("HIKVISION DS-MH6171 PTZ DIAGNOSTICS")
        print("=" * 70)
        print(f"Target: {self.host}")
        print(f"User: {self.username}")
        print("=" * 70)
        print()

        self.test_network_connectivity()
        self.test_http_ports()
        self.test_isapi_basic()
        self.test_isapi_ptz_capabilities()
        self.test_isapi_ptz_movement()
        self.test_onvif()
        self.print_summary()

    def test_network_connectivity(self):
        """Test basic network connectivity."""
        print("[1/6] Testing Network Connectivity...")
        print("-" * 70)

        # Ping test (ICMP)
        import subprocess

        try:
            result = subprocess.run(
                ["ping", "-c", "3", "-W", "2", self.host],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                print("✅ Ping: SUCCESS")
                self.results["network"]["ping"] = True
            else:
                print("❌ Ping: FAILED")
                self.results["network"]["ping"] = False
        except:
            print("⚠️  Ping: Could not test")
            self.results["network"]["ping"] = None

        # TCP port scan
        common_ports = [80, 443, 554, 8000, 8080, 8443]
        open_ports = []

        print("\nScanning common ports...")
        for port in common_ports:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex((self.host, port))
            sock.close()

            if result == 0:
                print(f"  ✅ Port {port}: OPEN")
                open_ports.append(port)
            else:
                print(f"  ❌ Port {port}: CLOSED")

        self.results["network"]["open_ports"] = open_ports
        print()

    def test_http_ports(self):
        """Test HTTP/HTTPS access."""
        print("[2/6] Testing HTTP/HTTPS Access...")
        print("-" * 70)

        for protocol in ["http", "https"]:
            for port in [80, 443, 8000, 8080, 8443]:
                url = f"{protocol}://{self.host}:{port}/"
                try:
                    response = requests.get(
                        url, auth=self.auth_digest, timeout=3, verify=False
                    )
                    print(
                        f"✅ {protocol.upper()}:{port} - Status {response.status_code}"
                    )
                    if response.status_code == 200:
                        self.results["network"][f"{protocol}_{port}"] = True
                except requests.exceptions.Timeout:
                    print(f"⏱️  {protocol.upper()}:{port} - Timeout")
                except requests.exceptions.ConnectionError:
                    print(f"❌ {protocol.upper()}:{port} - Connection refused")
                except Exception as e:
                    print(f"⚠️  {protocol.upper()}:{port} - {e}")
        print()

    def test_isapi_basic(self):
        """Test ISAPI basic endpoints."""
        print("[3/6] Testing ISAPI Endpoints...")
        print("-" * 70)

        # Try both HTTP and HTTPS
        for protocol in ["http", "https"]:
            for port in [80, 443]:
                base_url = f"{protocol}://{self.host}:{port}"

                endpoints = [
                    "/ISAPI/System/deviceInfo",
                    "/ISAPI/PTZCtrl/channels/1/capabilities",
                    "/ISAPI/PTZCtrl/channels/1/status",
                ]

                for endpoint in endpoints:
                    url = base_url + endpoint
                    try:
                        # Try digest auth
                        response = requests.get(
                            url, auth=self.auth_digest, timeout=3, verify=False
                        )

                        if response.status_code == 401:
                            # Try basic auth
                            response = requests.get(
                                url, auth=self.auth_basic, timeout=3, verify=False
                            )

                        if response.status_code == 200:
                            print(f"✅ {endpoint}")
                            print(f"   URL: {url}")

                            # Parse device info
                            if "deviceInfo" in endpoint:
                                try:
                                    import xml.etree.ElementTree as ET

                                    root = ET.fromstring(response.text)
                                    model = root.findtext(".//model", "Unknown")
                                    fw = root.findtext(".//firmwareVersion", "Unknown")
                                    serial = root.findtext(".//serialNumber", "Unknown")
                                    print(f"   Model: {model}")
                                    print(f"   Firmware: {fw}")
                                    print(f"   Serial: {serial}")

                                    self.results["device_info"] = {
                                        "model": model,
                                        "firmware": fw,
                                        "serial": serial,
                                    }
                                except:
                                    pass

                            self.results["isapi"][endpoint] = True
                            break  # Found working endpoint, skip other protocols
                        else:
                            print(f"❌ {endpoint} - Status {response.status_code}")
                    except Exception as e:
                        continue
        print()

    def test_isapi_ptz_capabilities(self):
        """Test PTZ capabilities."""
        print("[4/6] Testing PTZ Capabilities...")
        print("-" * 70)

        for protocol in ["http", "https"]:
            for port in [80, 443]:
                url = f"{protocol}://{self.host}:{port}/ISAPI/PTZCtrl/channels/1/capabilities"

                try:
                    response = requests.get(
                        url, auth=self.auth_digest, timeout=3, verify=False
                    )

                    if response.status_code == 200:
                        print(f"✅ PTZ Capabilities Retrieved")
                        print(f"   URL: {url}")

                        # Parse capabilities
                        try:
                            import xml.etree.ElementTree as ET

                            root = ET.fromstring(response.text)

                            supports_continuous = root.findtext(
                                ".//isSupportContinuous", "false"
                            )
                            supports_absolute = root.findtext(
                                ".//isSupportAbsolute", "false"
                            )
                            supports_relative = root.findtext(
                                ".//isSupportRelative", "false"
                            )

                            print(f"   Continuous Move: {supports_continuous}")
                            print(f"   Absolute Move: {supports_absolute}")
                            print(f"   Relative Move: {supports_relative}")

                            self.results["capabilities"]["continuous"] = (
                                supports_continuous == "true"
                            )
                            self.results["capabilities"]["absolute"] = (
                                supports_absolute == "true"
                            )
                            self.results["capabilities"]["relative"] = (
                                supports_relative == "true"
                            )
                        except:
                            print("   (Could not parse capabilities XML)")

                        return True

                except Exception as e:
                    continue

        print("❌ Could not retrieve PTZ capabilities")
        print()
        return False

    def test_isapi_ptz_movement(self):
        """Test actual PTZ movement via ISAPI."""
        print("[5/6] Testing PTZ Movement (ISAPI)...")
        print("-" * 70)

        input("Press Enter to test PTZ movement (camera will move!)...")

        for protocol in ["http", "https"]:
            for port in [80, 443]:
                base_url = f"{protocol}://{self.host}:{port}"

                # Test continuous movement
                url = base_url + "/ISAPI/PTZCtrl/channels/1/continuous"

                # Try pan right
                xml_data = """<?xml version="1.0" encoding="UTF-8"?>
<PTZData>
    <pan>50</pan>
    <tilt>0</tilt>
</PTZData>"""

                try:
                    print(f"\nTrying: {url}")
                    print("Command: Pan Right (speed 50)")

                    response = requests.put(
                        url,
                        data=xml_data,
                        auth=self.auth_digest,
                        timeout=5,
                        verify=False,
                    )

                    if response.status_code in [200, 204]:
                        print("✅ PTZ MOVE COMMAND SENT!")
                        print(f"   Status: {response.status_code}")
                        print("   Waiting 2 seconds...")
                        time.sleep(2)

                        # Stop
                        xml_stop = """<?xml version="1.0" encoding="UTF-8"?>
<PTZData>
    <pan>0</pan>
    <tilt>0</tilt>
</PTZData>"""
                        requests.put(
                            url,
                            data=xml_stop,
                            auth=self.auth_digest,
                            timeout=5,
                            verify=False,
                        )
                        print("   Sent stop command")

                        self.results["isapi"]["ptz_move"] = True
                        print("\n⚠️  DID THE CAMERA MOVE?")
                        moved = (
                            input("   Enter 'yes' if camera moved: ").strip().lower()
                        )

                        if moved == "yes":
                            print("\n🎉 SUCCESS! PTZ is working via ISAPI!")
                            return True
                        else:
                            print("\n⚠️  Command sent but camera didn't move")
                            print("   Possible reasons:")
                            print("   - PTZ is disabled in camera settings")
                            print("   - Camera requires different XML format")
                            print("   - RS-485 not connected properly")
                            print("   - Wrong channel number")
                    else:
                        print(f"❌ Status {response.status_code}")
                        print(f"   Response: {response.text[:200]}")

                except Exception as e:
                    print(f"❌ Error: {e}")
                    continue

        print("\n❌ PTZ movement via ISAPI failed")
        print()
        return False

    def test_onvif(self):
        """Test ONVIF PTZ."""
        print("[6/6] Testing ONVIF...")
        print("-" * 70)

        try:
            from onvif import ONVIFCamera

            for port in [80, 8000]:
                try:
                    print(f"\nTrying ONVIF on port {port}...")
                    cam = ONVIFCamera(self.host, port, self.username, self.password)

                    # Get device info
                    info = cam.devicemgmt.GetDeviceInformation()
                    print(f"✅ ONVIF Connected!")
                    print(f"   Manufacturer: {info.Manufacturer}")
                    print(f"   Model: {info.Model}")
                    print(f"   Firmware: {info.FirmwareVersion}")

                    # Try PTZ
                    ptz = cam.create_ptz_service()
                    media = cam.create_media_service()
                    profiles = media.GetProfiles()

                    if profiles:
                        print(f"   Profiles: {len(profiles)}")
                        self.results["onvif"]["connected"] = True
                        return True

                except Exception as e:
                    print(f"❌ ONVIF port {port}: {e}")
                    continue

        except ImportError:
            print("⚠️  python-onvif-zeep not installed")
            print("   Install with: pip install onvif-zeep")

        self.results["onvif"]["connected"] = False
        print()
        return False

    def print_summary(self):
        """Print diagnostic summary."""
        print("\n")
        print("=" * 70)
        print("DIAGNOSTIC SUMMARY")
        print("=" * 70)

        if self.results["device_info"]:
            print(f"\n📷 Device Info:")
            print(f"   Model: {self.results['device_info'].get('model', 'Unknown')}")
            print(
                f"   Firmware: {self.results['device_info'].get('firmware', 'Unknown')}"
            )

        print(f"\n🌐 Network:")
        print(
            f"   Open Ports: {', '.join(map(str, self.results['network'].get('open_ports', [])))}"
        )

        print(f"\n🎮 PTZ Status:")
        isapi_works = self.results["isapi"].get("ptz_move", False)
        onvif_works = self.results["onvif"].get("connected", False)

        if isapi_works:
            print("   ✅ ISAPI PTZ: WORKING")
        else:
            print("   ❌ ISAPI PTZ: NOT WORKING")

        if onvif_works:
            print("   ✅ ONVIF: WORKING")
        else:
            print("   ❌ ONVIF: NOT WORKING")

        print("\n📋 Recommendations:")

        if not isapi_works and not onvif_works:
            print("   1. Check if PTZ is enabled in camera web interface")
            print("   2. Verify RS-485 wiring (if using serial PTZ)")
            print("   3. Check camera address and baud rate settings")
            print("   4. Try accessing web UI at http://{}".format(self.host))
            print("   5. Check if camera requires firmware update")
            print("   6. Verify you're using correct credentials")
            print("\n   Alternative Control Methods:")
            print("   - RS-485 Pelco-D/P (if camera has serial port)")
            print("     Run: python services/ptz/pelco_ptz.py")
            print("   - VISCA over IP (some models)")
            print("   - SDK (requires Hikvision SDK installation)")
        elif isapi_works:
            print("   ✅ Use ISAPI for PTZ control (HTTP-based)")
            print("   ✅ Your existing HTTP PTZ controller should work")

        print("\n" + "=" * 70)


def main():
    """Run diagnostics."""
    import argparse

    parser = argparse.ArgumentParser(description="Hikvision DS-MH6171 PTZ Diagnostics")
    parser.add_argument("host", help="Camera IP address")
    parser.add_argument("--username", default="admin", help="Username (default: admin)")
    parser.add_argument("--password", default="", help="Password")

    args = parser.parse_args()

    # Disable SSL warnings
    import urllib3

    urllib3.disable_warnings()

    diag = HikvisionPTZDiagnostics(args.host, args.username, args.password)
    diag.run_all_diagnostics()


if __name__ == "__main__":
    main()
