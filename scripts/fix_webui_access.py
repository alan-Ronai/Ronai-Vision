#!/usr/bin/env python3
"""Check and modify user permissions via ISAPI to enable web UI access."""

import requests
from requests.auth import HTTPDigestAuth
import xml.etree.ElementTree as ET


def check_user_permissions(host, username, password):
    """Check current user's permissions."""
    auth = HTTPDigestAuth(username, password)

    print("=" * 70)
    print("CHECKING USER PERMISSIONS")
    print("=" * 70)
    print(f"Host: {host}")
    print(f"User: {username}")
    print()

    # Try to get user info
    endpoints = [
        f"/ISAPI/Security/users/{username}",
        "/ISAPI/Security/users/1",  # Admin is usually ID 1
        "/ISAPI/Security/userCheck",
    ]

    for endpoint in endpoints:
        url = f"http://{host}{endpoint}"
        print(f"Trying: {endpoint}")

        try:
            response = requests.get(url, auth=auth, timeout=5, verify=False)

            if response.status_code == 200:
                print(f"✅ {endpoint} accessible")

                if "<html" in response.text.lower():
                    print("   (Returns HTML, not XML)")
                    continue

                try:
                    root = ET.fromstring(response.text)
                    print("\nUser Information:")
                    print("-" * 70)

                    for elem in root.iter():
                        if elem.text and elem.text.strip() and len(list(elem)) == 0:
                            print(f"  {elem.tag}: {elem.text.strip()}")

                    return True

                except Exception as e:
                    print(f"   Could not parse: {e}")

            else:
                print(f"   Status: {response.status_code}")

        except Exception as e:
            print(f"   Error: {e}")

    print("\n❌ Could not retrieve user permissions")
    return False


def enable_web_ui_access(host, username, password):
    """Try to enable web UI access for user."""
    auth = HTTPDigestAuth(username, password)

    print("\n" + "=" * 70)
    print("ATTEMPTING TO ENABLE WEB UI ACCESS")
    print("=" * 70)

    # Try to modify user to grant web UI access
    xml_configs = [
        # Grant all permissions
        """<?xml version="1.0" encoding="UTF-8"?>
<User>
    <id>1</id>
    <userName>admin</userName>
    <userLevel>Administrator</userLevel>
</User>""",
        # Enable HTTP
        """<?xml version="1.0" encoding="UTF-8"?>
<AdminAccessProtocolList>
    <AdminAccessProtocol>
        <id>1</id>
        <protocol>HTTP</protocol>
        <enabled>true</enabled>
    </AdminAccessProtocol>
</AdminAccessProtocolList>""",
    ]

    endpoints = [
        f"/ISAPI/Security/users/{username}",
        "/ISAPI/Security/adminAccesses",
    ]

    for i, endpoint in enumerate(endpoints):
        url = f"http://{host}{endpoint}"
        print(f"\nTrying: {endpoint}")

        try:
            response = requests.put(
                url,
                data=xml_configs[i],
                auth=auth,
                headers={"Content-Type": "application/xml"},
                timeout=5,
                verify=False,
            )

            if response.status_code in [200, 204]:
                print(f"✅ Successfully updated (status {response.status_code})")
            else:
                print(f"❌ Failed: {response.status_code}")
                if response.text and len(response.text) < 200:
                    print(f"   Response: {response.text}")

        except Exception as e:
            print(f"❌ Error: {e}")


def reset_password_via_isapi(host, username, old_password, new_password):
    """Try to reset password via ISAPI."""
    auth = HTTPDigestAuth(username, old_password)

    print("\n" + "=" * 70)
    print("ATTEMPTING PASSWORD RESET")
    print("=" * 70)

    xml_data = f"""<?xml version="1.0" encoding="UTF-8"?>
<User>
    <id>1</id>
    <userName>{username}</userName>
    <password>{new_password}</password>
</User>"""

    url = f"http://{host}/ISAPI/Security/users/1"

    try:
        response = requests.put(
            url,
            data=xml_data,
            auth=auth,
            headers={"Content-Type": "application/xml"},
            timeout=5,
            verify=False,
        )

        if response.status_code in [200, 204]:
            print(f"✅ Password reset successful!")
            print(f"   New password: {new_password}")
            return True
        else:
            print(f"❌ Failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def unlock_user(host, username, password):
    """Try to unlock user account."""
    auth = HTTPDigestAuth(username, password)

    print("\n" + "=" * 70)
    print("ATTEMPTING TO UNLOCK USER")
    print("=" * 70)

    # Try to get locked users
    url = f"http://{host}/ISAPI/Security/userCheck/{username}"

    try:
        response = requests.get(url, auth=auth, timeout=5, verify=False)

        if response.status_code == 200:
            print("✅ User check endpoint accessible")

            try:
                root = ET.fromstring(response.text)
                locked = root.findtext(".//isLocked", "").lower()

                if locked == "true":
                    print("⚠️  USER IS LOCKED!")
                    print("\nTo unlock, you may need to:")
                    print("  1. Wait 30 minutes (auto-unlock)")
                    print("  2. Factory reset DVR")
                    print("  3. Contact Hikvision support")
                else:
                    print("✅ User is not locked")

            except:
                print("Could not parse lock status")

        else:
            print(f"Status: {response.status_code}")

    except Exception as e:
        print(f"Error: {e}")


def main():
    import argparse
    import urllib3

    urllib3.disable_warnings()

    parser = argparse.ArgumentParser(description="Fix Hikvision user permissions")
    parser.add_argument("host", help="DVR IP")
    parser.add_argument("--username", default="admin", help="Username")
    parser.add_argument("--password", required=True, help="Current password")
    parser.add_argument("--new-password", help="New password (if resetting)")
    parser.add_argument(
        "--enable-webui", action="store_true", help="Try to enable web UI"
    )
    parser.add_argument(
        "--check-only", action="store_true", help="Only check, don't modify"
    )

    args = parser.parse_args()

    # Check current permissions
    check_user_permissions(args.host, args.username, args.password)

    # Check if locked
    unlock_user(args.host, args.username, args.password)

    if not args.check_only:
        if args.enable_webui:
            enable_web_ui_access(args.host, args.username, args.password)

        if args.new_password:
            reset_password_via_isapi(
                args.host, args.username, args.password, args.new_password
            )

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("\n1. Try accessing web UI again:")
    print(f"   http://{args.host}")
    print(f"   Username: {args.username}")
    print(f"   Password: {args.password}")

    print("\n2. If still locked out, try:")
    print("   - Wait 30 minutes for auto-unlock")
    print("   - Use SADP tool to reset password")
    print("   - Factory reset (requires physical access)")

    print("\n3. If web UI still won't load:")
    print("   - Check browser (try Chrome/Firefox)")
    print("   - Clear browser cache")
    print("   - Try incognito/private mode")
    print("   - Check if web UI port is different")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
