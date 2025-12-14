"""SRTP (Secure RTP) Detection and Handling

SRTP encrypts RTP payload and adds authentication tag.
If the sender is using SRTP, you need:
1. Crypto suite (AES-128, AES-256, etc.)
2. Master key
3. Master salt
4. Key derivation parameters

This is typically negotiated via SDP in RTSP or SIP.
"""

import struct


def detect_srtp(rtp_data: bytes) -> dict:
    """Detect if RTP packet might be SRTP.

    Returns:
        dict with detection results
    """
    if len(rtp_data) < 12:
        return {"is_srtp": False, "reason": "Too short"}

    # Parse RTP header
    byte0, byte1 = rtp_data[0], rtp_data[1]
    version = (byte0 >> 6) & 0x03
    payload_type = byte1 & 0x7F

    results = {
        "is_srtp": False,
        "confidence": "unknown",
        "indicators": [],
        "recommendations": [],
    }

    # Check version
    if version != 2:
        results["is_srtp"] = True
        results["confidence"] = "high"
        results["indicators"].append(f"Invalid RTP version: {version}")
        results["recommendations"].append("Packet may be encrypted")
        return results

    # SRTP indicators:
    # 1. Payload looks random (high entropy)
    # 2. Authentication tag at end (4-16 bytes)
    # 3. Unusual payload types

    header_len = 12
    csrc_count = byte0 & 0x0F
    header_len += csrc_count * 4

    # Check for extension
    extension = (byte0 >> 4) & 0x01
    if extension and len(rtp_data) >= header_len + 4:
        ext_header = struct.unpack("!HH", rtp_data[header_len : header_len + 4])
        ext_len = ext_header[1] * 4
        header_len += 4 + ext_len

    if len(rtp_data) <= header_len:
        results["indicators"].append("No payload")
        return results

    payload = rtp_data[header_len:]

    # Check entropy (encrypted data should be high entropy)
    entropy = calculate_entropy(payload[: min(100, len(payload))])

    if entropy > 7.5:  # Very high entropy (max is 8.0)
        results["indicators"].append(f"High entropy: {entropy:.2f}/8.0 (encrypted?)")
        results["confidence"] = "medium"

    # Check for common unencrypted patterns
    if payload[:4] == b"\x00\x00\x00\x00":
        results["indicators"].append("Null bytes detected (unlikely SRTP)")

    # Check payload size (SRTP adds 4-16 bytes for auth tag)
    if len(payload) % 20 == 10 or len(payload) % 20 == 14:
        results["indicators"].append("Payload size suggests auth tag present")
        results["confidence"] = "medium"

    # Recommendations
    if results["confidence"] in ["medium", "high"]:
        results["is_srtp"] = True
        results["recommendations"].extend(
            [
                "Contact sender to confirm if SRTP is used",
                "Request crypto parameters: key, salt, suite",
                "Check SDP/RTSP session description",
                "Consider using libsrtp library for decryption",
            ]
        )
    else:
        results["recommendations"].append("Likely plain RTP (not encrypted)")

    return results


def calculate_entropy(data: bytes) -> float:
    """Calculate Shannon entropy of byte data."""
    if not data:
        return 0.0

    # Count byte frequencies
    freq = [0] * 256
    for byte in data:
        freq[byte] += 1

    # Calculate entropy
    import math

    entropy = 0.0
    data_len = len(data)

    for count in freq:
        if count > 0:
            probability = count / data_len
            entropy -= probability * math.log2(probability)

    return entropy


def print_srtp_guide():
    """Print guide for handling SRTP."""
    print("""
========================================
SRTP (Secure RTP) Setup Guide
========================================

If sender is using SRTP, you need:

1. Crypto Suite (algorithm)
   - AES_CM_128_HMAC_SHA1_80 (most common)
   - AES_CM_128_HMAC_SHA1_32
   - AES_256_CM_HMAC_SHA1_80

2. Master Key (hex string, 128 or 256 bits)
   Example: 2DE21B8C8D6C8B4E2A8F3C9D7E6F5A4B

3. Master Salt (hex string, 112 bits)
   Example: 3A8D7C6B5A4E3D2C1B

4. Key Derivation Rate (usually 0)

These are exchanged via:
- RTSP: a=crypto line in SDP
- SIP: SDES or DTLS-SRTP
- Manual configuration

Python libraries:
- pylibsrtp (wrapper for libsrtp)
- PySRTP (pure Python, slower)

Example SDP crypto line:
a=crypto:1 AES_CM_128_HMAC_SHA1_80 inline:2DE21B8C8D6C8B4E2A8F3C9D7E6F5A4B3A8D7C6B5A4E3D2C1B

Installation:
pip install pylibsrtp

========================================
""")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print_srtp_guide()
        print("Usage: python srtp_detector.py <packet_hex>")
        print("Example: python srtp_detector.py 80006b3a00000a1c12345678...")
        sys.exit(0)

    # Parse hex string
    hex_string = sys.argv[1].replace(" ", "").replace(":", "")
    try:
        packet_data = bytes.fromhex(hex_string)
    except ValueError as e:
        print(f"Error: Invalid hex string: {e}")
        sys.exit(1)

    print(f"Analyzing {len(packet_data)} byte packet...")
    print()

    results = detect_srtp(packet_data)

    print("SRTP Detection Results:")
    print("=" * 40)
    print(f"Is SRTP: {results['is_srtp']}")
    print(f"Confidence: {results['confidence']}")
    print()

    if results["indicators"]:
        print("Indicators:")
        for indicator in results["indicators"]:
            print(f"  • {indicator}")
        print()

    if results["recommendations"]:
        print("Recommendations:")
        for rec in results["recommendations"]:
            print(f"  → {rec}")
