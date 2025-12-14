# RTP Troubleshooting Guide

## Quick Diagnosis Steps

### 1. Run Debug Receiver (Most Important First!)
```bash
python scripts/debug_rtp_receiver.py --port 5004
```
This will show you:
- If ANY packets are arriving
- The RTP header details
- Payload type and codec
- Any parsing errors

### 2. Capture Raw Packets
```bash
./scripts/capture_rtp_packets.sh 5004 30
```
Verifies packets reach the network interface (bypasses application).

### 3. Check AWS Networking
```bash
./scripts/check_aws_network.sh
```
Checks ports, firewall, routing, instance metadata.

---

## Problem: No Packets Received

### Check 1: Verify Destination IP
```bash
# Get your public IP
curl ifconfig.me

# Ask sender: "Are you sending to this IP?"
```

### Check 2: AWS Security Group
```
EC2 Console → Security Groups → Inbound Rules
Must have: UDP port 5004 from sender's IP (or 0.0.0.0/0)
```

### Check 3: AWS Network ACL (Often Forgotten!)
```
VPC Console → Network ACLs → Inbound/Outbound Rules
Must allow: UDP port 5004

Note: NACLs are STATELESS - need both inbound AND outbound rules!
```

### Check 4: Firewall on EC2 Instance
```bash
# Check iptables
sudo iptables -L -n | grep 5004

# Check firewalld (CentOS/RHEL)
sudo firewall-cmd --list-all

# Temporarily disable to test
sudo systemctl stop firewalld
```

### Check 5: Application Binding
```bash
# Check if app is listening
sudo netstat -ulnp | grep 5004
sudo lsof -i :5004

# Should show Python listening on 0.0.0.0:5004
```

---

## Problem: Dynamic Payload Type (96-127)

Military radios often use dynamic payload types. Your receiver will now:
1. Accept PT 96-127
2. Assume default codec (G.711 μ-law)
3. Log a warning

**To fix properly:**
1. Ask sender: "What codec are you using?"
2. Update decoder in [raw_rtp_receiver.py](services/audio/raw_rtp_receiver.py):
```python
codec_map = {
    0: ("g711_ulaw", "G.711 μ-law"),
    8: ("g711_alaw", "G.711 A-law"),
    96: ("actual_codec_here", "Their Codec"),  # Add this
}
```

Common military codecs:
- **MELPe** (2400/1200 bps) - Very low bandwidth
- **CVSD** - Bluetooth quality
- **Opus** - Modern, high quality
- **G.729** - 8 kbps compressed
- **G.711** - 64 kbps uncompressed (most common)

---

## Problem: Custom RTP Extensions

The parser already handles extensions. Debug receiver will show:
```
[EXT] Profile=0xABCD Length=4 words (16 bytes)
```

Extensions don't affect payload decoding - they're metadata.

---

## Problem: SRTP (Encrypted)

### Detection
```bash
# Run debug receiver, look for:
# - Very high entropy payload
# - Random-looking data
# - Unusual payload size

# Or test a packet directly
python scripts/srtp_detector.py "800063B400000A1C12345678..."
```

### Solution
SRTP requires crypto keys from sender. You need:
1. **Master Key** (hex, 128/256 bits)
2. **Master Salt** (hex, 112 bits)
3. **Crypto Suite** (usually AES_CM_128_HMAC_SHA1_80)

These are exchanged via RTSP SDP:
```
a=crypto:1 AES_CM_128_HMAC_SHA1_80 inline:KEY+SALT_base64
```

**To implement:**
```bash
pip install pylibsrtp
```

Then decrypt before passing to decoder (requires code changes).

---

## Problem: Wrong Port

### Verify Port
```bash
# Ask sender which port they're using
# Common RTP ports:
# - 5004 (raw RTP)
# - 5060 (SIP signaling)
# - 16384-32767 (dynamic range)

# Change debug receiver port:
python scripts/debug_rtp_receiver.py --port 16384
```

---

## Problem: NAT Traversal

If sender is behind NAT:
1. They need to configure **port forwarding**
2. Or use **STUN/TURN** server
3. Or establish connection in reverse (you send first, they reply)

---

## Problem: Packet Loss

Debug receiver shows:
```
[RTP] Received 50 packets (dropped: 15, reordered: 3)
```

### Causes:
1. Network congestion
2. Buffer too small
3. Processing too slow

### Solutions:
```python
# Increase jitter buffer
receiver = RawRTPReceiver(port=5004, jitter_buffer_ms=200)  # Default 100ms

# Or increase socket buffer
socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2*1024*1024)  # 2MB
```

---

## Testing Checklist

- [ ] Debug receiver shows packets arriving
- [ ] Security Group allows UDP 5004 inbound
- [ ] Network ACL allows UDP 5004 (both directions!)
- [ ] No firewall blocking on instance
- [ ] Sender confirmed destination IP
- [ ] Sender confirmed destination port
- [ ] Payload type identified (0-127)
- [ ] Codec confirmed (if dynamic PT)
- [ ] Not SRTP (or have crypto keys)
- [ ] tcpdump shows packets arriving

---

## Get Help

When asking for help, provide:
1. Output from `python scripts/debug_rtp_receiver.py`
2. Output from `./scripts/capture_rtp_packets.sh`
3. Sender's configuration (IP, port, codec)
4. First packet hex dump (from debug receiver)

Example:
```
[PACKET #1] From 203.0.113.5:45678 | Size: 172 bytes
[RAW HEX] 800000010000012c1234567890abcdef...
[RTP] Version=2 PT=96 Marker=0
[RTP] Seq=1 Timestamp=300 SSRC=0x12345678
[CODEC] Payload Type 96 = Dynamic (PT 96)
```

This tells us everything we need!
