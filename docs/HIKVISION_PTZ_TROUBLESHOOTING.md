# Hikvision DS-MH6171 PTZ Troubleshooting Guide

## 🚨 IMPORTANT: DS-MH6171 is a Mobile DVR/Encoder

The DS-MH6171 is **NOT a PTZ camera** - it's a mobile DVR/encoder that can **control** external PTZ cameras via:

-   RS-485 (Pelco-D/P, Hikvision protocol)
-   UTC protocol (over coax)
-   Network forwarding

**If PTZ commands succeed but nothing moves, you need to connect an external PTZ camera!**

---

## Quick Diagnosis

Run this diagnostic script:

```bash
python scripts/diagnose_hikvision_ptz.py YOUR_DVR_IP --username admin --password YOUR_PASSWORD
```

This tests:

1. Network connectivity
2. HTTP/HTTPS ports
3. ISAPI endpoints
4. Device information
5. PTZ capabilities
6. Actual PTZ movement

---

## Control Methods

### Method 1: ISAPI (HTTP) - Try First ✅

**Quick Test:**

```bash
# Get device info
curl "http://YOUR_DVR_IP/ISAPI/System/deviceInfo" --digest -u admin:PASSWORD

# Check PTZ capabilities
curl "http://YOUR_DVR_IP/ISAPI/PTZCtrl/channels/1/capabilities" --digest -u admin:PASSWORD

# Send PTZ command (pan right)
curl -X PUT "http://YOUR_DVR_IP/ISAPI/PTZCtrl/channels/1/continuous" \
  --digest -u admin:PASSWORD \
  -H "Content-Type: application/xml" \
  -d '<?xml version="1.0"?><PTZData><pan>50</pan><tilt>0</tilt><zoom>0</zoom></PTZData>'

# Stop movement
curl -X PUT "http://YOUR_DVR_IP/ISAPI/PTZCtrl/channels/1/continuous" \
  --digest -u admin:PASSWORD \
  -H "Content-Type: application/xml" \
  -d '<?xml version="1.0"?><PTZData><pan>0</pan><tilt>0</tilt><zoom>0</zoom></PTZData>'
```

**If 200/204 response:** Commands are reaching the DVR  
**If camera doesn't move:** See "Why Commands Work But Camera Doesn't Move" below

---

### Method 2: ONVIF

```bash
pip install onvif-zeep
python scripts/test_ptz.py
```

---

### Method 3: Pelco-D/P (RS-485) - If Network Control Fails

**You need:**

-   USB-to-RS-485 adapter
-   Physical access to DVR

**Setup:**

1. Connect adapter to DVR's RS-485 terminals (A→A, B→B)
2. Find camera settings in DVR:
    - Protocol: Pelco-D or Pelco-P
    - Baud rate: 2400/4800/9600
    - Address: Usually 1

**Test:**

```bash
python services/ptz/pelco_ptz.py
```

---

## Why Commands Work But Camera Doesn't Move

### Reason 1: PTZ Disabled in DVR

**Solution:** Access web UI (http://YOUR_DVR_IP) and:

1. Go to Configuration → PTZ
2. Enable PTZ for channel 1
3. Set protocol (Pelco-D/P, Hikvision, etc.)
4. Set baud rate (9600 is common)
5. Set address (usually 1)

**No web UI access?** See "Can't Access Web UI" below

---

### Reason 2: No External PTZ Connected

DS-MH6171 is an encoder - it **controls** external PTZ cameras, it's not a PTZ camera itself.

**Check:**

-   Is there a physical PTZ camera connected to the DVR's RS-485 port?
-   Are the RS-485 wires properly connected (A to A, B to B)?
-   Is the PTZ camera powered on?

---

### Reason 3: Wrong Protocol/Settings

PTZ camera and DVR must match:

-   Protocol: Pelco-D, Pelco-P, or Hikvision
-   Baud rate: 2400, 4800, 9600, or 19200
-   Address: 1-255 (usually 1)

**Find these settings:**

-   On PTZ camera: DIP switches or web interface
-   On DVR: Configuration → PTZ

---

### Reason 4: Wrong Channel Number

Try different channels:

```bash
# Channel 1
curl -X PUT "http://YOUR_DVR_IP/ISAPI/PTZCtrl/channels/1/continuous" ...

# Channel 2
curl -X PUT "http://YOUR_DVR_IP/ISAPI/PTZCtrl/channels/2/continuous" ...
```

---

## Can't Access Web UI

### Option 1: Factory Reset (Requires Physical Access)

1. Find reset button on DVR
2. Hold for 30 seconds while powered on
3. Default IP: 192.168.1.64
4. Default login: admin / 12345

### Option 2: SADP Tool (Network Reset)

1. Download SADP from Hikvision website
2. Run SADP - it will find the DVR
3. Select device → Reset password
4. Requires security code (from purchase or support)

### Option 3: Contact Hikvision Support

Provide:

-   Model: DS-MH6171
-   Serial number (on device label)
-   Proof of purchase

---

## Common Error Messages

### ❌ 401 Unauthorized

**Cause:** Wrong username/password  
**Fix:**

-   Try default: admin/12345 or admin/(blank)
-   Use SADP tool to reset password
-   Check if account is locked (wait 30 mins)

### ❌ 404 Not Found

**Cause:** Wrong URL or channel  
**Fix:**

-   Try port 8000 instead of 80
-   Try HTTPS (port 443)
-   Try different channel number (1, 2, 3, etc.)

### ❌ 403 Forbidden

**Cause:** User lacks PTZ permission  
**Fix:**

-   Check user permissions in web UI
-   Use admin account

### ❌ 500 Internal Server Error

**Cause:** DVR firmware bug or misconfiguration  
**Fix:**

-   Reboot DVR
-   Update firmware
-   Factory reset

---

## Wiring Diagram (RS-485)

```
[PTZ Camera]                [DS-MH6171 DVR]
RS-485 Terminals            RS-485 Terminals
    A/+ ────────────────────── A/+
    B/- ────────────────────── B/-

[USB-RS485 Adapter]         [DS-MH6171 DVR]
(for direct testing)
    A+ ─────────────────────── A/+
    B- ─────────────────────── B/-
    USB ───→ Computer
```

**Important:**

-   Use twisted pair cable (Cat5/6 works)
-   Keep cable length under 1200m (4000ft)
-   Add 120Ω termination resistor if cable >100m
-   Don't mix up A and B (camera won't respond)

---

## Diagnostic Checklist

-   [ ] DVR responds to ping
-   [ ] Ports 80/8000 are open (nmap)
-   [ ] Can access `/ISAPI/System/deviceInfo`
-   [ ] PTZ capabilities show `isSupportContinuous=true`
-   [ ] PTZ commands return 200/204
-   [ ] DVR web UI accessible
-   [ ] PTZ is enabled in DVR settings
-   [ ] External PTZ camera is connected (if applicable)
-   [ ] RS-485 wiring is correct (A to A, B to B)
-   [ ] Protocol/baud/address match between DVR and camera

---

## Still Not Working?

**Run this and share the output:**

```bash
python scripts/diagnose_hikvision_ptz.py YOUR_DVR_IP --username admin --password YOUR_PASSWORD > ptz_diagnosis.txt
```

**Also provide:**

1. Is there an external PTZ camera? (Model?)
2. How is it connected? (RS-485, network, coax?)
3. Can you access DVR web UI?
4. What's the exact error or behavior?

---

## Quick Reference Commands

```bash
# Test ISAPI connectivity
curl "http://IP/ISAPI/System/deviceInfo" --digest -u admin:pass

# Pan right (speed 50)
curl -X PUT "http://IP/ISAPI/PTZCtrl/channels/1/continuous" --digest -u admin:pass \
  -H "Content-Type: application/xml" \
  -d '<?xml version="1.0"?><PTZData><pan>50</pan><tilt>0</tilt><zoom>0</zoom></PTZData>'

# Pan left (speed -50)
curl -X PUT "http://IP/ISAPI/PTZCtrl/channels/1/continuous" --digest -u admin:pass \
  -H "Content-Type: application/xml" \
  -d '<?xml version="1.0"?><PTZData><pan>-50</pan><tilt>0</tilt><zoom>0</zoom></PTZData>'

# Tilt up (speed 50)
curl -X PUT "http://IP/ISAPI/PTZCtrl/channels/1/continuous" --digest -u admin:pass \
  -H "Content-Type: application/xml" \
  -d '<?xml version="1.0"?><PTZData><pan>0</pan><tilt>50</tilt><zoom>0</zoom></PTZData>'

# Stop all movement
curl -X PUT "http://IP/ISAPI/PTZCtrl/channels/1/continuous" --digest -u admin:pass \
  -H "Content-Type: application/xml" \
  -d '<?xml version="1.0"?><PTZData><pan>0</pan><tilt>0</tilt><zoom>0</zoom></PTZData>'

# Go to preset 1
curl -X PUT "http://IP/ISAPI/PTZCtrl/channels/1/presets/1/goto" --digest -u admin:pass

# Set preset 1
curl -X PUT "http://IP/ISAPI/PTZCtrl/channels/1/presets/1" --digest -u admin:pass \
  -H "Content-Type: application/xml" \
  -d '<?xml version="1.0"?><PTZPreset><id>1</id><presetName>Position1</presetName></PTZPreset>'
```
