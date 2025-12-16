# RTP TCP Relay - No Router Configuration Required!

This system allows you to receive RTP audio from your EC2 instance on your local machine **without configuring your router or port forwarding**.

## How It Works

1. **Radio Device** → sends RTP packets via UDP to EC2 (port 5004)
2. **EC2 Server** (`rtp_tcp_server.py`) → receives RTP via UDP, relays to TCP clients
3. **Your Local Machine** (`rtp_tcp_client.py`) → connects to EC2 via TCP, receives RTP packets, saves audio

**Why this works:** Your local machine initiates the connection to EC2 (outbound), which works through NAT/firewalls without any router configuration!

## Setup Instructions

### Step 1: On Your EC2 Instance

Make sure TCP port 5005 is open in your EC2 security group:
- Go to EC2 Console → Security Groups
- Add inbound rule: **TCP port 5005** from **0.0.0.0/0** (or your IP)

Then run the server:

```bash
cd /path/to/Ronai-Vision
python services/audio/rtp_tcp_server.py
```

**Options:**
```bash
python services/audio/rtp_tcp_server.py --rtp-port 5004 --tcp-port 5005
```

You should see:
```
RTP listener bound to 0.0.0.0:5004
TCP server listening on 0.0.0.0:5005
Server running. Press Ctrl+C to stop...
Clients can connect to: <EC2_IP>:5005
```

### Step 2: On Your Local Machine

Connect to the EC2 server:

```bash
python services/audio/rtp_tcp_client.py --server YOUR_EC2_IP
```

For example, if your EC2 IP is `54.123.45.67`:
```bash
python services/audio/rtp_tcp_client.py --server 54.123.45.67 --port 5005
```

**Options:**
```bash
python services/audio/rtp_tcp_client.py \
  --server 54.123.45.67 \
  --port 5005 \
  --sample-rate 16000 \
  --output-dir audio_storage/recordings
```

You should see:
```
Connected to RTP relay server!
Client started, receiving RTP packets...
First packet received, creating audio files...
WAV file created: audio_storage/recordings/20251216_203045_rtp_tcp_1734385845.wav
```

### Step 3: Test It!

When the radio device sends audio to EC2:
1. EC2 receives RTP packets on UDP port 5004
2. EC2 relays them to your local machine via TCP
3. Your local machine saves audio to WAV and PCM files
4. Files auto-save after 3 seconds of silence

## Benefits

✅ **No router configuration needed** - outbound connection from your machine
✅ **Works from anywhere** - coffee shop, office, home
✅ **Multiple clients** - multiple people can connect simultaneously
✅ **Reliable TCP** - no packet loss like UDP
✅ **Same sample rate** - 16kHz raw PCM audio preserved

## Troubleshooting

### Can't connect to EC2?
- Check EC2 security group allows TCP port 5005
- Verify EC2 public IP address
- Test connection: `telnet YOUR_EC2_IP 5005`

### No audio being saved?
- Check that radio device is sending RTP to EC2:5004
- Verify EC2 security group allows UDP port 5004
- Look for "Detected RTP source" in EC2 server logs

### Connection keeps dropping?
- Check your internet connection
- Try increasing timeout values
- Check EC2 server logs for errors

## Architecture Diagram

```
┌──────────────┐                  ┌──────────────┐                  ┌──────────────┐
│              │  RTP/UDP         │              │  RTP/TCP         │              │
│ Radio Device │─────────────────>│  EC2 Server  │─────────────────>│ Your Machine │
│              │  port 5004       │              │  port 5005       │              │
└──────────────┘                  └──────────────┘                  └──────────────┘
                                         │                                 │
                                         │                                 │
                                    Receives UDP                      Initiates TCP
                                    Relays to TCP                     Saves audio files
```

## Performance

- **Latency:** ~50-200ms depending on internet connection
- **Bandwidth:** ~128 kbps for 16kHz mono PCM
- **CPU Usage:** Minimal on both EC2 and local machine
