#!/bin/bash
# Capture RTP packets using tcpdump for debugging
# This verifies that packets are actually arriving at the network interface

PORT="${1:-5004}"
DURATION="${2:-30}"
OUTPUT_FILE="rtp_capture_$(date +%Y%m%d_%H%M%S).pcap"

echo "=========================================="
echo "RTP Packet Capture"
echo "=========================================="
echo "Port: $PORT"
echo "Duration: ${DURATION}s"
echo "Output: $OUTPUT_FILE"
echo ""
echo "This will capture ALL UDP packets on port $PORT"
echo "Press Ctrl+C to stop early"
echo "=========================================="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Note: tcpdump requires sudo privileges"
    echo "You may be prompted for your password..."
    echo ""
fi

# Capture packets
sudo tcpdump -i any -n -w "$OUTPUT_FILE" -c 1000 "udp port $PORT" &
TCPDUMP_PID=$!

echo "Capturing packets... (PID: $TCPDUMP_PID)"
echo ""

# Wait for duration or Ctrl+C
sleep "$DURATION" 2>/dev/null || true

# Stop tcpdump
sudo kill -SIGINT $TCPDUMP_PID 2>/dev/null || true
wait $TCPDUMP_PID 2>/dev/null || true

echo ""
echo "=========================================="
echo "Capture complete!"
echo "=========================================="

# Check if file exists and has content
if [ -f "$OUTPUT_FILE" ]; then
    SIZE=$(stat -f%z "$OUTPUT_FILE" 2>/dev/null || stat -c%s "$OUTPUT_FILE" 2>/dev/null)
    echo "Captured file: $OUTPUT_FILE ($SIZE bytes)"
    
    # Try to read packet count
    echo ""
    echo "Analyzing capture..."
    sudo tcpdump -r "$OUTPUT_FILE" -n 2>/dev/null | head -20
    
    PACKET_COUNT=$(sudo tcpdump -r "$OUTPUT_FILE" -n 2>&1 | grep -c "UDP" || echo "0")
    echo ""
    echo "Total UDP packets: $PACKET_COUNT"
    
    if [ "$PACKET_COUNT" -eq 0 ]; then
        echo ""
        echo "⚠️  WARNING: No packets captured!"
        echo "Possible reasons:"
        echo "  1. No packets are being sent to this port"
        echo "  2. Firewall is blocking packets before they reach the interface"
        echo "  3. Wrong port number"
        echo "  4. Packets are being sent to different IP address"
    else
        echo ""
        echo "✓ Packets captured successfully!"
        echo ""
        echo "To analyze further:"
        echo "  tcpdump -r $OUTPUT_FILE -n -XX"
        echo "  wireshark $OUTPUT_FILE"
    fi
else
    echo "❌ Error: Capture file not created"
fi

echo "=========================================="
