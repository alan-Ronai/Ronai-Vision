#!/bin/bash
# go2rtc Startup Script for HAMAL-AI

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/go2rtc.yaml"

# Check if binary exists
if [ ! -f "$SCRIPT_DIR/go2rtc" ]; then
    echo "go2rtc binary not found. Running install script..."
    bash "$SCRIPT_DIR/install.sh"
fi

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found at $CONFIG_FILE"
    exit 1
fi

echo "=== Starting go2rtc ==="
echo "Config: $CONFIG_FILE"
echo "Web UI: http://localhost:1984"
echo "WebRTC: udp://localhost:8555"
echo "RTSP:   rtsp://localhost:8554"
echo ""

cd "$SCRIPT_DIR"
exec ./go2rtc -config "$CONFIG_FILE"
