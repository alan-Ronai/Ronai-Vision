#!/bin/bash
# Start script for AI service with GStreamer support on macOS
# This sets the required library paths before starting Python

# GStreamer library paths for Homebrew on Apple Silicon
export GST_PLUGIN_PATH=/opt/homebrew/lib/gstreamer-1.0
export GI_TYPELIB_PATH=/opt/homebrew/lib/girepository-1.0
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/glib/lib:/opt/homebrew/lib:$DYLD_LIBRARY_PATH

# Verify GStreamer is working
echo "Testing GStreamer..."
if gst-launch-1.0 --version 2>/dev/null | head -1; then
    echo "✓ GStreamer is working"
else
    echo "✗ GStreamer test failed"
    echo "Try running: brew reinstall gstreamer gst-plugins-base gst-plugins-good"
    exit 1
fi

# Change to script directory
cd "$(dirname "$0")"

# Load environment variables from dev.env
if [ -f dev.env ]; then
    export $(grep -v '^#' dev.env | grep -v '^$' | xargs)
fi

# Start the AI service
echo "Starting AI service with GStreamer backend..."
python main.py
