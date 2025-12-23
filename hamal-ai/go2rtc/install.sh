#!/bin/bash
# go2rtc Installation Script for HAMAL-AI
# Downloads and sets up go2rtc binary

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Get latest version from GitHub API
echo "Checking latest go2rtc version..."
GO2RTC_VERSION=$(curl -s "https://api.github.com/repos/AlexxIT/go2rtc/releases/latest" | grep '"tag_name"' | sed -E 's/.*"v([^"]+)".*/\1/')

if [ -z "$GO2RTC_VERSION" ]; then
    GO2RTC_VERSION="1.9.13"  # Fallback version
fi

# Detect OS and architecture
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

case "$ARCH" in
    x86_64)
        ARCH="amd64"
        ;;
    arm64|aarch64)
        ARCH="arm64"
        ;;
    armv7l)
        ARCH="arm"
        ;;
    *)
        echo "Unsupported architecture: $ARCH"
        exit 1
        ;;
esac

case "$OS" in
    darwin)
        PLATFORM="mac"
        EXT=".zip"
        ;;
    linux)
        PLATFORM="linux"
        EXT=""
        ;;
    *)
        echo "Unsupported OS: $OS"
        exit 1
        ;;
esac

BINARY_NAME="go2rtc_${PLATFORM}_${ARCH}${EXT}"
DOWNLOAD_URL="https://github.com/AlexxIT/go2rtc/releases/download/v${GO2RTC_VERSION}/${BINARY_NAME}"

echo "=== go2rtc Installation ==="
echo "Version: ${GO2RTC_VERSION}"
echo "Platform: ${PLATFORM}"
echo "Architecture: ${ARCH}"
echo "Download: ${DOWNLOAD_URL}"
echo ""

# Check if already installed
if [ -f "$SCRIPT_DIR/go2rtc" ]; then
    echo "go2rtc binary already exists."
    CURRENT_VERSION=$("$SCRIPT_DIR/go2rtc" --version 2>/dev/null | head -1 || echo "unknown")
    echo "Current: $CURRENT_VERSION"

    read -p "Do you want to reinstall? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Keeping existing installation."
        exit 0
    fi
    rm -f "$SCRIPT_DIR/go2rtc"
fi

echo "Downloading go2rtc..."

if [ "$EXT" = ".zip" ]; then
    # macOS - download and extract zip
    curl -L -o "$SCRIPT_DIR/go2rtc.zip" "$DOWNLOAD_URL"
    cd "$SCRIPT_DIR"
    unzip -o go2rtc.zip
    rm go2rtc.zip
else
    # Linux - direct binary download
    curl -L -o "$SCRIPT_DIR/go2rtc" "$DOWNLOAD_URL"
fi

echo "Making binary executable..."
chmod +x "$SCRIPT_DIR/go2rtc"

echo ""
echo "=== Installation Complete ==="
"$SCRIPT_DIR/go2rtc" --version
echo ""
echo "To start go2rtc:"
echo "  cd $SCRIPT_DIR && ./start.sh"
echo ""
echo "Web UI will be available at: http://localhost:1984"
