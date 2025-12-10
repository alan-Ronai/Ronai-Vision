#!/bin/bash
# Kill any process listening on a specific port
# Usage: ./scripts/kill_port.sh 8000

PORT=${1:-8000}

echo "Attempting to kill processes on port $PORT..."

# macOS and Linux compatible way to find and kill processes
PID=$(lsof -i :$PORT | grep -v COMMAND | awk '{print $2}' | head -1)

if [ -z "$PID" ]; then
    echo "No process found on port $PORT"
else
    echo "Found PID $PID on port $PORT, killing..."
    kill -9 $PID
    echo "Killed PID $PID"
fi
