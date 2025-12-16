#!/bin/bash

# HAMAL-AI Start Script
# Starts all services in separate terminal windows/tabs

echo "🛡️  Starting HAMAL-AI Security Command Center"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "Checking prerequisites..."

if ! command_exists node; then
    echo "❌ Node.js not found. Please install Node.js 20+"
    exit 1
fi

if ! command_exists python3; then
    echo "❌ Python3 not found. Please install Python 3.11+"
    exit 1
fi

echo "✅ Prerequisites OK"
echo ""

# Install dependencies if needed
echo "Installing dependencies..."

# Backend
if [ ! -d "$SCRIPT_DIR/backend/node_modules" ]; then
    echo "${BLUE}Installing backend dependencies...${NC}"
    cd "$SCRIPT_DIR/backend" && npm install
fi

# Frontend
if [ ! -d "$SCRIPT_DIR/frontend/node_modules" ]; then
    echo "${BLUE}Installing frontend dependencies...${NC}"
    cd "$SCRIPT_DIR/frontend" && npm install
fi

# AI Service
if [ ! -d "$SCRIPT_DIR/ai-service/venv" ]; then
    echo "${BLUE}Creating AI service virtual environment...${NC}"
    cd "$SCRIPT_DIR/ai-service" && python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
fi

echo ""
echo "✅ Dependencies installed"
echo ""

# Start services based on OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS - use AppleScript to open new terminal tabs

    # Start Backend
    osascript -e "tell application \"Terminal\" to do script \"cd '$SCRIPT_DIR/backend' && npm run dev\""

    sleep 2

    # Start Frontend
    osascript -e "tell application \"Terminal\" to do script \"cd '$SCRIPT_DIR/frontend' && npm run dev\""

    sleep 2

    # Start AI Service
    osascript -e "tell application \"Terminal\" to do script \"cd '$SCRIPT_DIR/ai-service' && source venv/bin/activate && python main.py\""

    echo ""
    echo "🚀 All services started in separate terminal windows!"
    echo ""
    echo "Access the application:"
    echo "  - Frontend: http://localhost:5173"
    echo "  - Backend:  http://localhost:3000"
    echo "  - AI API:   http://localhost:8000"
    echo ""

else
    # Linux/Other - run in background
    echo "Starting services in background..."

    cd "$SCRIPT_DIR/backend" && npm run dev &
    BACKEND_PID=$!

    cd "$SCRIPT_DIR/frontend" && npm run dev &
    FRONTEND_PID=$!

    cd "$SCRIPT_DIR/ai-service" && source venv/bin/activate && python main.py &
    AI_PID=$!

    echo ""
    echo "🚀 All services started!"
    echo ""
    echo "PIDs:"
    echo "  - Backend:  $BACKEND_PID"
    echo "  - Frontend: $FRONTEND_PID"
    echo "  - AI:       $AI_PID"
    echo ""
    echo "To stop all: kill $BACKEND_PID $FRONTEND_PID $AI_PID"
    echo ""
    echo "Access the application:"
    echo "  - Frontend: http://localhost:5173"
    echo "  - Backend:  http://localhost:3000"
    echo "  - AI API:   http://localhost:8000"

    # Wait for all processes
    wait
fi
