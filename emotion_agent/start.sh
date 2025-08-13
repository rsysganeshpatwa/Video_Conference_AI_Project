#!/bin/bash

# Emotion Agent Startup Script

set -e

echo "Starting Emotion Agent..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create dummy model if it doesn't exist
if [ ! -f "model/emotion_model.pt" ]; then
    echo "Creating dummy emotion model..."
    python3 model/create_dummy_model.py
fi

# Set environment variables if not set
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Check if Redis is available
if command -v redis-cli &> /dev/null; then
    if redis-cli ping &> /dev/null; then
        echo "Redis is available"
        export REDIS_HOST=localhost
        export REDIS_PORT=6379
    else
        echo "Redis is not running, starting without Redis support"
    fi
else
    echo "Redis not installed, running without Redis support"
fi

# Start the emotion agent
echo "Starting Emotion Agent on WebSocket port 8765..."
python3 run.py
