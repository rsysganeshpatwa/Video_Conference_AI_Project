# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    portaudio19-dev \
    redis-server \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directory for uploaded files and models
RUN mkdir -p /app/uploads /app/logs

# Expose ports
EXPOSE 8765

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/model/emotion_model.pt
ENV REDIS_HOST=localhost
ENV REDIS_PORT=6379
ENV WS_HOST=0.0.0.0
ENV WS_PORT=8765

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import asyncio; import websockets; asyncio.run(websockets.connect('ws://localhost:8765').close())" || exit 1

# Run the application
CMD ["python", "run.py"]
