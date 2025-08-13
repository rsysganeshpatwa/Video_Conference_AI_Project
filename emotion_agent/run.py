#!/usr/bin/env python3
"""
Emotion Agent Runner

This script starts the emotion recognition session manager.
"""

import asyncio
import os
import sys
import logging
from pathlib import Path

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent / "app"))

from app.session_manager import EmotionSessionManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Main function to run the emotion recognition session manager."""
    
    # Configuration
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", 6379))
    websocket_host = os.getenv("WS_HOST", "0.0.0.0")
    websocket_port = int(os.getenv("WS_PORT", 8765))
    
    logger.info("Starting Emotion Recognition Session Manager...")
    logger.info(f"Redis: {redis_host}:{redis_port}")
    logger.info(f"WebSocket: {websocket_host}:{websocket_port}")
    
    # Initialize session manager
    session_manager = EmotionSessionManager(
        redis_host=redis_host,
        redis_port=redis_port,
        websocket_host=websocket_host,
        websocket_port=websocket_port
    )
    
    try:
        # Start the session manager
        await session_manager.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        await session_manager.stop()

if __name__ == "__main__":
    asyncio.run(main())
