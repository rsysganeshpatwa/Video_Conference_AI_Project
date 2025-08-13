#!/usr/bin/env python3
"""
Example client for the Emotion Agent.

This script demonstrates how to connect to the emotion recognition system,
join a room, and send audio data for emotion processing.
"""

import asyncio
import websockets
import json
import base64
import numpy as np
import logging
from typing import Optional
import soundfile as sf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionAgentClient:
    """Client for connecting to the emotion recognition service."""
    
    def __init__(self, server_url: str = "ws://localhost:8765"):
        """
        Initialize the client.
        
        Args:
            server_url: WebSocket URL of the emotion agent server
        """
        self.server_url = server_url
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.user_id: Optional[str] = None
        self.room_id: Optional[str] = None
        self.is_connected = False
    
    async def connect(self):
        """Connect to the emotion agent server."""
        try:
            self.websocket = await websockets.connect(self.server_url)
            self.is_connected = True
            logger.info(f"Connected to emotion agent at {self.server_url}")
            
            # Start message handler
            asyncio.create_task(self._handle_messages())
            
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from the server."""
        if self.websocket:
            await self.websocket.close()
        self.is_connected = False
        logger.info("Disconnected from emotion agent")
    
    async def join_room(self, user_id: str, room_id: str, user_info: dict = None):
        """
        Join a room for emotion processing.
        
        Args:
            user_id: Unique identifier for the user
            room_id: Room identifier
            user_info: Optional user information
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to server")
        
        self.user_id = user_id
        self.room_id = room_id
        
        message = {
            'type': 'join_room',
            'user_id': user_id,
            'room_id': room_id,
            'user_info': user_info or {}
        }
        
        await self.websocket.send(json.dumps(message))
        logger.info(f"Requested to join room {room_id} as user {user_id}")
    
    async def leave_room(self):
        """Leave the current room."""
        if not self.is_connected or not self.room_id:
            return
        
        message = {
            'type': 'leave_room',
            'room_id': self.room_id,
            'user_id': self.user_id
        }
        
        await self.websocket.send(json.dumps(message))
        logger.info(f"Left room {self.room_id}")
        
        self.room_id = None
        self.user_id = None
    
    async def add_audio_track(self, track_id: str):
        """
        Notify server about a new audio track.
        
        Args:
            track_id: Unique identifier for the audio track
        """
        if not self.is_connected:
            return
        
        message = {
            'type': 'audio_track_added',
            'track_id': track_id
        }
        
        await self.websocket.send(json.dumps(message))
        logger.info(f"Added audio track {track_id}")
    
    async def remove_audio_track(self, track_id: str):
        """
        Notify server about audio track removal.
        
        Args:
            track_id: Unique identifier for the audio track
        """
        if not self.is_connected:
            return
        
        message = {
            'type': 'audio_track_removed',
            'track_id': track_id
        }
        
        await self.websocket.send(json.dumps(message))
        logger.info(f"Removed audio track {track_id}")
    
    async def send_audio_data(self, audio_data: np.ndarray, 
                            sample_rate: int = 16000,
                            track_id: str = "default"):
        """
        Send audio data for emotion processing.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            track_id: Audio track identifier
        """
        if not self.is_connected:
            return
        
        # Convert audio to bytes and encode as base64
        audio_bytes = audio_data.astype(np.float32).tobytes()
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        message = {
            'type': 'audio_data',
            'audio_data': audio_b64,
            'sample_rate': sample_rate,
            'track_id': track_id
        }
        
        await self.websocket.send(json.dumps(message))
    
    async def get_room_status(self):
        """Get the current room status."""
        if not self.is_connected:
            return
        
        message = {
            'type': 'get_room_status'
        }
        
        await self.websocket.send(json.dumps(message))
    
    async def _handle_messages(self):
        """Handle incoming messages from the server."""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    message_type = data.get('type')
                    
                    if message_type == 'room_joined':
                        await self._handle_room_joined(data['data'])
                    
                    elif message_type == 'emotion_update':
                        await self._handle_emotion_update(data['data'])
                    
                    elif message_type == 'room_status':
                        await self._handle_room_status(data['data'])
                    
                    elif message_type == 'audio_track_subscribed':
                        await self._handle_track_subscribed(data['data'])
                    
                    elif message_type == 'error':
                        logger.error(f"Server error: {data['message']}")
                    
                    else:
                        logger.debug(f"Unknown message type: {message_type}")
                
                except json.JSONDecodeError:
                    logger.error("Invalid JSON received from server")
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
        
        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection to server closed")
            self.is_connected = False
        except Exception as e:
            logger.error(f"Message handling error: {e}")
    
    async def _handle_room_joined(self, data):
        """Handle room joined confirmation."""
        logger.info(f"Successfully joined room {data['room_id']}")
        logger.info(f"Participants: {data['participants']}")
    
    async def _handle_emotion_update(self, data):
        """Handle emotion update from another user."""
        logger.info(f"Emotion update from user {data['user_id']}: "
                   f"{data['emotion']} (confidence: {data['confidence']:.2f})")
    
    async def _handle_room_status(self, data):
        """Handle room status response."""
        logger.info(f"Room status: {json.dumps(data, indent=2)}")
    
    async def _handle_track_subscribed(self, data):
        """Handle audio track subscription confirmation."""
        logger.info(f"Audio track {data['track_id']} subscribed for user {data['user_id']}")

async def simulate_audio_stream(client: EmotionAgentClient, duration: int = 30):
    """
    Simulate sending audio data to the emotion agent.
    
    Args:
        client: Emotion agent client
        duration: Duration to stream in seconds
    """
    sample_rate = 16000
    chunk_duration = 1.0  # 1 second chunks
    chunk_samples = int(sample_rate * chunk_duration)
    
    logger.info(f"Starting audio simulation for {duration} seconds")
    
    for i in range(duration):
        # Generate dummy audio data (sine wave with some noise)
        t = np.linspace(0, chunk_duration, chunk_samples)
        frequency = 440 + (i % 10) * 50  # Varying frequency
        audio_chunk = 0.1 * np.sin(2 * np.pi * frequency * t)
        audio_chunk += 0.05 * np.random.randn(len(audio_chunk))  # Add noise
        
        # Send audio chunk
        await client.send_audio_data(audio_chunk, sample_rate)
        
        # Wait before next chunk
        await asyncio.sleep(chunk_duration)
    
    logger.info("Audio simulation completed")

async def main():
    """Main function demonstrating client usage."""
    client = EmotionAgentClient()
    
    try:
        # Connect to server
        await client.connect()
        
        # Join a room
        user_id = "test_user_1"
        room_id = "test_room"
        await client.join_room(user_id, room_id, {"name": "Test User"})
        
        # Wait a bit for room join confirmation
        await asyncio.sleep(1)
        
        # Add an audio track
        await client.add_audio_track("audio_track_1")
        
        # Get room status
        await client.get_room_status()
        
        # Simulate audio streaming
        await simulate_audio_stream(client, duration=10)
        
        # Leave room
        await client.leave_room()
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
