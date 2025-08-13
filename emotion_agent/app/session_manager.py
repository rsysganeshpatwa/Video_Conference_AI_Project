import asyncio
import websockets
import json
import redis.asyncio as redis
import logging
from typing import Dict, Set, Optional, Any
from datetime import datetime
import uuid
from dataclasses import asdict

from .emotion_manager import EmotionManager, EmotionResult
from .audio_processor import AudioProcessor

logger = logging.getLogger(__name__)

class Room:
    """Represents a conference room with active participants."""
    
    def __init__(self, room_id: str):
        self.room_id = room_id
        self.participants: Dict[str, Dict] = {}  # user_id -> participant info
        self.created_at = datetime.now()
        self.active_tracks: Dict[str, Set[str]] = {}  # user_id -> set of track_ids
    
    def add_participant(self, user_id: str, websocket, participant_info: Dict = None):
        """Add a participant to the room."""
        self.participants[user_id] = {
            'websocket': websocket,
            'joined_at': datetime.now(),
            'info': participant_info or {}
        }
        self.active_tracks[user_id] = set()
        logger.info(f"User {user_id} joined room {self.room_id}")
    
    def remove_participant(self, user_id: str):
        """Remove a participant from the room."""
        if user_id in self.participants:
            del self.participants[user_id]
            self.active_tracks.pop(user_id, None)
            logger.info(f"User {user_id} left room {self.room_id}")
    
    def add_audio_track(self, user_id: str, track_id: str):
        """Add an audio track for a user."""
        if user_id in self.active_tracks:
            self.active_tracks[user_id].add(track_id)
            logger.info(f"Audio track {track_id} added for user {user_id} in room {self.room_id}")
    
    def remove_audio_track(self, user_id: str, track_id: str):
        """Remove an audio track for a user."""
        if user_id in self.active_tracks and track_id in self.active_tracks[user_id]:
            self.active_tracks[user_id].remove(track_id)
            logger.info(f"Audio track {track_id} removed for user {user_id} in room {self.room_id}")
    
    async def broadcast_emotion(self, emotion_result: EmotionResult):
        """Broadcast emotion result to all participants in the room."""
        message = {
            'type': 'emotion_update',
            'data': {
                'user_id': emotion_result.user_id,
                'room_id': emotion_result.room_id,
                'timestamp': emotion_result.timestamp.isoformat(),
                'emotion': emotion_result.predicted_emotion,
                'confidence': emotion_result.confidence,
                'emotion_scores': emotion_result.emotion_scores
            }
        }
        
        # Send to all participants except the sender
        for user_id, participant in self.participants.items():
            if user_id != emotion_result.user_id:
                try:
                    await participant['websocket'].send(json.dumps(message))
                except Exception as e:
                    logger.error(f"Error sending emotion update to user {user_id}: {e}")

class EmotionSessionManager:
    """
    Manages emotion detection sessions for multiple rooms.
    Handles WebRTC audio track subscriptions and real-time emotion processing.
    """
    
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379, 
                 websocket_host: str = "0.0.0.0", websocket_port: int = 8765):
        """Initialize the session manager."""
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.websocket_host = websocket_host
        self.websocket_port = websocket_port
        
        # Components
        self.emotion_manager = EmotionManager(callback=self._handle_emotion_result)
        self.audio_processor = AudioProcessor()
        
        # State management
        self.rooms: Dict[str, Room] = {}
        self.user_sessions: Dict[str, Dict] = {}  # websocket -> user info
        self.redis_client: Optional[redis.Redis] = None
        
        # Processing
        self.is_running = False
        self.websocket_server = None
    
    async def start(self):
        """Start the session manager."""
        try:
            # Connect to Redis
            self.redis_client = redis.Redis(host=self.redis_host, port=self.redis_port, decode_responses=True)
            await self.redis_client.ping()
            logger.info(f"Connected to Redis at {self.redis_host}:{self.redis_port}")
        except Exception as e:
            logger.warning(f"Could not connect to Redis: {e}. Running without Redis.")
            self.redis_client = None
        
        # Start emotion processing
        emotion_task = asyncio.create_task(self.emotion_manager.start_processing())
        
        # Start WebSocket server
        logger.info(f"Starting WebSocket server on {self.websocket_host}:{self.websocket_port}")
        self.websocket_server = await websockets.serve(
            self._handle_websocket_connection,
            self.websocket_host,
            self.websocket_port
        )
        
        self.is_running = True
        logger.info("Emotion Session Manager started successfully")
        
        # Keep running
        try:
            await asyncio.gather(emotion_task)
        except Exception as e:
            logger.error(f"Error in session manager: {e}")
    
    async def stop(self):
        """Stop the session manager."""
        self.is_running = False
        
        # Stop emotion processing
        await self.emotion_manager.stop_processing()
        
        # Close WebSocket server
        if self.websocket_server:
            self.websocket_server.close()
            await self.websocket_server.wait_closed()
        
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Emotion Session Manager stopped")
    
    async def _handle_websocket_connection(self, websocket, path):
        """Handle new WebSocket connections."""
        user_id = None
        room_id = None
        
        try:
            logger.info(f"New WebSocket connection from {websocket.remote_address}")
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    message_type = data.get('type')
                    
                    if message_type == 'join_room':
                        user_id, room_id = await self._handle_join_room(websocket, data)
                    
                    elif message_type == 'leave_room':
                        await self._handle_leave_room(websocket, data)
                    
                    elif message_type == 'audio_track_added':
                        await self._handle_audio_track_added(websocket, data)
                    
                    elif message_type == 'audio_track_removed':
                        await self._handle_audio_track_removed(websocket, data)
                    
                    elif message_type == 'audio_data':
                        await self._handle_audio_data(websocket, data)
                    
                    elif message_type == 'get_room_status':
                        await self._handle_get_room_status(websocket, data)
                    
                    else:
                        logger.warning(f"Unknown message type: {message_type}")
                
                except json.JSONDecodeError:
                    logger.error("Invalid JSON received")
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
        
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            # Clean up on disconnect
            if user_id and room_id:
                await self._cleanup_user_session(user_id, room_id, websocket)
    
    async def _handle_join_room(self, websocket, data) -> tuple:
        """Handle user joining a room."""
        user_id = data.get('user_id')
        room_id = data.get('room_id')
        user_info = data.get('user_info', {})
        
        if not user_id or not room_id:
            await self._send_error(websocket, "Missing user_id or room_id")
            return None, None
        
        # Create room if it doesn't exist
        if room_id not in self.rooms:
            self.rooms[room_id] = Room(room_id)
            logger.info(f"Created new room: {room_id}")
        
        # Add user to room
        room = self.rooms[room_id]
        room.add_participant(user_id, websocket, user_info)
        
        # Store session info
        self.user_sessions[websocket] = {
            'user_id': user_id,
            'room_id': room_id,
            'joined_at': datetime.now()
        }
        
        # Send confirmation
        await websocket.send(json.dumps({
            'type': 'room_joined',
            'data': {
                'room_id': room_id,
                'user_id': user_id,
                'participants': list(room.participants.keys())
            }
        }))
        
        # Store in Redis if available
        if self.redis_client:
            await self.redis_client.sadd(f"room:{room_id}:participants", user_id)
            await self.redis_client.hset(f"user:{user_id}", mapping={
                'room_id': room_id,
                'status': 'active',
                'joined_at': datetime.now().isoformat()
            })
        
        return user_id, room_id
    
    async def _handle_leave_room(self, websocket, data):
        """Handle user leaving a room."""
        session = self.user_sessions.get(websocket)
        if not session:
            return
        
        user_id = session['user_id']
        room_id = session['room_id']
        
        await self._cleanup_user_session(user_id, room_id, websocket)
    
    async def _handle_audio_track_added(self, websocket, data):
        """Handle audio track being added."""
        session = self.user_sessions.get(websocket)
        if not session:
            await self._send_error(websocket, "Not in a room")
            return
        
        user_id = session['user_id']
        room_id = session['room_id']
        track_id = data.get('track_id')
        
        if not track_id:
            await self._send_error(websocket, "Missing track_id")
            return
        
        # Add track to room
        if room_id in self.rooms:
            self.rooms[room_id].add_audio_track(user_id, track_id)
        
        # Confirm track subscription
        await websocket.send(json.dumps({
            'type': 'audio_track_subscribed',
            'data': {
                'user_id': user_id,
                'track_id': track_id
            }
        }))
    
    async def _handle_audio_track_removed(self, websocket, data):
        """Handle audio track being removed."""
        session = self.user_sessions.get(websocket)
        if not session:
            return
        
        user_id = session['user_id']
        room_id = session['room_id']
        track_id = data.get('track_id')
        
        if track_id and room_id in self.rooms:
            self.rooms[room_id].remove_audio_track(user_id, track_id)
    
    async def _handle_audio_data(self, websocket, data):
        """Handle incoming audio data for emotion processing."""
        session = self.user_sessions.get(websocket)
        if not session:
            return
        
        user_id = session['user_id']
        room_id = session['room_id']
        
        # Extract audio data
        audio_bytes = data.get('audio_data')  # Base64 encoded audio
        sample_rate = data.get('sample_rate', 16000)
        track_id = data.get('track_id')
        
        if not audio_bytes:
            return
        
        try:
            # Decode base64 audio data
            import base64
            audio_binary = base64.b64decode(audio_bytes)
            
            # Process audio chunk
            await self.emotion_manager.add_audio_chunk(
                user_id=user_id,
                room_id=room_id,
                audio_data=audio_binary,
                sample_rate=sample_rate
            )
            
        except Exception as e:
            logger.error(f"Error processing audio data from user {user_id}: {e}")
    
    async def _handle_get_room_status(self, websocket, data):
        """Handle room status request."""
        session = self.user_sessions.get(websocket)
        if not session:
            await self._send_error(websocket, "Not in a room")
            return
        
        room_id = session['room_id']
        
        if room_id not in self.rooms:
            await self._send_error(websocket, "Room not found")
            return
        
        room = self.rooms[room_id]
        
        # Get room status
        status = {
            'room_id': room_id,
            'participants': list(room.participants.keys()),
            'active_tracks': {user_id: list(tracks) for user_id, tracks in room.active_tracks.items()},
            'created_at': room.created_at.isoformat()
        }
        
        await websocket.send(json.dumps({
            'type': 'room_status',
            'data': status
        }))
    
    async def _cleanup_user_session(self, user_id: str, room_id: str, websocket):
        """Clean up user session on disconnect."""
        try:
            # Remove from room
            if room_id in self.rooms:
                self.rooms[room_id].remove_participant(user_id)
                
                # Remove empty rooms
                if not self.rooms[room_id].participants:
                    del self.rooms[room_id]
                    logger.info(f"Removed empty room: {room_id}")
            
            # Remove session
            self.user_sessions.pop(websocket, None)
            
            # Clean up Redis
            if self.redis_client:
                await self.redis_client.srem(f"room:{room_id}:participants", user_id)
                await self.redis_client.delete(f"user:{user_id}")
            
            logger.info(f"Cleaned up session for user {user_id} in room {room_id}")
            
        except Exception as e:
            logger.error(f"Error cleaning up session: {e}")
    
    async def _handle_emotion_result(self, emotion_result: EmotionResult):
        """Handle emotion detection result."""
        try:
            # Broadcast to room participants
            if emotion_result.room_id in self.rooms:
                await self.rooms[emotion_result.room_id].broadcast_emotion(emotion_result)
            
            # Store in Redis if available
            if self.redis_client:
                emotion_data = asdict(emotion_result)
                emotion_data['timestamp'] = emotion_result.timestamp.isoformat()
                
                # Store emotion result
                await self.redis_client.lpush(
                    f"emotions:{emotion_result.room_id}:{emotion_result.user_id}",
                    json.dumps(emotion_data)
                )
                
                # Keep only last 100 results per user
                await self.redis_client.ltrim(
                    f"emotions:{emotion_result.room_id}:{emotion_result.user_id}",
                    0, 99
                )
            
            logger.debug(f"Processed emotion result: {emotion_result.user_id} -> {emotion_result.predicted_emotion}")
            
        except Exception as e:
            logger.error(f"Error handling emotion result: {e}")
    
    async def _send_error(self, websocket, message: str):
        """Send error message to client."""
        try:
            await websocket.send(json.dumps({
                'type': 'error',
                'message': message
            }))
        except Exception as e:
            logger.error(f"Error sending error message: {e}")
    
    def get_room_stats(self) -> Dict:
        """Get statistics about active rooms."""
        return {
            'total_rooms': len(self.rooms),
            'total_participants': sum(len(room.participants) for room in self.rooms.values()),
            'rooms': {
                room_id: {
                    'participants': len(room.participants),
                    'active_tracks': sum(len(tracks) for tracks in room.active_tracks.values())
                }
                for room_id, room in self.rooms.items()
            }
        }
