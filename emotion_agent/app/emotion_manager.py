import os
import torch
import librosa
import numpy as np
import asyncio
import logging
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import json

logger = logging.getLogger(__name__)

@dataclass
class EmotionResult:
    """Emotion detection result."""
    user_id: str
    room_id: str
    timestamp: datetime
    predicted_emotion: str
    confidence: float
    emotion_scores: Dict[str, float]
    audio_duration: float

class EmotionManager:
    """
    Manages emotion detection for audio streams.
    Processes audio chunks and detects emotions in real-time.
    """
    
    def __init__(self, model_path: str = None, callback: Callable = None):
        """
        Initialize the emotion manager.
        
        Args:
            model_path: Path to the emotion recognition model
            callback: Callback function to handle emotion results
        """
        self.model_path = model_path or os.getenv("MODEL_PATH", "./model/emotion_model.pt")
        self.callback = callback
        self.model = None
        self.emotions = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
        self.sample_rate = 16000
        self.min_chunk_duration = 0.5  # Minimum duration in seconds
        self.max_chunk_duration = 5.0  # Maximum duration in seconds
        
        # Audio processing settings
        self.audio_buffer = {}  # Buffer for each user
        self.processing_queue = asyncio.Queue()
        self.is_running = False
        
        self.load_model()
    
    def load_model(self):
        """Load the emotion recognition model."""
        try:
            if os.path.exists(self.model_path):
                self.model = torch.load(self.model_path, map_location='cpu')
                self.model.eval()
                logger.info(f"Model loaded successfully from {self.model_path}")
            else:
                logger.warning(f"Model file not found at {self.model_path}. Using dummy model.")
                self.model = self._create_dummy_model()
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = self._create_dummy_model()
    
    def _create_dummy_model(self):
        """Create a dummy model for testing purposes."""
        class DummyModel:
            def __call__(self, features):
                # Return realistic emotion probabilities
                emotions = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
                # Generate more realistic probabilities with neutral being more common
                probs = np.random.dirichlet([0.5, 0.5, 0.5, 2.0, 5.0, 1.0, 0.5])
                return torch.tensor(probs).unsqueeze(0)
        return DummyModel()
    
    def extract_features(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract audio features for emotion recognition."""
        try:
            # Resample if necessary
            if sample_rate != self.sample_rate:
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=self.sample_rate)
            
            # Ensure audio is long enough
            min_samples = int(self.sample_rate * 0.1)  # 100ms minimum
            if len(audio_data) < min_samples:
                audio_data = np.pad(audio_data, (0, min_samples - len(audio_data)), 'constant')
            
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(
                y=audio_data, 
                sr=self.sample_rate, 
                n_mfcc=13,
                n_fft=2048,
                hop_length=512
            )
            
            # Extract spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=self.sample_rate)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)
            
            # Extract chroma features
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=self.sample_rate)
            
            # Combine features and handle variable length
            features = np.concatenate([
                np.mean(mfccs, axis=1),
                np.std(mfccs, axis=1),
                np.mean(spectral_centroids, axis=1),
                np.mean(spectral_rolloff, axis=1),
                np.mean(zero_crossing_rate, axis=1),
                np.mean(chroma, axis=1)
            ])
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            # Return dummy features if extraction fails
            return np.random.rand(40)
    
    async def process_audio_chunk(self, user_id: str, room_id: str, audio_data: np.ndarray, sample_rate: int) -> Optional[EmotionResult]:
        """
        Process an audio chunk and detect emotion.
        
        Args:
            user_id: ID of the user
            room_id: ID of the room
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio
            
        Returns:
            EmotionResult if successful, None otherwise
        """
        try:
            duration = len(audio_data) / sample_rate
            
            # Skip very short chunks
            if duration < self.min_chunk_duration:
                return None
            
            # Truncate very long chunks
            if duration > self.max_chunk_duration:
                max_samples = int(self.max_chunk_duration * sample_rate)
                audio_data = audio_data[:max_samples]
                duration = self.max_chunk_duration
            
            # Extract features
            features = self.extract_features(audio_data, sample_rate)
            
            # Convert to tensor
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            
            # Predict emotion
            with torch.no_grad():
                predictions = self.model(features_tensor)
                probabilities = torch.softmax(predictions, dim=1)
            
            # Convert to dictionary
            emotion_scores = {}
            for i, emotion in enumerate(self.emotions):
                emotion_scores[emotion] = float(probabilities[0][i])
            
            # Get the predicted emotion
            predicted_emotion = max(emotion_scores, key=emotion_scores.get)
            confidence = emotion_scores[predicted_emotion]
            
            # Create result
            result = EmotionResult(
                user_id=user_id,
                room_id=room_id,
                timestamp=datetime.now(),
                predicted_emotion=predicted_emotion,
                confidence=confidence,
                emotion_scores=emotion_scores,
                audio_duration=duration
            )
            
            # Call callback if provided
            if self.callback:
                await self.callback(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing audio chunk for user {user_id}: {e}")
            return None
    
    async def add_audio_chunk(self, user_id: str, room_id: str, audio_data: bytes, sample_rate: int = 16000):
        """
        Add audio chunk to processing queue.
        
        Args:
            user_id: ID of the user
            room_id: ID of the room
            audio_data: Raw audio bytes
            sample_rate: Sample rate of audio
        """
        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.float32)
            
            # Add to processing queue
            await self.processing_queue.put({
                'user_id': user_id,
                'room_id': room_id,
                'audio_data': audio_array,
                'sample_rate': sample_rate
            })
            
        except Exception as e:
            logger.error(f"Error adding audio chunk for user {user_id}: {e}")
    
    async def start_processing(self):
        """Start the audio processing loop."""
        self.is_running = True
        logger.info("Starting emotion processing loop")
        
        while self.is_running:
            try:
                # Wait for audio chunk with timeout
                chunk = await asyncio.wait_for(self.processing_queue.get(), timeout=1.0)
                
                # Process the chunk
                result = await self.process_audio_chunk(
                    chunk['user_id'],
                    chunk['room_id'],
                    chunk['audio_data'],
                    chunk['sample_rate']
                )
                
                if result:
                    logger.debug(f"Emotion detected for user {chunk['user_id']}: {result.predicted_emotion} ({result.confidence:.2f})")
                
            except asyncio.TimeoutError:
                # No chunks to process, continue
                continue
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                await asyncio.sleep(0.1)
    
    async def stop_processing(self):
        """Stop the audio processing loop."""
        self.is_running = False
        logger.info("Stopping emotion processing loop")
    
    def get_user_emotion_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """
        Get emotion history for a user.
        This would typically be stored in a database.
        """
        # This is a placeholder - in a real implementation, 
        # you'd fetch from a database
        return []
    
    def get_room_emotion_summary(self, room_id: str) -> Dict:
        """
        Get emotion summary for a room.
        This would typically aggregate data from a database.
        """
        # This is a placeholder - in a real implementation,
        # you'd aggregate data from a database
        return {
            "room_id": room_id,
            "active_users": 0,
            "dominant_emotion": "neutral",
            "emotion_distribution": {emotion: 0.0 for emotion in self.emotions}
        }
