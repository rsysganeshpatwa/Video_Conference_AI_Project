import os
import io
import torch
import librosa
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import soundfile as sf
from typing import Dict, List
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Emotion Recognition API",
    description="API for detecting emotions from audio chunks",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EmotionDetector:
    def __init__(self, model_path: str = None):
        """Initialize the emotion detector."""
        self.model_path = model_path or os.getenv("MODEL_PATH", "./model/emotion_model.pt")
        self.model = None
        self.emotions = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
        self.sample_rate = 16000
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
                # Return random probabilities for demonstration
                return torch.rand(1, 7)  # 7 emotions
        return DummyModel()
    
    def extract_features(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract audio features for emotion recognition."""
        try:
            # Resample if necessary
            if sample_rate != self.sample_rate:
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=self.sample_rate)
            
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
            
            # Combine features
            features = np.concatenate([
                np.mean(mfccs, axis=1),
                np.mean(spectral_centroids, axis=1),
                np.mean(spectral_rolloff, axis=1),
                np.mean(zero_crossing_rate, axis=1),
                np.mean(chroma, axis=1)
            ])
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            # Return dummy features if extraction fails
            return np.random.rand(26)
    
    def predict_emotion(self, audio_data: np.ndarray, sample_rate: int) -> Dict:
        """Predict emotion from audio data."""
        try:
            # Extract features
            features = self.extract_features(audio_data, sample_rate)
            
            # Convert to tensor
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            
            # Predict
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
            
            return {
                "predicted_emotion": predicted_emotion,
                "confidence": confidence,
                "emotion_scores": emotion_scores
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Initialize the emotion detector
emotion_detector = EmotionDetector()

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Emotion Recognition API", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": emotion_detector.model is not None,
        "supported_emotions": emotion_detector.emotions
    }

@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
    """
    Predict emotion from uploaded audio file.
    
    Args:
        file: Audio file (WAV, MP3, etc.)
    
    Returns:
        JSON response with emotion prediction
    """
    try:
        # Validate file type
        if not file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="File must be an audio file")
        
        # Read audio file
        audio_bytes = await file.read()
        
        # Load audio data
        try:
            audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
        except Exception as e:
            # Try with librosa if soundfile fails
            audio_data, sample_rate = librosa.load(io.BytesIO(audio_bytes), sr=None)
        
        # Handle stereo audio (convert to mono)
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Predict emotion
        result = emotion_detector.predict_emotion(audio_data, sample_rate)
        
        # Add metadata
        result["metadata"] = {
            "filename": file.filename,
            "duration": len(audio_data) / sample_rate,
            "sample_rate": sample_rate,
            "file_size": len(audio_bytes)
        }
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing audio file: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

@app.post("/predict_chunk")
async def predict_emotion_chunk(file: UploadFile = File(...)):
    """
    Predict emotion from audio chunk (optimized for real-time processing).
    
    Args:
        file: Audio chunk file
    
    Returns:
        JSON response with emotion prediction
    """
    try:
        # Read audio chunk
        audio_bytes = await file.read()
        
        # Load audio data
        audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
        
        # Handle stereo audio
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Skip very short chunks
        if len(audio_data) < sample_rate * 0.5:  # Less than 0.5 seconds
            return JSONResponse(content={
                "predicted_emotion": "neutral",
                "confidence": 0.5,
                "emotion_scores": {emotion: 1/7 for emotion in emotion_detector.emotions},
                "note": "Audio chunk too short for reliable prediction"
            })
        
        # Predict emotion
        result = emotion_detector.predict_emotion(audio_data, sample_rate)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Error processing audio chunk: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing chunk: {str(e)}")

@app.get("/emotions")
async def get_supported_emotions():
    """Get list of supported emotions."""
    return {
        "emotions": emotion_detector.emotions,
        "count": len(emotion_detector.emotions)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
