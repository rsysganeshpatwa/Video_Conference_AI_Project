"""
Emotion Agent Application Package

This package contains the emotion recognition system for video conferencing.
It processes audio tracks in real-time and detects emotions from speech.
"""

from .emotion_manager import EmotionManager, EmotionResult
from .session_manager import EmotionSessionManager, Room
from .audio_processor import AudioProcessor

__version__ = "1.0.0"
__author__ = "Emotion Agent Team"

__all__ = [
    "EmotionManager",
    "EmotionResult", 
    "EmotionSessionManager",
    "Room",
    "AudioProcessor"
]
