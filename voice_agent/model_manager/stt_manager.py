#from faster_whisper import WhisperModel
import re
from utils.model_service import ModelService

class STTManager:
    def __init__(self, model_size_or_path: str = "models/ggml-medium.en.bin", compute_type: str = "int8", language='en'):
        #print("🎤 Loading Whisper STT...")
        self.model =  None#WhisperModel('medium', device="cuda",compute_type=compute_type)
        self.model_service = ModelService()
        self.language = language

    def correct_transcript(self, text: str) -> str:
        """Post-processing to fix common recognition errors."""
        corrections = {
            r"\bour systems\b": "R Systems",
            r"\bour system\b": "R System",
            r"\bthe r system\b": "R Systems",
            r"\br system\b": "R Systems",
            r"\art System\b":"R Systems",
            r"\Art Systems\b":"R Systems",
            r"\br systems international\b": "R Systems International",
            r"\bour systems international\b": "R Systems International",
            
        }
        for pattern, replacement in corrections.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def transcribe(self, audio_path: str):
        """Transcribe audio and apply correction logic."""
        
        # segments, info = self.model.transcribe(audio_path, language='en',  )
        # corrected_segments = []

        # for seg in segments:
        #     original = seg.text
        #     corrected = self.correct_transcript(original)
        #     seg.text = corrected
        #     corrected_segments.append(seg)
        segments  = self.model_service.transcribe(audio_path)
        
        # correct 
        corrected_segments = self.correct_transcript(segments)
        
        
        print(f"🔍 Transcription client result: {corrected_segments}")

        return corrected_segments, None
