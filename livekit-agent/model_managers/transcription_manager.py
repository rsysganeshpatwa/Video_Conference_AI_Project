# managers/transcription_manager.py

# import whisper
import os
from utils.model_service import ModelService

class TranscriptionManager:
    def __init__(self, model_name: str = "base", device: str = "cuda"):
        try:
            #print(f"🎤 Loading Whisper model: {model_name} on {device}")
            #self.model = whisper.load_model(model_name, device=device)
            self.model_service = ModelService()
            
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load Whisper model: {e}")

    def transcribe(self, wav_files: list[tuple[str, str]], output_dir: str = ".") -> str:
        """
        Args:
            wav_files: List of tuples (identity, file_path)
            output_dir: Directory to save transcript.txt

        Returns:
            Full transcript string
        """
        lines = []
        for identity, path in wav_files:
            if not os.path.exists(path):
                print(f"⚠️ Skipping missing file: {path}")
                continue

            #result = self.model.transcribe(path, language='en', verbose=False)
            result = self.model_service.transcribe(path)
            print(f"transcript completed {result}")
            text = result
            if text:
                lines.append(f"{identity}: {text}")

        full_text = "\n".join(lines)

        transcript_path = os.path.join(output_dir, "transcript.txt")
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(full_text)

        return full_text
