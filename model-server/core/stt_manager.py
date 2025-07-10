from faster_whisper import WhisperModel
import os

class STTManager:
    def __init__(self, model_size="medium", device="cuda"):
        print(f"🎤 Loading Whisper model: {model_size} on {device}")
        self.model = WhisperModel(model_size, device=device, compute_type="float16")
        print("✅ Whisper model loaded successfully")

    def transcribe(self, audio_path: str) -> str:
        print(f"🔍 Received audio path: {audio_path}")
        if not audio_path or not isinstance(audio_path, str):
            raise ValueError("Audio path must be a non-empty string")
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"❌ File not found: {audio_path}")

        print("📊 Checking file metadata:")
        os.system(f"ffmpeg -i {audio_path} -hide_banner")

        print(f"🎧 Starting transcription for: {audio_path}")
        segments, info = self.model.transcribe(audio_path, language="en")
        segments = list(segments)  # 👈 Fix applied here
        
        if not segments:
            print("⚠️ No speech detected.")
            return "[no speech detected]"

        for i, seg in enumerate(segments):
            print(f"[Segment {i}] {seg.start:.2f}s - {seg.end:.2f}s: {seg.text}")

        full_text = " ".join([seg.text for seg in segments])
        print(f"✅ Transcription result: {full_text}")
        return full_text.strip()
