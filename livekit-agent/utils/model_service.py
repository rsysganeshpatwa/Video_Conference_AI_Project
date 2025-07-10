# utils/model_service.py
import os
import requests

MODEL_SERVER_URL = os.getenv("MODEL_SERVER", "http://model-server:8000")

class ModelService:
    def __init__(self, base_url: str = MODEL_SERVER_URL):
        self.base_url = base_url

    def transcribe(self, audio_path: str) -> str:
        try:
            response = requests.post(f"{self.base_url}/transcribe", json={"audio_path": audio_path})
            response.raise_for_status()
            return response.json().get("text", "")
        except Exception as e:
            print(f"❌ Transcription failed: {e}")
            return ""

    def chat(self, user_query: str) -> str:
        try:
            response = requests.post(f"{self.base_url}/chat", json={
             
                "user_query": user_query
            })
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            print(f"❌ Chat failed: {e}")
            return ""
