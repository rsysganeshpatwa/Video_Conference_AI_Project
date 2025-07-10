# context.py
from model_manager.llm_manager import LLMManager
from model_manager.stt_manager import STTManager
#from model_manager.tts_manager import TTSManager
from utils.tts import TTSManager as TTSManager
#from utils.llm import LLMManager as LLMManager

def create_managers():
    print("✅ Creating AI managers...")
    return {
        "llm": LLMManager(),
        "stt": STTManager(),
        "tts": TTSManager()
    }
