# context.py
_mom_manager = None
_transcription_manager = None
_emotion_manager = None

def init_all_managers():
    global _mom_manager, _transcription_manager, _emotion_manager
    from model_managers.mom_manager import MoMManager
    from model_managers.transcription_manager import TranscriptionManager
    #from model_managers.emotion_manager import EmotionRecognizer

    _mom_manager = MoMManager()
    _transcription_manager = TranscriptionManager()
    _emotion_manager =None
def get_mom_manager():
    return _mom_manager

def get_transcription_manager():
    return _transcription_manager

def get_emotion_manager():
    return _emotion_manager
