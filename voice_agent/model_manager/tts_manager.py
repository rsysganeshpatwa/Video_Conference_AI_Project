import os
import torch
import sys
from scipy.signal import resample
import soundfile as sf
from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter

# Extend system path to import OpenVoice components
OPENVOICE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "OpenVoice"))
sys.path.append(OPENVOICE_PATH)

# Constants
ckpt_base = 'OpenVoice/checkpoints/base_speakers/EN'
ckpt_converter = 'OpenVoice/checkpoints/converter'
reference_speaker = 'models/voice_sample.wav'

class TTSManager:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.cuda.empty_cache()
        self._load_models()

    def _load_models(self):
        print("[TTSManager] Loading models...")
        self.base_speaker_tts = BaseSpeakerTTS(f'{ckpt_base}/config.json', device=self.device)
        self.base_speaker_tts.load_ckpt(f'{ckpt_base}/checkpoint.pth')

        self.tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=self.device)
        self.tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

        self.source_se = torch.load(f'{ckpt_base}/en_default_se.pth').to(self.device)
        self.target_se, _ = se_extractor.get_se(
            reference_speaker, self.tone_color_converter, target_dir='processed', vad=True
        )
        print("[TTSManager] Models loaded successfully.")

    def synthesize(self, text, speaker="default", speed=1.0, save_as="output.wav", output_dir=output_dir):
        print(f"[TTSManager] Synthesizing: '{text}'")
        src_path = os.path.join(output_dir, 'tmp.wav')
        tmp_out = os.path.join(output_dir, 'tmp_out.wav')

        # TTS
        self.base_speaker_tts.tts(
            text=text,
            speaker=speaker,
            language="English",
            speed=speed,
            output_path=src_path
        )

        # Voice conversion
        encode_message = "@MyShell"
        self.tone_color_converter.convert(
            audio_src_path=src_path,
            src_se=self.source_se,
            tgt_se=self.target_se,
            output_path=tmp_out,
            message=encode_message
        )

        # Resample to 48kHz
        self._resample_to_48kHz(tmp_out, save_as)
        print(f"[TTSManager] Saved final output to: {save_as}")

    def _resample_to_48kHz(self, in_path, out_path):
        audio_data, sr = sf.read(in_path)
        if sr == 48000:
            sf.write(out_path, audio_data, 48000)
            return

        target_len = int(len(audio_data) * 48000 / sr)
        resampled = resample(audio_data, target_len)
        sf.write(out_path, resampled, 48000)
        print(f"[TTSManager] Resampled from {sr}Hz to 48000Hz")

# For testing
if __name__ == "__main__":
    tts = TTSManager()
    tts.synthesize("Hello! This is a test of OpenVoice synthesis.")
    #play
    
