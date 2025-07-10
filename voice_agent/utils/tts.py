import torch
import numpy as np
import scipy.io.wavfile as wavfile
from torch.serialization import safe_globals
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts, XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
import soundfile as sf
from scipy.signal import resample
import re

class TTSManager:
    def __init__(self, model_dir="XTTS-v2"):
        self.model_dir = model_dir
        self.model = None
        self.config = None
        self.tokenizer = None  # Placeholder for tokenizer, if needed
        
        self._initialize_model()
        self.reference_voice_path = "models/voice_sample.wav"  # Default reference voice path

    def _initialize_model(self):
        print("[XTTS] Initializing model...")

        with safe_globals([
            XttsConfig,
            XttsAudioConfig,
            BaseDatasetConfig,
            Xtts,
            XttsArgs
        ]):
            self.config = XttsConfig()
            self.config.load_json(f"{self.model_dir}/config.json")
            self.model = Xtts.init_from_config(self.config)
            self.model.load_checkpoint(self.config, checkpoint_dir=self.model_dir, eval=True)
            self.tokenizer = self.model.tokenizer
            self.tokenizer.char_limits["en"] = 1000
            self.tokenizer.gpt_max_text_tokens = 1000  # Adjusted for XTTS

        if torch.cuda.is_available():
            self.model.cuda()

        print("[XTTS] Model initialized.")

    def synthesize(self, text: str, save_as: str,out_path: str):
        if self.model is None or self.config is None:
            raise RuntimeError("Model not initialized.")

        print(f"[XTTS] Synthesizing: {text}")
        try:
            outputs = self.model.synthesize(
                text=text,
                config=self.config,
                speaker_wav=self.reference_voice_path,
                language="en"
            )
            temp_save_path = f"{out_path}/temp_output.wav"
            self._save_wav(outputs["wav"], filename=temp_save_path)
            self._resample_to_48kHz(temp_save_path, save_as)  # Ensure output is at 48kHz
            print(f"[XTTS] Audio saved to {save_as}")
        except Exception as e:
            print(f"[XTTS] Error during synthesis: {e}")
            raise
        
    def _resample_to_48kHz(self, in_path: str, out_path: str):
        audio_data, sr = sf.read(in_path)
        if sr == 48000:
            sf.write(out_path, audio_data, 48000)
            print(f"[TTSManager] No resampling needed. Already at 48000Hz.")
            return

        target_len = int(len(audio_data) * 48000 / sr)
        resampled = resample(audio_data, target_len)
        sf.write(out_path, resampled, 48000)
        print(f"[TTSManager] Resampled from {sr}Hz to 48000Hz and saved to {out_path}")




    def chunk_text(self, text: str, lang="en", max_tokens=400, max_chars=250):
        if not text or not isinstance(text, str):
            print("[chunk_text] ⚠️ Input text is empty or invalid.")
            return []

        print(f"[chunk_text] 🔍 Original text: {text[:100]}{'...' if len(text) > 100 else ''}")
        
        # 1. Split into sentences
        try:
            pattern = r'(?<=[.!?])\s+|(?=\n\d+\.\s)'


            sentences = re.split(pattern, text.strip())
        except Exception as e:
            print(f"[chunk_text] ❌ Sentence split failed: {e}")
            return []
        
        print(f"[chunk_text] 🧩 Sentences found: {len(sentences)}")
        if not sentences:
            return []

        chunks = []
        current_chunk = ""

        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                print(f"[chunk_text] ⚠️ Skipping empty sentence {i+1}")
                continue

            print(f"\n[chunk_text] ✨ Sentence {i+1}: {sentence}")

            test_chunk = f"{current_chunk} {sentence}".strip() if current_chunk else sentence

            try:
                token_len = len(self.tokenizer.encode(test_chunk, lang=lang))
                print(f"[chunk_text] 🔠 Token length: {token_len}, Char length: {len(test_chunk)}")
            except Exception as e:
                print(f"[chunk_text] ❌ Tokenizer failed: {e}")
                continue

            if token_len <= max_tokens and len(test_chunk) <= max_chars:
                current_chunk = test_chunk
                print(f"[chunk_text] ✅ Appended to current_chunk.")
            else:
                if current_chunk:
                    print(f"[chunk_text] 📦 Pushing chunk: {current_chunk}")
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
                print(f"[chunk_text] 🆕 Starting new chunk.")

        if current_chunk:
            print(f"\n[chunk_text] 📦 Final chunk: {current_chunk}")
            chunks.append(current_chunk.strip())

        print(f"\n[chunk_text] ✅ Done. Total chunks: {len(chunks)}")
        return chunks



    @staticmethod
    def _save_wav(wav: np.ndarray, sample_rate: int = 24000, filename: str = 'output.wav'):
        wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
        wav_norm = wav_norm.astype(np.int16)
        wavfile.write(filename=filename, rate=sample_rate, data=wav_norm)


# Standalone test
if __name__ == "__main__":
    tts = TTSManager(model_dir="XTTS-v2")
    tts.synthesize(
        text="Hello, this is a test of the XTTS synthesis.",
        
        save_as="output.wav"
    )
