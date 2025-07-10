# utils/vad_utils.py
import torch
import numpy as np
import soundfile as sf
from scipy.signal import resample

# Load Silero VAD model and utilities
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

ORIGINAL_SAMPLE_RATE = 48000
VAD_SAMPLE_RATE = 16000
MIN_FRAME_SAMPLES = 3200  # ~0.2s at 16kHz

class VADProcessor:
    def __init__(self, input_sample_rate=ORIGINAL_SAMPLE_RATE, vad_sample_rate=VAD_SAMPLE_RATE):
        self.sample_rate = input_sample_rate
        self.vad_sample_rate = vad_sample_rate
        self.model = model
        self.speech_segments = []

    def resample_to_vad(self, audio_np: np.ndarray) -> np.ndarray:
        """Resample from original sample rate to 16kHz"""
        target_len = int(len(audio_np) * self.vad_sample_rate / self.sample_rate)
        return resample(audio_np, target_len).astype(np.float32)

    def is_human_voice(self, audio_np: np.ndarray) -> bool:
        #print(f"[VAD] Raw shape: {audio_np.shape}, dtype: {audio_np.dtype}, max: {np.max(audio_np)}, min: {np.min(audio_np)}")

        # Resample to 16kHz
        audio_resampled = self.resample_to_vad(audio_np)

        if len(audio_resampled) < MIN_FRAME_SAMPLES:
            #print(f"[VAD] Frame too short ({len(audio_resampled)} samples), skipping.")
            return False

        # Normalize
        max_val = np.max(np.abs(audio_resampled))
        #print(f"[VAD] After float conversion, max: {max_val}")

        if max_val > 0:
            audio_resampled /= max_val
        else:
            #print("[VAD] Audio is silent (all zeros).")
            return False

        audio_tensor = torch.from_numpy(audio_resampled).float()
        timestamps = get_speech_timestamps(audio_tensor, self.model, sampling_rate=self.vad_sample_rate)
        #print(f"[VAD] Timestamps: {timestamps}")

        return len(timestamps) > 0

    def add_speech_segment(self, audio_chunk: np.ndarray):
        self.speech_segments.append(audio_chunk.copy())

    def save_segments(self, filename="speech_only.wav"):
        if not self.speech_segments:
            print("⚠️ No speech segments to save.")
            return

        audio_data = np.concatenate(self.speech_segments)
        audio_data = audio_data / np.max(np.abs(audio_data))  # Normalize
        audio_data = (audio_data * 32767).astype(np.int16)

        sf.write(filename, audio_data, self.sample_rate, subtype='PCM_16')
        print(f"✅ Saved normalized speech to {filename}")
