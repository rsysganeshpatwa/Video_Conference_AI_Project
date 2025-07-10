import numpy as np
from silero_vad import get_speech_timestamps, load_silero_vad

class VADBuffer:
    def __init__(self, sample_rate=48000, min_duration_sec=0.5):
        self.sample_rate = sample_rate
        self.min_samples = int(sample_rate * min_duration_sec)
        self.buffer = []
        self.model = load_silero_vad()

    def add_audio(self, pcm_float32: np.ndarray):
        """Append incoming float32 PCM samples to the buffer."""
        self.buffer.extend(pcm_float32)
        print(f"📏 Buffer size: {len(self.buffer)} samples")

    def is_ready(self):
        """Check if buffer has enough data to run VAD."""
        return len(self.buffer) >= self.min_samples

    def is_speech(self, clear_after=True) -> bool:
        """Run VAD if ready. Returns True if speech detected."""
        if not self.is_ready():
            print(f"⏳ Waiting: {len(self.buffer)} samples (< {self.min_samples})")
            return False

        try:
            audio_np = np.array(self.buffer, dtype=np.float32)
            speech_timestamps = get_speech_timestamps(
                audio_np,
                self.model,
                sampling_rate=self.sample_rate
            )
            if speech_timestamps:
                print("✅ Human speech detected.")
                result = True
            else:
                print("🔇 No human speech detected.")
                result = False
        except Exception as e:
            print(f"❌ VAD error: {e}")
            result = False

        if clear_after:
            self.buffer = []

        return result

    def clear(self):
        """Manually clear buffer."""
        self.buffer = []
