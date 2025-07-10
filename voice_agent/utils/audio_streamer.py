import wave
import numpy as np
from livekit import rtc

SAMPLE_RATE = 48000
NUM_CHANNELS = 1
SAMPLES_PER_FRAME = 960  # 10ms of audio at 48kHz

async def stream_audio_file(file_path: str, source: rtc.AudioSource):
    print(f"📡 Streaming audio from {file_path}...")
    with wave.open(file_path, 'rb') as wf:
        
        assert wf.getframerate() == SAMPLE_RATE, "Audio must be 48kHz"
        assert wf.getnchannels() == NUM_CHANNELS, "Audio must be mono"
        assert wf.getsampwidth() == 2, "Audio must be 16-bit PCM"

        frame = rtc.AudioFrame.create(SAMPLE_RATE, NUM_CHANNELS, SAMPLES_PER_FRAME)
        buffer = np.frombuffer(frame.data, dtype=np.int16)

        while True:
            data = wf.readframes(SAMPLES_PER_FRAME)
            if not data:
                break

            chunk = np.frombuffer(data, dtype=np.int16)

            if chunk.shape[0] < SAMPLES_PER_FRAME:
                # Pad with zeros if last chunk is shorter
                chunk = np.pad(chunk, (0, SAMPLES_PER_FRAME - chunk.shape[0]))

            np.copyto(buffer, chunk)
            await source.capture_frame(frame)

                