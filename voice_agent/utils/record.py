import sounddevice as sd
from scipy.io.wavfile import write
from scipy.io import wavfile

def record_audio(filename="input.wav", duration=5, fs=48000):
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1,dtype='int16')
    sd.wait()
    write(filename, fs, audio)
    #check sample rate and channels using scipy.io.wavfile
  
    sample_rate, data = wavfile.read(filename)
    print(f"Sample rate: {sample_rate}, Channels: {data.shape[1] if len(data.shape) > 1 else 1}")
    if sample_rate != fs:
        raise ValueError(f"Sample rate mismatch: expected {fs}, got {sample_rate}")

    print(f"Recording saved to {filename}")


# Example usage:
if __name__ == "__main__":
    record_audio("output.wav", duration=5, fs=48000)
    print("Audio recording complete.")