import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile

def record_audio(duration, sample_rate, filename):
    """Records audio from the microphone and saves it to a WAV file.

    Args:
        duration (int): Recording duration in seconds.
        sample_rate (int): Sample rate of the audio.
        filename (str): Output file name.
    """
    print("Recording...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=2)
    sd.wait()
    print("Recording finished.")
    wavfile.write(filename, sample_rate, recording)

if __name__ == "__main__":
    duration = 5  # Recording duration in seconds
    sample_rate = 44100  # Sample rate
    filename = "output.wav"

    record_audio(duration, sample_rate, filename)
    print(f"Audio saved to {filename}")