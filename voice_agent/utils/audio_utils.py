# agent/audio_utils.py
from scipy.signal import butter, lfilter
from transformers import AutoTokenizer
import os
import numpy as np
import asyncio
from scipy.io.wavfile import write

def noise_reduction(audio, cutoff=100.0, fs=48000, order=5):
    def butter_highpass(cutoff, fs, order):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    def highpass_filter(data, cutoff, fs, order):
        b, a = butter_highpass(cutoff, fs, order)
        return lfilter(b, a, data)

    return highpass_filter(audio, cutoff, fs, order)
