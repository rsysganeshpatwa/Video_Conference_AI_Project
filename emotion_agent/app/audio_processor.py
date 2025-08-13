import numpy as np
import librosa
import logging
from typing import Optional, Tuple
from pydub import AudioSegment
import io

logger = logging.getLogger(__name__)

class AudioProcessor:
    """
    Handles audio processing tasks including format conversion,
    resampling, and audio quality enhancement.
    """
    
    def __init__(self, target_sample_rate: int = 16000):
        """
        Initialize the audio processor.
        
        Args:
            target_sample_rate: Target sample rate for processing
        """
        self.target_sample_rate = target_sample_rate
        self.supported_formats = ['wav', 'mp3', 'webm', 'ogg', 'm4a']
    
    def convert_audio_format(self, audio_data: bytes, 
                           source_format: str = 'webm',
                           target_format: str = 'wav') -> Optional[bytes]:
        """
        Convert audio from one format to another.
        
        Args:
            audio_data: Raw audio bytes
            source_format: Source audio format
            target_format: Target audio format
            
        Returns:
            Converted audio bytes or None if conversion failed
        """
        try:
            # Load audio using pydub
            audio_segment = AudioSegment.from_file(
                io.BytesIO(audio_data),
                format=source_format
            )
            
            # Convert to target format
            output_buffer = io.BytesIO()
            audio_segment.export(output_buffer, format=target_format)
            
            return output_buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Error converting audio format: {e}")
            return None
    
    def resample_audio(self, audio_data: np.ndarray, 
                      source_rate: int, 
                      target_rate: Optional[int] = None) -> np.ndarray:
        """
        Resample audio to target sample rate.
        
        Args:
            audio_data: Audio data as numpy array
            source_rate: Source sample rate
            target_rate: Target sample rate (uses class default if None)
            
        Returns:
            Resampled audio data
        """
        target_rate = target_rate or self.target_sample_rate
        
        if source_rate == target_rate:
            return audio_data
        
        try:
            return librosa.resample(
                audio_data, 
                orig_sr=source_rate, 
                target_sr=target_rate
            )
        except Exception as e:
            logger.error(f"Error resampling audio: {e}")
            return audio_data
    
    def normalize_audio(self, audio_data: np.ndarray, 
                       target_peak: float = 0.95) -> np.ndarray:
        """
        Normalize audio to target peak level.
        
        Args:
            audio_data: Audio data as numpy array
            target_peak: Target peak level (0.0 to 1.0)
            
        Returns:
            Normalized audio data
        """
        try:
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                normalization_factor = target_peak / max_val
                return audio_data * normalization_factor
            return audio_data
        except Exception as e:
            logger.error(f"Error normalizing audio: {e}")
            return audio_data
    
    def remove_silence(self, audio_data: np.ndarray, 
                      sample_rate: int,
                      threshold: float = 0.01,
                      min_silence_duration: float = 0.5) -> np.ndarray:
        """
        Remove silence from audio data.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio
            threshold: Silence threshold (0.0 to 1.0)
            min_silence_duration: Minimum silence duration to remove (seconds)
            
        Returns:
            Audio data with silence removed
        """
        try:
            # Calculate frame size for silence detection
            frame_size = int(sample_rate * 0.01)  # 10ms frames
            hop_length = frame_size // 2
            
            # Calculate RMS energy for each frame
            rms = librosa.feature.rms(
                y=audio_data,
                frame_length=frame_size,
                hop_length=hop_length
            )[0]
            
            # Find non-silent frames
            non_silent_frames = rms > threshold
            
            # Convert frame indices to sample indices
            non_silent_samples = np.zeros(len(audio_data), dtype=bool)
            for i, is_non_silent in enumerate(non_silent_frames):
                start_sample = i * hop_length
                end_sample = min(start_sample + frame_size, len(audio_data))
                if is_non_silent:
                    non_silent_samples[start_sample:end_sample] = True
            
            # Remove short silent segments
            min_silence_samples = int(min_silence_duration * sample_rate)
            
            # Find silent segments
            silent_start = None
            for i, is_non_silent in enumerate(non_silent_samples):
                if not is_non_silent and silent_start is None:
                    silent_start = i
                elif is_non_silent and silent_start is not None:
                    silence_duration = i - silent_start
                    if silence_duration >= min_silence_samples:
                        # Keep this silence as it's long enough
                        pass
                    else:
                        # Remove short silence
                        non_silent_samples[silent_start:i] = True
                    silent_start = None
            
            return audio_data[non_silent_samples]
            
        except Exception as e:
            logger.error(f"Error removing silence: {e}")
            return audio_data
    
    def apply_noise_reduction(self, audio_data: np.ndarray, 
                            sample_rate: int) -> np.ndarray:
        """
        Apply basic noise reduction to audio.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio
            
        Returns:
            Audio data with noise reduction applied
        """
        try:
            # Apply high-pass filter to remove low-frequency noise
            from scipy.signal import butter, filtfilt
            
            # Design high-pass filter (80 Hz cutoff)
            nyquist = sample_rate / 2
            cutoff = 80 / nyquist
            b, a = butter(4, cutoff, btype='high')
            
            # Apply filter
            filtered_audio = filtfilt(b, a, audio_data)
            
            return filtered_audio
            
        except ImportError:
            logger.warning("scipy not available for noise reduction")
            return audio_data
        except Exception as e:
            logger.error(f"Error applying noise reduction: {e}")
            return audio_data
    
    def process_audio_chunk(self, audio_data: bytes, 
                          source_format: str = 'webm',
                          source_rate: Optional[int] = None,
                          apply_noise_reduction: bool = True,
                          normalize: bool = True) -> Tuple[np.ndarray, int]:
        """
        Process an audio chunk with all enhancements.
        
        Args:
            audio_data: Raw audio bytes
            source_format: Source audio format
            source_rate: Source sample rate (auto-detected if None)
            apply_noise_reduction: Whether to apply noise reduction
            normalize: Whether to normalize audio
            
        Returns:
            Tuple of (processed_audio_array, sample_rate)
        """
        try:
            # Convert format if needed
            if source_format != 'wav':
                audio_data = self.convert_audio_format(
                    audio_data, 
                    source_format, 
                    'wav'
                )
                if audio_data is None:
                    raise ValueError("Format conversion failed")
            
            # Load audio with librosa
            audio_array, detected_rate = librosa.load(
                io.BytesIO(audio_data),
                sr=source_rate
            )
            
            # Resample to target rate
            if detected_rate != self.target_sample_rate:
                audio_array = self.resample_audio(
                    audio_array, 
                    detected_rate, 
                    self.target_sample_rate
                )
            
            # Apply noise reduction
            if apply_noise_reduction:
                audio_array = self.apply_noise_reduction(
                    audio_array, 
                    self.target_sample_rate
                )
            
            # Normalize audio
            if normalize:
                audio_array = self.normalize_audio(audio_array)
            
            return audio_array, self.target_sample_rate
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            # Return empty array as fallback
            return np.array([]), self.target_sample_rate
    
    def is_audio_valid(self, audio_data: np.ndarray, 
                      min_duration: float = 0.1,
                      sample_rate: int = None) -> bool:
        """
        Check if audio data is valid for processing.
        
        Args:
            audio_data: Audio data as numpy array
            min_duration: Minimum duration in seconds
            sample_rate: Sample rate (uses class default if None)
            
        Returns:
            True if audio is valid, False otherwise
        """
        sample_rate = sample_rate or self.target_sample_rate
        
        if len(audio_data) == 0:
            return False
        
        duration = len(audio_data) / sample_rate
        if duration < min_duration:
            return False
        
        # Check if audio has any non-zero samples
        if np.max(np.abs(audio_data)) < 1e-6:
            return False
        
        return True
    
    def get_audio_features(self, audio_data: np.ndarray, 
                          sample_rate: int) -> dict:
        """
        Extract basic audio features for analysis.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio
            
        Returns:
            Dictionary of audio features
        """
        try:
            duration = len(audio_data) / sample_rate
            rms_energy = np.sqrt(np.mean(audio_data ** 2))
            peak_amplitude = np.max(np.abs(audio_data))
            
            # Zero crossing rate
            zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_data)))) / 2
            zero_crossing_rate = zero_crossings / len(audio_data)
            
            return {
                'duration': duration,
                'rms_energy': float(rms_energy),
                'peak_amplitude': float(peak_amplitude),
                'zero_crossing_rate': float(zero_crossing_rate),
                'sample_rate': sample_rate,
                'num_samples': len(audio_data)
            }
            
        except Exception as e:
            logger.error(f"Error extracting audio features: {e}")
            return {}
