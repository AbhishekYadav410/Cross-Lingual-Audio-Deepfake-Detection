"""Audio loading and standardization pipeline for deepfake detection.

Handles mono conversion, deterministic resampling to 16 kHz, fixed-duration
cropping/padding, and amplitude normalization.
"""

from typing import Tuple, Union, Optional
import io
import numpy as np
import soundfile as sf
import scipy.signal


class AudioStandardizer:
    """Standardizes arbitrary input audio into a fixed 16kHz mono tensor/array."""

    def __init__(
        self,
        target_sample_rate: int = 16000,
        target_duration_sec: float = 3.0,
        normalize_audio: bool = True,
    ):
        self.target_sample_rate = target_sample_rate
        self.target_duration_sec = target_duration_sec
        self.target_num_samples = int(target_sample_rate * target_duration_sec)
        self.normalize_audio = normalize_audio

    def load_from_bytes(self, audio_bytes: bytes) -> Tuple[np.ndarray, int]:
        """Loads audio from raw byte buffer."""
        buffer = io.BytesIO(audio_bytes)
        audio, sr = sf.read(buffer, dtype="float32")
        return audio, sr

    def load_from_file(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Loads audio from disk."""
        audio, sr = sf.read(file_path, dtype="float32")
        return audio, sr

    def to_mono(self, audio: np.ndarray) -> np.ndarray:
        """Converts multi-channel audio to single-channel mono by averaging."""
        if audio.ndim == 1:
            return audio
        elif audio.ndim == 2:
            # Shape can be (channels, samples) or (samples, channels)
            if audio.shape[0] < audio.shape[1]:
                return np.mean(audio, axis=0)
            else:
                return np.mean(audio, axis=1)
        else:
            raise ValueError(f"Unsupported audio array dimension: {audio.ndim}")

    def resample(self, audio: np.ndarray, orig_sr: int) -> np.ndarray:
        """Resamples audio to target sample rate using polyphase filtering."""
        if orig_sr == self.target_sample_rate:
            return audio
        num_target_samples = int(round(len(audio) * float(self.target_sample_rate) / orig_sr))
        resampled = scipy.signal.resample(audio, num_target_samples)
        return resampled.astype(np.float32)

    def standardize_duration(self, audio: np.ndarray) -> np.ndarray:
        """Enforces exact fixed duration (target_num_samples).

        If audio is longer, applies center cropping.
        If audio is shorter, symmetrically pads with silence (zeros).
        """
        num_samples = len(audio)
        if num_samples == self.target_num_samples:
            return audio

        if num_samples > self.target_num_samples:
            # Center crop
            start = (num_samples - self.target_num_samples) // 2
            return audio[start : start + self.target_num_samples]
        else:
            # Symmetric zero pad
            pad_total = self.target_num_samples - num_samples
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            return np.pad(audio, (pad_left, pad_right), mode="constant", constant_values=0.0)

    def peak_normalize(self, audio: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Normalizes audio waveform to peak absolute amplitude of 0.95."""
        peak = np.max(np.abs(audio))
        if peak > eps:
            return (audio / peak) * 0.95
        return audio

    def process(self, audio_input: Union[str, bytes, np.ndarray], orig_sr: Optional[int] = None) -> np.ndarray:
        """Executes full standardization: load -> mono -> resample -> crop/pad -> normalize."""
        if isinstance(audio_input, str):
            audio, sr = self.load_from_file(audio_input)
        elif isinstance(audio_input, (bytes, bytearray)):
            audio, sr = self.load_from_bytes(audio_input)
        elif isinstance(audio_input, np.ndarray):
            if orig_sr is None:
                raise ValueError("orig_sr must be provided when audio_input is a NumPy array.")
            audio, sr = audio_input, orig_sr
        else:
            raise TypeError(f"Unsupported audio_input type: {type(audio_input)}")

        mono_audio = self.to_mono(audio)
        resampled_audio = self.resample(mono_audio, sr)
        fixed_audio = self.standardize_duration(resampled_audio)
        if self.normalize_audio:
            fixed_audio = self.peak_normalize(fixed_audio)

        return fixed_audio.astype(np.float32)


def load_and_standardize_audio(
    file_or_bytes: Union[str, bytes, np.ndarray],
    orig_sr: Optional[int] = None,
    target_sr: int = 16000,
    duration: float = 3.0,
) -> np.ndarray:
    """Convenience function for single-call audio standardization."""
    standardizer = AudioStandardizer(target_sample_rate=target_sr, target_duration_sec=duration)
    return standardizer.process(file_or_bytes, orig_sr=orig_sr)
