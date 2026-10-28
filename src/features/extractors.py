"""Feature extraction engine for audio deepfake detection.

Provides:
1. Log-Mel Spectrogram extraction (80 mel bands).
2. Linear Frequency Cepstral Coefficients (LFCC) extraction with linear filterbanks.
"""

from typing import Tuple, Optional
import numpy as np
import scipy.signal
import scipy.fftpack


class FeatureExtractor:
    """Extracts 2D spectral representations (Log-Mel or LFCC) for neural and classical models."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 512,
        win_length: Optional[int] = 1024,
        n_mels: int = 80,
        n_lfcc: int = 80,
        f_min: float = 0.0,
        f_max: float = 8000.0,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.n_mels = n_mels
        self.n_lfcc = n_lfcc
        self.f_min = f_min
        self.f_max = f_max

        self.window = np.hamming(self.win_length).astype(np.float32)
        self.mel_basis = self._create_mel_filterbank()
        self.linear_basis = self._create_linear_filterbank()

    def _hz_to_mel(self, hz: float) -> float:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def _mel_to_hz(self, mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def _create_mel_filterbank(self) -> np.ndarray:
        """Constructs triangular Mel filterbank matrix."""
        mel_min = self._hz_to_mel(self.f_min)
        mel_max = self._hz_to_mel(self.f_max)
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = self._mel_to_hz(mel_points)
        bins = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)

        weights = np.zeros((self.n_mels, int(self.n_fft // 2 + 1)), dtype=np.float32)
        for i in range(1, self.n_mels + 1):
            left = bins[i - 1]
            center = bins[i]
            right = bins[i + 1]

            if center > left:
                weights[i - 1, left:center] = (np.arange(left, center) - left) / (center - left)
            if right > center:
                weights[i - 1, center:right] = (right - np.arange(center, right)) / (right - center)

        return weights

    def _create_linear_filterbank(self) -> np.ndarray:
        """Constructs triangular Linear Frequency filterbank matrix."""
        hz_points = np.linspace(self.f_min, self.f_max, self.n_lfcc + 2)
        bins = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)

        weights = np.zeros((self.n_lfcc, int(self.n_fft // 2 + 1)), dtype=np.float32)
        for i in range(1, self.n_lfcc + 1):
            left = bins[i - 1]
            center = bins[i]
            right = bins[i + 1]

            if center > left:
                weights[i - 1, left:center] = (np.arange(left, center) - left) / (center - left)
            if right > center:
                weights[i - 1, center:right] = (right - np.arange(center, right)) / (right - center)

        return weights

    def compute_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Computes Short-Time Fourier Transform (STFT) magnitude power spectrum."""
        f, t, zxx = scipy.signal.stft(
            audio,
            fs=self.sample_rate,
            window="hamming",
            nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length,
            nfft=self.n_fft,
            boundary=None,
            padded=False,
        )
        power_spec = np.abs(zxx) ** 2
        return power_spec.astype(np.float32)

    def extract_log_mel(self, audio: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Extracts Log-Mel Spectrogram. Shape: (n_mels, time_frames)."""
        power_spec = self.compute_spectrogram(audio)
        mel_spec = np.dot(self.mel_basis, power_spec)
        log_mel = np.log(mel_spec + eps)
        # Instance standardization (zero mean, unit variance)
        norm_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + eps)
        return norm_mel.astype(np.float32)

    def extract_lfcc(self, audio: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Extracts Linear Frequency Cepstral Coefficients (LFCC). Shape: (n_lfcc, time_frames)."""
        power_spec = self.compute_spectrogram(audio)
        linear_spec = np.dot(self.linear_basis, power_spec)
        log_linear = np.log(linear_spec + eps)
        # Discrete Cosine Transform Type-II
        lfcc = scipy.fftpack.dct(log_linear, type=2, axis=0, norm="ortho")
        norm_lfcc = (lfcc - np.mean(lfcc)) / (np.std(lfcc) + eps)
        return norm_lfcc.astype(np.float32)
