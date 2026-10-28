"""Audio lossy codec degradation and augmentation engine.

Implements realistic MP3 and Opus compression transforms using the bundled
static FFmpeg executable, simulating social messaging (e.g. WhatsApp, Telegram)
and low-bandwidth transmission.
"""

from typing import Union, Optional, Tuple
import os
import subprocess
import tempfile
import io
import numpy as np
import soundfile as sf
import imageio_ffmpeg


class AudioCodecSimulator:
    """Simulates lossy audio transmission using FFmpeg."""

    SUPPORTED_CONFIGS = {
        "mp3_64k": {"format": "mp3", "bitrate": "64k", "codec": "libmp3lame"},
        "mp3_32k": {"format": "mp3", "bitrate": "32k", "codec": "libmp3lame"},
        "opus_24k": {"format": "opus", "bitrate": "24k", "codec": "libopus"},
        "opus_16k": {"format": "opus", "bitrate": "16k", "codec": "libopus"},
    }

    def __init__(self, ffmpeg_path: Optional[str] = None):
        self.ffmpeg_path = ffmpeg_path or imageio_ffmpeg.get_ffmpeg_exe()
        if not os.path.exists(self.ffmpeg_path):
            raise FileNotFoundError(f"FFmpeg binary not found at: {self.ffmpeg_path}")

    def compress_decompress(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        codec_profile: str = "mp3_64k",
    ) -> np.ndarray:
        """Applies lossy compression and decodes back to linear PCM NumPy array.

        Args:
            audio: 1D float32 audio array.
            sample_rate: Sampling frequency (default 16000 Hz).
            codec_profile: One of ['mp3_64k', 'mp3_32k', 'opus_24k', 'opus_16k'].

        Returns:
            Compressed and decompressed float32 audio array of matching length.
        """
        if codec_profile not in self.SUPPORTED_CONFIGS:
            raise ValueError(
                f"Unknown codec profile: {codec_profile}. "
                f"Choose from: {list(self.SUPPORTED_CONFIGS.keys())}"
            )

        cfg = self.SUPPORTED_CONFIGS[codec_profile]
        orig_length = len(audio)

        with tempfile.TemporaryDirectory() as temp_dir:
            input_wav = os.path.join(temp_dir, "input.wav")
            compressed_file = os.path.join(temp_dir, f"compressed.{cfg['format']}")
            output_wav = os.path.join(temp_dir, "output.wav")

            # 1. Write clean audio to temporary WAV
            sf.write(input_wav, audio, sample_rate, subtype="PCM_16")

            # 2. Encode to lossy format with FFmpeg
            encode_cmd = [
                self.ffmpeg_path,
                "-y",
                "-loglevel", "error",
                "-i", input_wav,
                "-b:a", cfg["bitrate"],
                compressed_file,
            ]
            subprocess.run(encode_cmd, check=True)

            # 3. Decode back to linear PCM 16kHz WAV
            decode_cmd = [
                self.ffmpeg_path,
                "-y",
                "-loglevel", "error",
                "-i", compressed_file,
                "-ar", str(sample_rate),
                "-ac", "1",
                output_wav,
            ]
            subprocess.run(decode_cmd, check=True)

            # 4. Read decoded audio
            decoded_audio, _ = sf.read(output_wav, dtype="float32")

            # 5. Maintain exact original array length
            if len(decoded_audio) >= orig_length:
                return decoded_audio[:orig_length]
            else:
                return np.pad(
                    decoded_audio,
                    (0, orig_length - len(decoded_audio)),
                    mode="constant",
                    constant_values=0.0,
                )


class CodecAugmenter:
    """Stochastic on-the-fly codec data augmentation for training."""

    def __init__(
        self,
        aug_prob: float = 0.5,
        profiles: Optional[list] = None,
        sample_rate: int = 16000,
    ):
        self.aug_prob = aug_prob
        self.profiles = profiles or ["mp3_64k", "mp3_32k", "opus_24k", "opus_16k"]
        self.sample_rate = sample_rate
        self.simulator = AudioCodecSimulator()

    def __call__(self, audio: np.ndarray) -> Tuple[np.ndarray, str]:
        """Conditionally compresses audio with probability `aug_prob`."""
        if np.random.rand() < self.aug_prob:
            selected_profile = np.random.choice(self.profiles)
            try:
                augmented = self.simulator.compress_decompress(
                    audio, sample_rate=self.sample_rate, codec_profile=selected_profile
                )
                return augmented, selected_profile
            except Exception as e:
                # If compression fails, fall back to clean audio safely
                return audio, "clean_fallback"
        return audio, "clean"
