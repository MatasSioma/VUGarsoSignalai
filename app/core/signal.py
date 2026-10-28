from dataclasses import dataclass

import numpy as np


@dataclass
class AudioSignal:
    """Audio signal held in memory.

    samples: shape (n_samples, n_channels), native dtype
             (int16/int32 for PCM, 24-bit stored in int32 with real 24-bit values; float32/64 for IEEE float).
    """
    samples: np.ndarray
    sample_rate: int
    bits: int
    fmt: str  # 'pcm' or 'float'
    name: str = ""

    @property
    def n_samples(self) -> int:
        return self.samples.shape[0]

    @property
    def n_channels(self) -> int:
        return self.samples.shape[1]

    @property
    def duration(self) -> float:
        return self.n_samples / self.sample_rate

    @property
    def amplitude_range(self) -> tuple[float, float]:
        """Theoretical amplitude limits for the quantization depth."""
        if self.fmt == "float":
            return -1.0, 1.0
        if self.bits == 8:
            return 0, 255  # 8-bit WAV is unsigned
        return -(2 ** (self.bits - 1)), 2 ** (self.bits - 1) - 1

    def time_axis(self) -> np.ndarray:
        return np.arange(self.n_samples) / self.sample_rate


def to_float(sig: AudioSignal) -> np.ndarray:
    """Samples as float64 normalised to [-1, 1], shape (n_samples, n_channels)."""
    x = sig.samples.astype(np.float64)
    if sig.fmt == "float":
        return x
    if sig.bits == 8:
        return (x - 128) / 128
    return x / 2 ** (sig.bits - 1)


def from_float(x: np.ndarray, like: AudioSignal, name: str = "") -> AudioSignal:
    """Convert normalised float samples back to the bit depth and format of `like`, clipping to [-1, 1]."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    x = np.clip(x, -1.0, 1.0)
    if like.fmt == "float":
        samples = x.astype(like.samples.dtype)
    elif like.bits == 8:
        samples = np.clip(np.round(x * 128 + 128), 0, 255).astype(np.uint8)
    else:
        full = 2 ** (like.bits - 1)
        samples = np.clip(np.round(x * full), -full, full - 1).astype(np.int16 if like.bits <= 16 else np.int32)
    return AudioSignal(samples, like.sample_rate, like.bits, like.fmt, name or like.name)
