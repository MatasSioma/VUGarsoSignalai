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

    def segment(self, start_s: float, duration_s: float) -> "AudioSignal":
        a = max(0, int(round(start_s * self.sample_rate)))
        b = min(self.n_samples, a + int(round(duration_s * self.sample_rate)))
        return AudioSignal(self.samples[a:b].copy(), self.sample_rate, self.bits, self.fmt, self.name)
