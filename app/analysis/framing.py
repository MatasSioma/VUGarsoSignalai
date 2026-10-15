import numpy as np


def frame_length(sample_rate: int, frame_ms: float) -> int:
    return max(1, int(round(sample_rate * frame_ms / 1000)))


def frames(x: np.ndarray, sample_rate: int, frame_ms: float = 20, overlap: float = 0.0):
    n = frame_length(sample_rate, frame_ms)
    hop = max(1, int(round(n * (1 - overlap))))  # žingsnis tarp kadrų pradžių (persidengimas jį sumažina)
    starts = range(0, len(x) - n + 1, hop)  # nepilnas paskutinis kadras atmetamas
    w = np.array([x[s:s + n] for s in starts]).reshape(-1, n)
    t = np.array([(s + n / 2) / sample_rate for s in starts])
    return w, t
