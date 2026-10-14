import numpy as np


def frame_length(sample_rate: int, frame_ms: float) -> int:
    """Number of samples in a frame of frame_ms milliseconds."""
    return max(1, int(round(sample_rate * frame_ms / 1000)))


def frames(x: np.ndarray, sample_rate: int, frame_ms: float = 20, overlap: float = 0.0):
    """Split a 1-D signal into frames of frame_ms (overlap is a fraction of the frame, 0 = none).

    Returns (frames of shape (n_frames, n), frame centre times in seconds).
    The incomplete tail frame is dropped. Frames are a view, no copy is made.
    """
    n = frame_length(sample_rate, frame_ms)
    hop = max(1, int(round(n * (1 - overlap))))
    if len(x) < n:
        return np.empty((0, n), dtype=x.dtype), np.empty(0)
    w = np.lib.stride_tricks.sliding_window_view(x, n)[::hop]
    t = (np.arange(len(w)) * hop + n / 2) / sample_rate
    return w, t
