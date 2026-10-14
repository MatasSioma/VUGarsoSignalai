"""Segmental signal-to-noise ratio analysis (UZD_02, task 6).

SNR_dB = 10 log10(E_sig / E_tr), where E_sig is the energy of the useful signal and
E_tr the energy of the noise. The noise is generated white noise added with a weight k:
noisy = signal + k * noise.
"""
from dataclasses import dataclass

import numpy as np

from ..core.signal import AudioSignal
from .framing import frames

# Usual per-frame limits for segmental SNR (silent frames would otherwise dominate the mean)
FRAME_SNR_LIMITS = (-10.0, 35.0)


def channel_as_float(sig: AudioSignal, ch: int = 0) -> np.ndarray:
    """One channel as float64 with real sample values centred on zero."""
    x = sig.samples[:, ch].astype(np.float64)
    if sig.fmt == "pcm" and sig.bits == 8:
        x -= 128.0  # 8-bit WAV is unsigned, 128 = silence
    return x


def generate_noise(n: int, rms: float = 1.0, seed=None) -> np.ndarray:
    """Zero-mean white Gaussian noise of n samples, scaled to exactly the given RMS.

    Scaling the noise to the signal's RMS makes the weight k a noise/signal amplitude
    ratio, so k = 1 means equal energies (0 dB).
    """
    g = np.random.default_rng(seed).standard_normal(n)
    if n == 0:
        return g
    g -= g.mean()
    g_rms = np.sqrt(np.mean(g ** 2))
    return g * (rms / g_rms) if g_rms > 0 else g


def energy(x: np.ndarray) -> float:
    return float(np.dot(x, x))


def snr_db(signal: np.ndarray, noise: np.ndarray) -> float:
    """SNR of the whole signal: 10 log10(E_sig / E_tr)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(10 * np.log10(np.float64(energy(signal)) / energy(noise)))


@dataclass
class SegmentalSnr:
    frame_snr: np.ndarray  # SNR of each frame in dB, nan where undefined (0/0)
    times: np.ndarray      # frame centre times in seconds
    mean: float            # segmental SNR: mean of the defined frame values, dB
    n_undefined: int       # frames left out of the mean (zero or infinite energy ratio)


def _frame_energies(signal, noise, sample_rate, frame_ms, overlap):
    s, t = frames(signal, sample_rate, frame_ms, overlap)
    e, _ = frames(noise, sample_rate, frame_ms, overlap)
    return np.einsum("ij,ij->i", s, s), np.einsum("ij,ij->i", e, e), t


def _summarise(frame_snr: np.ndarray, times: np.ndarray, limits) -> SegmentalSnr:
    if limits is not None:
        frame_snr = np.clip(frame_snr, *limits)  # ±inf become the limits, nan stays nan
    frame_snr = np.where(np.isfinite(frame_snr), frame_snr, np.nan)
    valid = ~np.isnan(frame_snr)
    mean = float(frame_snr[valid].mean()) if valid.any() else float("nan")
    return SegmentalSnr(frame_snr, times, mean, int((~valid).sum()))


def segmental_snr(signal: np.ndarray, noise: np.ndarray, sample_rate: int, frame_ms: float = 20,
                  overlap: float = 0.0, limits=None) -> SegmentalSnr:
    """Split signal and noise into frames, compute the SNR of each frame and average them.

    limits=(lo, hi) clips each frame's SNR before averaging; None uses the raw values and
    leaves frames where the SNR is not finite (silence in signal or noise) out of the mean.
    """
    es, en, t = _frame_energies(signal, noise, sample_rate, frame_ms, overlap)
    with np.errstate(divide="ignore", invalid="ignore"):
        frame_snr = 10 * np.log10(es / en)
    return _summarise(frame_snr, t, limits)


def snr_vs_weight(signal: np.ndarray, noise: np.ndarray, weights, sample_rate: int, frame_ms: float = 20,
                  overlap: float = 0.0, limits=None) -> tuple[np.ndarray, np.ndarray]:
    """Overall and segmental SNR for noisy = signal + k * noise, for every k in weights.

    Scaling the noise by k multiplies its energy by k², so each SNR moves by -20 log10(k);
    the frame energies are therefore computed once.
    """
    weights = np.asarray(weights, dtype=np.float64)
    shift = 20 * np.log10(weights)
    overall = snr_db(signal, noise) - shift
    es, en, t = _frame_energies(signal, noise, sample_rate, frame_ms, overlap)
    with np.errstate(divide="ignore", invalid="ignore"):
        base = 10 * np.log10(es / en)
    segmental = np.array([_summarise(base - d, t, limits).mean for d in shift])
    return overall, segmental
