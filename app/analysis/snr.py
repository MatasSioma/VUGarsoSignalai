from dataclasses import dataclass

import numpy as np

from ..core.signal import AudioSignal
from .framing import frames

FRAME_SNR_LIMITS = (-10.0, 35.0)


def channel_as_float(sig: AudioSignal, ch: int = 0) -> np.ndarray:
    x = sig.samples[:, ch].astype(np.float64)
    if sig.fmt == "pcm" and sig.bits == 8:
        x = x - 128.0  # 8 bitų WAV be ženklo: tyla yra 128, ne 0
    return x


def generate_noise(n: int, rms: float = 1.0, seed=None) -> np.ndarray:
    noise = np.random.default_rng(seed).standard_normal(n)
    if n == 0:
        return noise
    noise = noise - noise.mean()
    noise_rms = np.sqrt(np.mean(noise ** 2))
    if noise_rms == 0:
        return noise
    return noise * (rms / noise_rms)  #k = 1 reiškia 0 dB


def energy(x: np.ndarray) -> float:
    return float(np.sum(x ** 2))


def to_db(signal_energy, noise_energy):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10 * np.log10(np.divide(signal_energy, noise_energy))  # x/0 -> inf, 0/0 -> nan


def snr_db(signal: np.ndarray, noise: np.ndarray) -> float:
    return float(to_db(energy(signal), energy(noise)))


@dataclass
class SegmentalSnr:
    frame_snr: np.ndarray
    times: np.ndarray
    mean: float
    n_undefined: int


def _frame_snr(signal, noise, sample_rate, frame_ms, overlap):
    signal_frames, times = frames(signal, sample_rate, frame_ms, overlap)
    noise_frames, _ = frames(noise, sample_rate, frame_ms, overlap)
    signal_energy = np.sum(signal_frames ** 2, axis=1)  # kiekvieno kadro energija
    noise_energy = np.sum(noise_frames ** 2, axis=1)
    return to_db(signal_energy, noise_energy), times


def _summarise(frame_snr, times, limits) -> SegmentalSnr:
    if limits is not None:
        frame_snr = np.clip(frame_snr, limits[0], limits[1])  # ±inf tampa ribomis, nan lieka nan
    frame_snr = np.where(np.isinf(frame_snr), np.nan, frame_snr)
    valid = ~np.isnan(frame_snr)  # neapibrėžti kadrai neįskaičiuojami į vidurkį
    mean = float(np.mean(frame_snr[valid])) if valid.any() else float("nan")
    return SegmentalSnr(frame_snr, times, mean, int(np.sum(~valid)))


def segmental_snr(signal: np.ndarray, noise: np.ndarray, sample_rate: int, frame_ms: float = 20,
                  overlap: float = 0.0, limits=None) -> SegmentalSnr:
    frame_snr, times = _frame_snr(signal, noise, sample_rate, frame_ms, overlap)
    return _summarise(frame_snr, times, limits)


def snr_vs_weight(signal: np.ndarray, noise: np.ndarray, weights, sample_rate: int, frame_ms: float = 20,
                  overlap: float = 0.0, limits=None) -> tuple[np.ndarray, np.ndarray]:
    overall_at_1 = snr_db(signal, noise)
    frame_snr_at_1, times = _frame_snr(signal, noise, sample_rate, frame_ms, overlap)
    overall, segmental = [], []
    for k in weights:
        shift = 20 * np.log10(k)
        overall.append(overall_at_1 - shift)
        segmental.append(_summarise(frame_snr_at_1 - shift, times, limits).mean)
    return np.array(overall), np.array(segmental)
