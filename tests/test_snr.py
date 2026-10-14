"""Tests for UZD_02 task 6: segmental SNR analysis."""
import unittest
from pathlib import Path

import numpy as np

from app.analysis.framing import frame_length, frames
from app.analysis.snr import (FRAME_SNR_LIMITS, channel_as_float, energy, generate_noise, segmental_snr, snr_db,
                              snr_vs_weight)
from app.core.signal import AudioSignal
from app.core.wavio import read_wav

DATA = Path(__file__).resolve().parent.parent / "garsai#1"


def rms(x):
    return np.sqrt(np.mean(x ** 2))


class FramingTest(unittest.TestCase):
    def test_frame_length_20_and_30_ms(self):
        self.assertEqual(frame_length(16000, 20), 320)
        self.assertEqual(frame_length(44100, 30), 1323)
        self.assertEqual(frame_length(22050, 20), 441)

    def test_non_overlapping_frames_drop_the_tail(self):
        x = np.arange(1000.0)
        w, t = frames(x, 1000, frame_ms=20)  # 20 samples per frame
        self.assertEqual(w.shape, (50, 20))
        w, t = frames(x[:1010], 1000, frame_ms=30)  # 1010 // 30 = 33 frames, tail of 20 dropped
        self.assertEqual(w.shape, (33, 30))
        np.testing.assert_array_equal(w[1], x[30:60])
        np.testing.assert_allclose(t[:2], [0.015, 0.045])  # frame centres, s

    def test_half_overlap(self):
        w, t = frames(np.arange(100.0), 1000, frame_ms=20, overlap=0.5)  # hop 10
        self.assertEqual(w.shape, (9, 20))
        np.testing.assert_array_equal(w[1], np.arange(10.0, 30.0))

    def test_signal_shorter_than_a_frame(self):
        w, t = frames(np.zeros(5), 1000, frame_ms=20)
        self.assertEqual(w.shape, (0, 20))
        self.assertEqual(len(t), 0)


class NoiseTest(unittest.TestCase):
    def test_noise_is_random_zero_mean_with_requested_rms(self):
        n = generate_noise(48000, rms=1234.5, seed=1)
        self.assertEqual(len(n), 48000)
        self.assertAlmostEqual(n.mean(), 0.0, places=6)
        self.assertAlmostEqual(rms(n), 1234.5, places=6)
        # white: neighbouring samples are uncorrelated
        self.assertLess(abs(np.corrcoef(n[:-1], n[1:])[0, 1]), 0.02)

    def test_seed_makes_noise_reproducible(self):
        np.testing.assert_array_equal(generate_noise(1000, seed=7), generate_noise(1000, seed=7))
        self.assertFalse(np.array_equal(generate_noise(1000, seed=7), generate_noise(1000, seed=8)))


class OverallSnrTest(unittest.TestCase):
    def test_formula_10_log10_energy_ratio(self):
        s = np.full(100, 10.0)  # E = 10000
        e = np.full(100, 1.0)   # E = 100
        self.assertAlmostEqual(energy(s), 10000.0)
        self.assertAlmostEqual(snr_db(s, e), 20.0)
        self.assertAlmostEqual(snr_db(e, s), -20.0)
        self.assertAlmostEqual(snr_db(s, s), 0.0)

    def test_snr_depends_on_noise_weight_as_minus_20_log10_k(self):
        x = np.sin(2 * np.pi * 440 * np.arange(22050) / 22050) * 8000
        n = generate_noise(len(x), rms(x), seed=0)  # noise with the signal's RMS
        for k, expected in ((0.01, 40.0), (0.1, 20.0), (0.5, 6.0206), (1.0, 0.0), (2.0, -6.0206), (10.0, -20.0)):
            self.assertAlmostEqual(snr_db(x, k * n), expected, places=3, msg=f"k={k}")
        # doubling the weight lowers the SNR by 6.02 dB
        self.assertAlmostEqual(snr_db(x, 0.2 * n) - snr_db(x, 0.4 * n), 20 * np.log10(2), places=9)

    def test_no_noise_gives_infinite_snr(self):
        self.assertEqual(snr_db(np.ones(10), np.zeros(10)), float("inf"))


class SegmentalSnrTest(unittest.TestCase):
    def setUp(self):
        # 4 frames of 20 samples (20 ms at 1 kHz): two loud frames (±10), two quiet ones (±1).
        # Noise is ±1 with the same energy in every frame.
        alt = np.tile([1.0, -1.0], 40)
        self.signal = alt * np.repeat([10.0, 10.0, 1.0, 1.0], 20)
        self.noise = alt.copy()

    def test_frame_values_and_their_mean(self):
        r = segmental_snr(self.signal, self.noise, 1000, frame_ms=20)
        np.testing.assert_allclose(r.frame_snr, [20.0, 20.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(r.times, [0.01, 0.03, 0.05, 0.07])
        self.assertAlmostEqual(r.mean, 10.0)
        self.assertEqual(r.n_undefined, 0)

    def test_overall_and_segmental_snr_differ(self):
        overall = snr_db(self.signal, self.noise)  # 10 log10(4040 / 80)
        self.assertAlmostEqual(overall, 10 * np.log10(50.5))
        self.assertAlmostEqual(overall, 17.0329, places=4)
        seg = segmental_snr(self.signal, self.noise, 1000, frame_ms=20).mean
        self.assertAlmostEqual(overall - seg, 7.0329, places=4)

    def test_overlapping_frames(self):
        r = segmental_snr(self.signal, self.noise, 1000, frame_ms=20, overlap=0.5)
        # hop 10: frames over loud/loud, loud/loud, loud/loud, loud/quiet, quiet/quiet, quiet/quiet, quiet/quiet
        mixed = 10 * np.log10((10 * 100 + 10 * 1) / 20)
        np.testing.assert_allclose(r.frame_snr, [20, 20, 20, mixed, 0, 0, 0], atol=1e-12)

    def test_stationary_signal_segmental_equals_overall(self):
        sr = 16000
        x = np.sin(2 * np.pi * 400 * np.arange(sr * 2) / sr) * 1000  # 400 Hz: whole periods in 20 ms
        n = generate_noise(len(x), rms(x), seed=3)
        for k in (0.1, 1.0):
            overall = snr_db(x, k * n)
            seg = segmental_snr(x, k * n, sr, frame_ms=20).mean
            self.assertAlmostEqual(seg, overall, delta=0.1, msg=f"k={k}")

    def test_silent_frames_are_left_out_of_raw_mean(self):
        s = np.concatenate([self.signal[:40], np.zeros(40)])  # last two frames digital silence -> -inf dB
        r = segmental_snr(s, self.noise, 1000, frame_ms=20)
        self.assertTrue(np.isnan(r.frame_snr[2:]).all())
        self.assertEqual(r.n_undefined, 2)
        self.assertAlmostEqual(r.mean, 20.0)

    def test_limits_clip_each_frame_before_averaging(self):
        s = np.concatenate([self.signal[:40] * 100, np.zeros(40)])  # frames at 60 dB, 60 dB, -inf, -inf
        r = segmental_snr(s, self.noise, 1000, frame_ms=20, limits=FRAME_SNR_LIMITS)
        np.testing.assert_allclose(r.frame_snr, [35.0, 35.0, -10.0, -10.0])
        self.assertAlmostEqual(r.mean, 12.5)
        self.assertEqual(r.n_undefined, 0)

    def test_sweep_matches_direct_computation(self):
        rng = np.random.default_rng(5)
        x = rng.standard_normal(8000) * np.repeat(rng.uniform(0.01, 3, 20), 400)  # level varies in time
        n = generate_noise(len(x), rms(x), seed=6)
        weights = [0.003, 0.05, 1.0, 7.0]
        for limits in (None, FRAME_SNR_LIMITS):
            overall, seg = snr_vs_weight(x, n, weights, 8000, frame_ms=25, overlap=0.5, limits=limits)
            for i, k in enumerate(weights):
                self.assertAlmostEqual(overall[i], snr_db(x, k * n), places=9)
                self.assertAlmostEqual(seg[i], segmental_snr(x, k * n, 8000, 25, 0.5, limits).mean, places=9)


class ChannelTest(unittest.TestCase):
    def test_eight_bit_is_centred(self):
        sig = AudioSignal(np.array([[128], [255], [0]], dtype=np.uint8), 8000, 8, "pcm")
        np.testing.assert_array_equal(channel_as_float(sig), [0.0, 127.0, -128.0])

    def test_picks_channel(self):
        sig = AudioSignal(np.array([[1, -5], [2, -6]], dtype=np.int16), 8000, 16, "pcm")
        np.testing.assert_array_equal(channel_as_float(sig, 1), [-5.0, -6.0])


@unittest.skipUnless(DATA.is_dir(), "test recordings not present")
class RecordingsTest(unittest.TestCase):
    """Expected outputs on the course recordings, noise weight k = 0.1 (noise RMS = 10 % of signal RMS)."""

    def analyse(self, name, ch=0, k=0.1, frame_ms=20, limits=None):
        sig = read_wav(DATA / name)
        x = channel_as_float(sig, ch)
        n = k * generate_noise(len(x), rms(x), seed=0)
        return snr_db(x, n), segmental_snr(x, n, sig.sample_rate, frame_ms, limits=limits)

    def test_sine_segmental_equals_overall(self):
        for name in ("Sinus_125Hz [mono, 22,05 kHz, 16-bit].wav", "Sinus_500Hz [mono, 22,05 kHz, 16-bit].wav"):
            overall, seg = self.analyse(name)
            self.assertAlmostEqual(overall, 20.0, places=6)
            self.assertAlmostEqual(seg.mean, 20.0, delta=0.1, msg=name)

    def test_tone_bursts_segmental_much_lower(self):
        # DTMF and Pure_tones contain pauses: quiet frames have very low SNR and pull the mean down
        for name in ("DTMF_tones [mono, 16 kHz, 16-bit].wav", "Pure_tones [mono, 44,1 kHz, 16-bit].wav"):
            overall, seg = self.analyse(name)
            self.assertAlmostEqual(overall, 20.0, places=6)
            self.assertLess(seg.mean, 0.0, msg=name)
            self.assertLess(np.nanmin(seg.frame_snr), -20.0)
            self.assertGreater(np.nanmax(seg.frame_snr), 20.0)

    def test_music_segmental_lower_than_overall(self):
        overall, seg = self.analyse("Music#1 [stereo, 48 kHz, 24-bit].wav")
        self.assertAlmostEqual(overall, 20.0, places=6)
        self.assertLess(seg.mean, overall - 1.0)
        self.assertEqual(len(seg.frame_snr), 1500)  # 30 s in 20 ms frames

    def test_frame_length_between_20_and_30_ms(self):
        _, s20 = self.analyse("Music#1 [stereo, 48 kHz, 24-bit].wav", frame_ms=20)
        _, s30 = self.analyse("Music#1 [stereo, 48 kHz, 24-bit].wav", frame_ms=30)
        self.assertEqual(len(s30.frame_snr), 1000)
        self.assertAlmostEqual(s20.mean, s30.mean, delta=1.0)

    def test_speech_with_silence_needs_limits(self):
        name = "Speech#1 [5.1, 48 kHz, 24-bit].wav"
        overall, raw = self.analyse(name, ch=2)
        self.assertGreater(raw.n_undefined, 0)  # digital silence: SNR of those frames is -inf
        _, lim = self.analyse(name, ch=2, limits=FRAME_SNR_LIMITS)
        self.assertEqual(lim.n_undefined, 0)
        self.assertTrue(((lim.frame_snr >= -10) & (lim.frame_snr <= 35)).all())
        self.assertLess(lim.mean, overall)


if __name__ == "__main__":
    unittest.main()
