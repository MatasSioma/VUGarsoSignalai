"""Offscreen check that the SNR window shows the analysis results."""
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6 import QtWidgets

from app.core.signal import AudioSignal
from app.core.wavio import write_wav
from app.gui.main_window import MainWindow

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def tone_with_pause(sr=16000):
    """1 s of 400 Hz tone followed by 1 s of near-silence, stereo int16."""
    t = np.arange(sr) / sr
    x = np.concatenate([np.sin(2 * np.pi * 400 * t) * 10000, np.sin(2 * np.pi * 400 * t) * 10])
    return AudioSignal(np.stack([x, x / 2], axis=1).astype(np.int16), sr, 16, "pcm", "tone.wav")


class SnrWindowTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        path = Path(self.tmp.name) / "tone.wav"
        write_wav(path, tone_with_pause())
        self.main = MainWindow()
        self.main.load(path)  # same path as File -> Open
        self.main.open_snr()
        self.win = self.main.snr_window

    def tearDown(self):
        self.win.close()
        self.main.close()
        self.tmp.cleanup()

    def test_shows_overall_and_segmental_snr(self):
        self.win.weight_box.setValue(0.1)
        self.win.recompute()
        self.assertAlmostEqual(self.win.overall, 20.0, places=6)
        self.assertLess(self.win.segmental.mean, self.win.overall - 10)  # the quiet second half
        self.assertEqual(len(self.win.segmental.frame_snr), 100)  # 2 s / 20 ms
        text = self.win.result_label.text()
        self.assertIn("Overall SNR: <b>20.00 dB</b>", text)
        self.assertIn("100 frames × 20 ms", text)

    def test_controls_change_the_analysis(self):
        self.win.weight_box.setValue(1.0)
        self.win.frame_box.setValue(25)
        self.win.overlap_box.setChecked(True)
        self.win.recompute()
        self.assertAlmostEqual(self.win.overall, 0.0, places=6)
        self.assertEqual(len(self.win.segmental.frame_snr), 159)  # hop 200 samples, (32000 - 400) / 200 + 1
        self.win.limit_box.setChecked(True)
        self.win.recompute()
        self.assertGreaterEqual(np.nanmin(self.win.segmental.frame_snr), -10.0)

    def test_frame_length_restricted_to_20_30_ms(self):
        self.win.frame_box.setValue(5)
        self.assertEqual(self.win.frame_box.value(), 20)
        self.win.frame_box.setValue(50)
        self.assertEqual(self.win.frame_box.value(), 30)

    def test_selected_segment_only(self):
        self.main.start_box.setValue(0.0)
        self.main.dur_box.setValue(1.0)  # the loud tone only
        self.win.segment_box.setChecked(True)
        self.win.recompute()
        self.assertEqual(len(self.win.segmental.frame_snr), 50)
        self.assertAlmostEqual(self.win.segmental.mean, self.win.overall, delta=0.2)  # stationary part
        self.assertAlmostEqual(self.win.segmental.times[0], 0.01)

    def test_selection_follows_main_window(self):
        self.win.segment_box.setChecked(True)
        self.main.start_box.setValue(1.0)
        self.main.dur_box.setValue(0.5)
        self.win.recompute()
        self.assertEqual(len(self.win.segmental.frame_snr), 25)
        # real file time, not segment-relative; a 0.5 s part is shown in milliseconds
        self.assertEqual(self.win.p_frames.getAxis("bottom").labelText, "Time [ms]")
        np.testing.assert_allclose(self.win.c_frames.xData[0], 1010.0)

    def test_empty_selection_clears_results(self):
        self.win.segment_box.setChecked(True)
        self.main.dur_box.setValue(0.0)
        self.win.recompute()
        self.assertIsNone(self.win.overall)
        self.assertIn("undefined", self.win.result_label.text())

    def test_sweep_plot_covers_weights(self):
        self.win.recompute()
        k, overall = self.win.c_sweep_overall.xData, self.win.c_sweep_overall.yData
        self.assertGreater(len(k), 50)
        np.testing.assert_allclose(overall, -20 * np.log10(k), atol=1e-9)


if __name__ == "__main__":
    unittest.main()
