import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..analysis.snr import (FRAME_SNR_LIMITS, channel_as_float, generate_noise, segmental_snr, snr_db,
                            snr_vs_weight)
from ..core.signal import AudioSignal
from ..core.timefmt import choose_time_unit
from .signal_plot import CHANNEL_NAMES

SWEEP_WEIGHTS = np.logspace(-3, 2, 101)


class SnrWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent, QtCore.Qt.Window)
        self.setWindowTitle("Segmentinė SST analizė")
        self.resize(1200, 950)
        self.signal: AudioSignal | None = None
        self.selection = (0.0, 0.0)
        self._noise_key = None
        self._unit_noise = None
        self.overall: float | None = None
        self.segmental = None

        self.channel_box = QtWidgets.QComboBox()
        self.weight_box = QtWidgets.QDoubleSpinBox(decimals=3, minimum=0.001, maximum=100, value=0.1, singleStep=0.01)
        self.frame_box = QtWidgets.QSpinBox(minimum=20, maximum=30, value=20, suffix=" ms")
        self.overlap_box = QtWidgets.QCheckBox("50 % overlap")
        self.limit_box = QtWidgets.QCheckBox(f"Limit frame SNR to [{FRAME_SNR_LIMITS[0]:g}, {FRAME_SNR_LIMITS[1]:g}] dB")
        self.seed_box = QtWidgets.QSpinBox(minimum=0, maximum=2 ** 31 - 1, value=0)
        self.segment_box = QtWidgets.QCheckBox("Selected segment only")

        self._timer = QtCore.QTimer(self, singleShot=True, interval=100)
        self._timer.timeout.connect(self.recompute)
        for sig in (self.channel_box.currentIndexChanged, self.weight_box.valueChanged, self.frame_box.valueChanged,
                    self.overlap_box.toggled, self.limit_box.toggled, self.seed_box.valueChanged,
                    self.segment_box.toggled):
            sig.connect(self._timer.start)

        bar = QtWidgets.QHBoxLayout()
        for label, w in (("Channel:", self.channel_box), ("Noise weight k:", self.weight_box),
                         ("Frame:", self.frame_box), (None, self.overlap_box), (None, self.limit_box),
                         ("Noise seed:", self.seed_box), (None, self.segment_box)):
            if label:
                bar.addWidget(QtWidgets.QLabel(label))
            bar.addWidget(w)
            bar.addSpacing(8)
        bar.addStretch()

        self.result_label = QtWidgets.QLabel("")
        self.result_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.result_label.setStyleSheet("font-size: 11pt;")

        self.layout_widget = pg.GraphicsLayoutWidget()
        self._build_plots()

        v = QtWidgets.QVBoxLayout(self)
        v.addLayout(bar)
        v.addWidget(self.result_label)
        v.addWidget(self.layout_widget)

    def _build_plots(self):
        lw = self.layout_widget
        self.title = lw.addLabel("", row=0, col=0, size="11pt", bold=True)
        self.p_clean = lw.addPlot(row=1, col=0)
        self.p_noise = lw.addPlot(row=2, col=0)
        self.p_noisy = lw.addPlot(row=3, col=0)
        self.p_frames = lw.addPlot(row=4, col=0)
        self.p_sweep = lw.addPlot(row=5, col=0)

        for p, label in ((self.p_clean, "Signal"), (self.p_noise, "k · noise"), (self.p_noisy, "Signal + k · noise")):
            p.setLabel("left", label)
            p.setClipToView(True)
            p.setDownsampling(auto=True, mode="peak")
            if p is not self.p_clean:
                p.setXLink(self.p_clean)
                p.setYLink(self.p_clean)
        self.c_clean = self.p_clean.plot(pen=pg.mkPen((30, 110, 200), width=1))
        self.c_noise = self.p_noise.plot(pen=pg.mkPen((200, 80, 40), width=1))
        self.c_noisy = self.p_noisy.plot(pen=pg.mkPen((90, 90, 90), width=1))

        self.p_frames.setXLink(self.p_clean)
        self.p_frames.setLabel("left", "Frame SNR [dB]")
        self.p_frames.addLegend(offset=(10, 5))
        self.c_frames = self.p_frames.plot(pen=pg.mkPen((30, 140, 60), width=1), name="frame SNR")
        self.l_overall = pg.InfiniteLine(angle=0, pen=pg.mkPen((200, 40, 40), width=2, style=QtCore.Qt.DashLine))
        self.l_segmental = pg.InfiniteLine(angle=0, pen=pg.mkPen((30, 60, 200), width=2, style=QtCore.Qt.DashLine))
        for line, name in ((self.l_overall, "overall SNR"), (self.l_segmental, "segmental SNR")):
            self.p_frames.addItem(line)
            self.p_frames.legend.addItem(pg.PlotDataItem(pen=line.pen), name)

        self.p_sweep.setLogMode(x=True)
        self.p_sweep.setLabel("left", "SNR [dB]")
        self.p_sweep.getAxis("bottom").setLabel("Noise weight k (log scale)")
        self.p_sweep.addLegend(offset=(-10, 5))
        self.c_sweep_overall = self.p_sweep.plot(pen=pg.mkPen((200, 40, 40), width=2), name="overall SNR")
        self.c_sweep_segmental = self.p_sweep.plot(pen=pg.mkPen((30, 60, 200), width=2), name="segmental SNR")
        self.l_weight = pg.InfiniteLine(angle=90, pen=pg.mkPen((120, 120, 120), style=QtCore.Qt.DotLine))
        self.p_sweep.addItem(self.l_weight)

        for p in (self.p_clean, self.p_noise, self.p_noisy, self.p_frames, self.p_sweep):
            p.showGrid(x=True, y=True, alpha=0.3)
            p.setMinimumHeight(140)

    def set_signal(self, sig: AudioSignal):
        self.signal = sig
        self._noise_key = None
        names = CHANNEL_NAMES.get(sig.n_channels, [f"Ch {i + 1}" for i in range(sig.n_channels)])
        self.channel_box.blockSignals(True)
        self.channel_box.clear()
        self.channel_box.addItems(names)
        self.channel_box.blockSignals(False)
        self.selection = (0.0, sig.duration)
        self.recompute()

    def set_selection(self, start_s: float, end_s: float):
        self.selection = (start_s, end_s)
        if self.segment_box.isChecked():
            self._timer.start()

    def _analysed(self) -> tuple[np.ndarray, float]:
        ch = max(0, self.channel_box.currentIndex())
        if self.segment_box.isChecked():
            start, end = self.selection
            seg = self.signal.segment(start, end - start)
            return channel_as_float(seg, ch), round(start * self.signal.sample_rate) / self.signal.sample_rate
        return channel_as_float(self.signal, ch), 0.0

    def _noise(self, x: np.ndarray) -> np.ndarray:
        key = (len(x), self.seed_box.value(), self.channel_box.currentIndex(), self.segment_box.isChecked(),
               self.selection if self.segment_box.isChecked() else None)
        if key != self._noise_key:
            rms = np.sqrt(np.mean(x ** 2)) if len(x) else 0.0
            self._unit_noise = generate_noise(len(x), rms, self.seed_box.value())
            self._noise_key = key
        return self._unit_noise

    def recompute(self):
        if self.signal is None:
            return
        sig = self.signal
        x, t0 = self._analysed()
        sr = sig.sample_rate
        k = self.weight_box.value()
        frame_ms = self.frame_box.value()
        overlap = 0.5 if self.overlap_box.isChecked() else 0.0
        limits = FRAME_SNR_LIMITS if self.limit_box.isChecked() else None

        self.title.setText(f"{sig.name}  |  {self.channel_box.currentText()}, {sr} Hz, {len(x) / sr:.3f} s analysed")
        if len(x) == 0 or not np.any(x):
            self.result_label.setText("The analysed part is empty or silent: SNR is undefined.")
            self.overall, self.segmental = None, None
            for c in (self.c_clean, self.c_noise, self.c_noisy, self.c_frames,
                      self.c_sweep_overall, self.c_sweep_segmental):
                c.setData([], [])
            return

        noise = self._noise(x)
        kn = k * noise
        overall = snr_db(x, kn)
        seg = segmental_snr(x, kn, sr, frame_ms, overlap, limits)

        scale, unit = choose_time_unit(len(x) / sr)
        t = (np.arange(len(x)) / sr + t0) * scale
        self.c_clean.setData(t, x)
        self.c_noise.setData(t, kn)
        self.c_noisy.setData(t, x + kn)
        self.c_frames.setData((seg.times + t0) * scale, seg.frame_snr, connect="finite")
        self.l_overall.setValue(overall)
        self.l_segmental.setValue(seg.mean)
        for p in (self.p_clean, self.p_noise, self.p_noisy, self.p_frames):
            p.getAxis("bottom").setLabel(f"Time [{unit}]")
        self.p_clean.enableAutoRange()
        self.p_frames.enableAutoRange(axis="y")

        sweep_overall, sweep_segmental = snr_vs_weight(x, noise, SWEEP_WEIGHTS, sr, frame_ms, overlap, limits)
        self.c_sweep_overall.setData(SWEEP_WEIGHTS, sweep_overall)
        self.c_sweep_segmental.setData(SWEEP_WEIGHTS, sweep_segmental)
        self.l_weight.setValue(np.log10(k))

        n_frames = len(seg.frame_snr)
        undefined = f", {seg.n_undefined} left out (silent)" if seg.n_undefined else ""
        self.result_label.setText(
            f"Overall SNR: <b>{overall:.2f} dB</b> &nbsp;|&nbsp; Segmental SNR: <b>{seg.mean:.2f} dB</b> "
            f"({n_frames} frames × {frame_ms} ms{', 50 % overlap' if overlap else ''}{undefined}) &nbsp;|&nbsp; "
            f"Difference (overall − segmental): <b>{overall - seg.mean:.2f} dB</b>")
        self.overall, self.segmental = overall, seg
