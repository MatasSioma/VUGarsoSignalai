import time
from pathlib import Path

from PySide6 import QtWidgets

from ..core.signal import AudioSignal
from ..core.wavio import read_wav, write_wav
from .signal_plot import SignalPlot


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Garso signalų atvaizdavimas")
        self.resize(1200, 800)
        self.signal: AudioSignal | None = None

        self.plot = SignalPlot()
        self.plot.regionChanged.connect(self._on_plot_region)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.plot)

        # Segment controls: start and duration in the plot's time unit
        self.start_box = QtWidgets.QDoubleSpinBox(decimals=3, singleStep=0.1)
        self.dur_box = QtWidgets.QDoubleSpinBox(decimals=3, singleStep=0.1)
        for box in (self.start_box, self.dur_box):
            box.setRange(0, 0)
            box.valueChanged.connect(self._on_boxes_changed)
        self.unit_label = QtWidgets.QLabel("")
        zoom_seg = QtWidgets.QPushButton("Zoom to segment")
        zoom_seg.clicked.connect(lambda: self.signal and self.plot.zoom_to(*self._segment_bounds()))
        zoom_all = QtWidgets.QPushButton("Show all")
        zoom_all.clicked.connect(self.plot.zoom_all)
        self.save_btn = QtWidgets.QPushButton("Save segment…")
        self.save_btn.clicked.connect(self.save_segment)
        open_btn = QtWidgets.QPushButton("Open WAV…")
        open_btn.clicked.connect(self.open_file)

        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(open_btn)
        bar.addSpacing(20)
        bar.addWidget(QtWidgets.QLabel("Segment start:"))
        bar.addWidget(self.start_box)
        bar.addWidget(QtWidgets.QLabel("duration:"))
        bar.addWidget(self.dur_box)
        bar.addWidget(self.unit_label)
        bar.addWidget(zoom_seg)
        bar.addWidget(zoom_all)
        bar.addWidget(self.save_btn)
        bar.addStretch()

        central = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(central)
        v.addLayout(bar)
        v.addWidget(scroll)
        self.setCentralWidget(central)
        self._set_controls_enabled(False)

        menu = self.menuBar().addMenu("&File")
        menu.addAction("&Open WAV…", self.open_file, "Ctrl+O")
        menu.addAction("&Save segment…", self.save_segment, "Ctrl+S")
        menu.addSeparator()
        menu.addAction("&Quit", self.close, "Ctrl+Q")

    def _set_controls_enabled(self, on: bool):
        for w in (self.start_box, self.dur_box, self.save_btn):
            w.setEnabled(on)

    def open_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open WAV file", "", "WAV files (*.wav);;All files (*)")
        if path:
            self.load(path)

    def load(self, path):
        try:
            t0 = time.perf_counter()
            sig = read_wav(path)
            t1 = time.perf_counter()
            self.signal = sig
            self.plot.set_signal(sig)
            t2 = time.perf_counter()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Cannot open {Path(path).name}:\n{e}")
            return

        scale, unit = self.plot.scale, self.plot.unit
        total = sig.duration * scale
        self.unit_label.setText(unit)
        for box in (self.start_box, self.dur_box):
            box.blockSignals(True)
            box.setRange(0, total)
            box.setSingleStep(max(total / 100, 0.001))
        self.start_box.setValue(total * 0.25)
        self.dur_box.setValue(total * 0.25)
        for box in (self.start_box, self.dur_box):
            box.blockSignals(False)
        self._on_boxes_changed()
        self._set_controls_enabled(True)
        self.setWindowTitle(f"Garso signalų atvaizdavimas — {sig.name}")
        self.statusBar().showMessage(
            f"{sig.n_samples:,} samples × {sig.n_channels} ch, {sig.duration:.3f} s  |  "
            f"read {1000 * (t1 - t0):.0f} ms, plot {1000 * (t2 - t1):.0f} ms")

    def _segment_bounds(self) -> tuple[float, float]:
        """Selected segment as (start, end) in seconds."""
        s = self.start_box.value() / self.plot.scale
        d = self.dur_box.value() / self.plot.scale
        return s, min(s + d, self.signal.duration)

    def _on_boxes_changed(self):
        if self.signal:
            self.plot.set_region(*self._segment_bounds())

    def _on_plot_region(self, start_s: float, end_s: float):
        scale = self.plot.scale
        for box, val in ((self.start_box, start_s * scale), (self.dur_box, (end_s - start_s) * scale)):
            box.blockSignals(True)
            box.setValue(val)
            box.blockSignals(False)

    def save_segment(self):
        if not self.signal:
            return
        start, end = self._segment_bounds()
        seg = self.signal.segment(start, end - start)
        if seg.n_samples == 0:
            QtWidgets.QMessageBox.warning(self, "Empty segment", "Selected segment contains no samples.")
            return
        default = f"{Path(self.signal.name).stem}_segment_{start:.3f}-{end:.3f}s.wav"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save segment", default, "WAV files (*.wav)")
        if not path:
            return
        if not path.lower().endswith(".wav"):
            path += ".wav"
        write_wav(path, seg)
        self.statusBar().showMessage(f"Saved {seg.n_samples:,} samples ({seg.duration:.3f} s) to {path}")
