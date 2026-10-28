import time
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from ..core.effect_base import Effect
from ..core.effects import EFFECTS
from ..core.signal import AudioSignal
from ..core.wavio import read_wav, write_wav
from .signal_view import SignalView


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Garso efektai")
        self.resize(1200, 900)
        # Effect chain: [(label, signal)], first entry is the loaded file, last is the input to the next effect
        self.history: list[tuple[str, AudioSignal]] = []
        self.processed: AudioSignal | None = None
        self.processed_name = ""

        # Controls: file, effect choice, its parameters, apply / keep / undo / reset, save
        open_btn = QtWidgets.QPushButton("Open WAV…")
        open_btn.clicked.connect(self.open_file)
        self.effect_box = QtWidgets.QComboBox()
        for eff in EFFECTS:
            self.effect_box.addItem(eff.name)
        self.effect_box.currentIndexChanged.connect(self._build_params)
        self.params_widget = QtWidgets.QWidget()
        self.params_layout = QtWidgets.QHBoxLayout(self.params_widget)
        self.params_layout.setContentsMargins(0, 0, 0, 0)
        self.param_inputs: dict[str, QtWidgets.QWidget] = {}
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.apply_btn.clicked.connect(self.apply_effect)
        self.keep_btn = QtWidgets.QPushButton("Keep result")
        self.keep_btn.setToolTip("Make the processed signal the input for the next effect")
        self.keep_btn.clicked.connect(self.keep_result)
        self.undo_btn = QtWidgets.QPushButton("Undo")
        self.undo_btn.setToolTip("Drop the last kept effect")
        self.undo_btn.clicked.connect(self.undo)
        self.reset_btn = QtWidgets.QPushButton("Reset")
        self.reset_btn.setToolTip("Go back to the loaded file")
        self.reset_btn.clicked.connect(self.reset)
        self.save_btn = QtWidgets.QPushButton("Save as…")
        self.save_btn.clicked.connect(self.save_as)

        row1 = QtWidgets.QHBoxLayout()
        row1.addWidget(open_btn)
        row1.addSpacing(20)
        row1.addWidget(QtWidgets.QLabel("Effect:"))
        row1.addWidget(self.effect_box)
        row1.addWidget(self.params_widget)
        row1.addWidget(self.apply_btn)
        row1.addWidget(self.keep_btn)
        row1.addSpacing(20)
        row1.addWidget(self.undo_btn)
        row1.addWidget(self.reset_btn)
        row1.addStretch()
        row1.addWidget(self.save_btn)

        # Visibility toggles and the current chain
        self.show_input = QtWidgets.QCheckBox("Input", checked=True)
        self.show_inter = QtWidgets.QCheckBox("Intermediate", checked=True)
        self.show_processed = QtWidgets.QCheckBox("Processed", checked=True)
        for box in (self.show_input, self.show_inter, self.show_processed):
            box.toggled.connect(self._update_visibility)
        self.chain_label = QtWidgets.QLabel("")

        row2 = QtWidgets.QHBoxLayout()
        row2.addWidget(QtWidgets.QLabel("Show:"))
        row2.addWidget(self.show_input)
        row2.addWidget(self.show_inter)
        row2.addWidget(self.show_processed)
        row2.addSpacing(30)
        row2.addWidget(self.chain_label)
        row2.addStretch()

        # Signals stacked top to bottom: input, intermediates, processed
        self.views: list[SignalView] = []  # every view with a player, so only one plays at a time
        self.input_view = self._add_view(SignalView())
        self.processed_view = self._add_view(SignalView())
        self.inter_views: list[SignalView] = []
        self.inter_box = QtWidgets.QWidget()
        self.inter_layout = QtWidgets.QVBoxLayout(self.inter_box)
        self.inter_layout.setContentsMargins(0, 0, 0, 0)

        stack = QtWidgets.QWidget()
        stack_layout = QtWidgets.QVBoxLayout(stack)
        stack_layout.addWidget(self.input_view)
        stack_layout.addWidget(self.inter_box)
        stack_layout.addWidget(self.processed_view)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(stack)

        central = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(central)
        v.addLayout(row1)
        v.addLayout(row2)
        v.addWidget(scroll)
        self.setCentralWidget(central)
        self._build_params()
        self._update_controls()
        self._update_visibility()

        menu = self.menuBar().addMenu("&File")
        menu.addAction("&Open WAV…", self.open_file, "Ctrl+O")
        menu.addAction("&Save as…", self.save_as, "Ctrl+S")
        menu.addSeparator()
        menu.addAction("&Quit", self.close, "Ctrl+Q")

    @property
    def signal(self) -> AudioSignal | None:
        """Current input: the loaded file with all kept effects applied."""
        return self.history[-1][1] if self.history else None

    @property
    def effect(self) -> Effect:
        return EFFECTS[self.effect_box.currentIndex()]

    def _add_view(self, view: SignalView) -> SignalView:
        view.playingChanged.connect(lambda playing, v=view: playing and self._stop_others(v))
        self.views.append(view)
        return view

    def _stop_others(self, active: SignalView):
        for v in self.views:
            if v is not active:
                v.stop()

    def _update_controls(self):
        loaded = bool(self.history)
        self.apply_btn.setEnabled(loaded)
        self.save_btn.setEnabled(loaded)
        self.keep_btn.setEnabled(self.processed is not None)
        self.undo_btn.setEnabled(len(self.history) > 1)
        self.reset_btn.setEnabled(len(self.history) > 1)
        self.chain_label.setText("Chain: " + " → ".join(name for name, _ in self.history) if loaded else "")

    def _update_visibility(self):
        shown = [(self.input_view, self.show_input.isChecked()),
                 (self.processed_view, self.show_processed.isChecked())]
        shown += [(v, self.show_inter.isChecked()) for v in self.inter_views]
        for view, on in shown:
            on = on and view.signal is not None
            if not on:
                view.stop()
            view.setVisible(on)

    def _build_params(self):
        while self.params_layout.count():
            self.params_layout.takeAt(0).widget().deleteLater()
        self.param_inputs.clear()
        for p in self.effect.params:
            if p.choices:
                w = QtWidgets.QComboBox()
                w.addItems(p.choices)
                w.setCurrentText(str(p.default))
            else:
                w = QtWidgets.QDoubleSpinBox(decimals=p.decimals, singleStep=p.step)
                w.setRange(p.minimum, p.maximum)
                w.setValue(p.default)
                if p.unit:
                    w.setSuffix(f" {p.unit}")
            self.params_layout.addWidget(QtWidgets.QLabel(f"{p.label}:"))
            self.params_layout.addWidget(w)
            self.param_inputs[p.key] = w

    def _param_values(self) -> dict:
        return {key: w.currentText() if isinstance(w, QtWidgets.QComboBox) else w.value()
                for key, w in self.param_inputs.items()}

    def open_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open WAV file", "", "WAV files (*.wav);;All files (*)")
        if path:
            self.load(path)

    def load(self, path):
        try:
            t0 = time.perf_counter()
            sig = read_wav(path)
            t1 = time.perf_counter()
            self.history = [(sig.name, sig)]
            self._show_input()
            t2 = time.perf_counter()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Cannot open {Path(path).name}:\n{e}")
            return

        self.setWindowTitle(f"Garso efektai — {sig.name}")
        self.statusBar().showMessage(
            f"{sig.n_samples:,} samples × {sig.n_channels} ch, {sig.duration:.3f} s  |  "
            f"read {1000 * (t1 - t0):.0f} ms, plot {1000 * (t2 - t1):.0f} ms")

    def _show_input(self):
        """Show the current input signal; any result belonged to the previous input, so it is dropped."""
        self._clear_results()
        names = [name for name, _ in self.history]
        title = f"Input: {names[0]}" + "".join(f" → {n}" for n in names[1:])
        self.input_view.set_signal(self.signal, title)
        self._update_controls()
        self._update_visibility()

    def _clear_results(self):
        self.processed = None
        self.processed_view.clear()
        for view in self.inter_views:
            view.clear()
            self.views.remove(view)
            view.deleteLater()
        self.inter_views.clear()

    def _show_result(self, caption: str, sig: AudioSignal, view: SignalView):
        view.set_signal(sig, caption)
        view.plot.link_x(self.input_view.plot)

    def apply_effect(self):
        if not self.signal:
            return
        effect = self.effect
        try:
            t0 = time.perf_counter()
            result = effect.apply(self.signal, **self._param_values())
            t1 = time.perf_counter()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Effect “{effect.name}” failed:\n{e}")
            return

        self._clear_results()
        self.processed, self.processed_name = result.output, effect.name
        for caption, sig in result.intermediates:
            view = self._add_view(SignalView())
            self._show_result(f"Intermediate: {caption}", sig, view)
            self.inter_layout.addWidget(view)
            self.inter_views.append(view)
        self._show_result(f"Processed: {effect.name}", result.output, self.processed_view)
        self._update_controls()
        self._update_visibility()
        self.statusBar().showMessage(f"{effect.name}: processed in {1000 * (t1 - t0):.0f} ms")

    def keep_result(self):
        if self.processed is None:
            return
        self.history.append((self.processed_name, self.processed))
        self._show_input()
        self.statusBar().showMessage(f"Kept {self.processed_name}; it is now the input for the next effect")

    def undo(self):
        if len(self.history) > 1:
            name, _ = self.history.pop()
            self._show_input()
            self.statusBar().showMessage(f"Removed {name} from the chain")

    def reset(self):
        if len(self.history) > 1:
            del self.history[1:]
            self._show_input()
            self.statusBar().showMessage("Back to the loaded file")

    def save_as(self):
        if not self.signal:
            return
        sig = self.processed if self.processed is not None else self.signal
        steps = [name for name, _ in self.history[1:]]
        if self.processed is not None:
            steps.append(self.processed_name)
        slug = lambda s: "_".join("".join(c if c.isalnum() else " " for c in s).split()).lower()
        suffix = "_".join(slug(s) for s in steps)
        default = f"{Path(self.history[0][0]).stem}_{suffix or 'copy'}.wav"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save as", default, "WAV files (*.wav)")
        if not path:
            return
        if not path.lower().endswith(".wav"):
            path += ".wav"
        write_wav(path, sig)
        self.statusBar().showMessage(f"Saved {sig.n_samples:,} samples ({sig.duration:.3f} s) to {path}")
