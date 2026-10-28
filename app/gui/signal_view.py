from PySide6 import QtCore, QtWidgets

from ..core.signal import AudioSignal
from .player import SignalPlayer
from .signal_plot import SignalPlot


class SignalView(QtWidgets.QWidget):
    """Signal plot with its own player: play / stop / back to start, and a red playhead."""
    playingChanged = QtCore.Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.signal: AudioSignal | None = None
        self.plot = SignalPlot()
        self.player = SignalPlayer(self)

        style = self.style()
        self.play_btn = QtWidgets.QPushButton(style.standardIcon(QtWidgets.QStyle.SP_MediaPlay), "Play")
        self.play_btn.clicked.connect(self.player.play)
        self.stop_btn = QtWidgets.QPushButton(style.standardIcon(QtWidgets.QStyle.SP_MediaStop), "Stop")
        self.stop_btn.clicked.connect(self.player.stop)
        self.rewind_btn = QtWidgets.QPushButton(style.standardIcon(QtWidgets.QStyle.SP_MediaSkipBackward),
                                                "To start")
        self.rewind_btn.clicked.connect(self.player.rewind)
        self.zoom_btn = QtWidgets.QPushButton("Show all")
        self.zoom_btn.clicked.connect(self.plot.zoom_all)
        self.time_label = QtWidgets.QLabel("")

        self.player.positionChanged.connect(self._on_position)
        self.player.playingChanged.connect(self._on_playing)
        self.plot.seekRequested.connect(self.player.seek)

        bar = QtWidgets.QHBoxLayout()
        for w in (self.play_btn, self.stop_btn, self.rewind_btn):
            bar.addWidget(w)
        bar.addSpacing(10)
        bar.addWidget(self.time_label)
        bar.addStretch()
        bar.addWidget(self.zoom_btn)

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.addLayout(bar)
        v.addWidget(self.plot)
        self._set_controls_enabled(False)

    def _set_controls_enabled(self, on: bool):
        for w in (self.play_btn, self.stop_btn, self.rewind_btn, self.zoom_btn):
            w.setEnabled(on)

    def set_signal(self, sig: AudioSignal, title: str | None = None):
        self.signal = sig
        self.plot.set_signal(sig, title)
        self.player.set_signal(sig)
        self._set_controls_enabled(True)

    def clear(self):
        self.player.stop()
        self.signal = None
        self.plot.clear()
        self.time_label.setText("")
        self._set_controls_enabled(False)

    def stop(self):
        self.player.stop()

    def _on_position(self, t_s: float):
        self.plot.set_playhead(t_s)
        if self.signal:
            scale, unit = self.plot.scale, self.plot.unit
            self.time_label.setText(f"{t_s * scale:.3f} / {self.signal.duration * scale:.3f} {unit}")

    def _on_playing(self, playing: bool):
        self.play_btn.setEnabled(not playing)
        self.playingChanged.emit(playing)
