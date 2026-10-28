import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.signal import AudioSignal
from ..core.timefmt import choose_time_unit

CHANNEL_NAMES = {
    1: ["Mono"],
    2: ["Left", "Right"],
    6: ["FL", "FR", "FC", "LFE", "BL", "BR"],
    8: ["FL", "FR", "FC", "LFE", "BL", "BR", "SL", "SR"],
}


class SignalPlot(QtWidgets.QWidget):
    """Stacked per-channel plots with linked time axes and a shared red playhead.

    Emits seekRequested(seconds) when the playhead is dragged.
    """
    seekRequested = QtCore.Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout_widget = pg.GraphicsLayoutWidget()
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.layout_widget)
        self.plots: list[pg.PlotItem] = []
        self.playheads: list[pg.InfiniteLine] = []
        self.scale = 1.0
        self.unit = "s"
        self._syncing = False

    def clear(self):
        if self.plots:
            self.plots[0].setXLink(None)
        self.layout_widget.clear()
        self.plots.clear()
        self.playheads.clear()

    def link_x(self, other: "SignalPlot"):
        """Zoom/pan together with another plot, when both use the same time unit."""
        if self.plots and other.plots and other is not self and self.unit == other.unit:
            self.plots[0].setXLink(other.plots[0])

    def set_signal(self, sig: AudioSignal, title: str | None = None):
        self.clear()
        self.scale, self.unit = choose_time_unit(sig.duration)
        t = sig.time_axis() * self.scale
        t_end = sig.duration * self.scale
        lo, hi = sig.amplitude_range
        names = CHANNEL_NAMES.get(sig.n_channels, [f"Ch {i + 1}" for i in range(sig.n_channels)])
        amp_unit = "float" if sig.fmt == "float" else f"{sig.bits}-bit int"

        if title is None:
            title = sig.name
        title += (f"  |  channels: {sig.n_channels}, sample rate: {sig.sample_rate} Hz, "
                  f"bit depth: {sig.bits} ({sig.fmt.upper()})")
        self.layout_widget.addLabel(title, row=0, col=0, size="11pt", bold=True)

        for ch in range(sig.n_channels):
            p = self.layout_widget.addPlot(row=ch + 1, col=0)
            p.setLabel("left", f"{names[ch]} [{amp_unit}]")
            p.showGrid(x=True, y=True, alpha=0.3)
            # Long-signal handling: draw only visible data, reduced to min/max per pixel column
            p.setClipToView(True)
            p.setDownsampling(auto=True, mode="peak")
            p.plot(t, np.asarray(sig.samples[:, ch]), pen=pg.mkPen((30, 110, 200), width=1))
            p.setYRange(lo, hi, padding=0.02)
            p.setLimits(xMin=t[0] if len(t) else 0, xMax=t_end, yMin=lo * 1.1, yMax=hi * 1.1)
            if self.plots:
                p.setXLink(self.plots[0])
            line = pg.InfiniteLine(pos=0, angle=90, movable=True, pen=pg.mkPen("r", width=2),
                                   hoverPen=pg.mkPen((255, 90, 90), width=3), bounds=(0, t_end))
            line.setZValue(20)
            line.sigPositionChanged.connect(self._on_line_dragged)
            p.addItem(line)
            self.plots.append(p)
            self.playheads.append(line)
        if self.plots:
            self.plots[-1].getAxis("bottom").setLabel(f"Time [{self.unit}]")
        # Keep each channel readable when there are many of them
        self.layout_widget.setMinimumHeight(160 * sig.n_channels + 40)

    def _on_line_dragged(self, line):
        if self._syncing:
            return
        t = line.value() / self.scale
        self._move_lines(t, source=line)
        self.seekRequested.emit(t)

    def _move_lines(self, t_s: float, source=None):
        self._syncing = True
        for line in self.playheads:
            if line is not source:
                line.setValue(t_s * self.scale)
        self._syncing = False

    def set_playhead(self, t_s: float):
        if not self.plots:
            return
        self._move_lines(t_s)
        # Keep the playhead in view when zoomed in
        x = t_s * self.scale
        x0, x1 = self.plots[0].viewRange()[0]
        if not x0 <= x <= x1:
            w = x1 - x0
            self.plots[0].setXRange(x, x + w, padding=0)

    def zoom_all(self):
        if self.plots:
            self.plots[0].enableAutoRange(axis="x")
