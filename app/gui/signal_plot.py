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
    """Stacked per-channel plots with linked time axes and a shared selection region.

    Emits regionChanged(start_s, end_s) whenever the selection is dragged.
    """
    regionChanged = QtCore.Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout_widget = pg.GraphicsLayoutWidget()
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.layout_widget)
        self.plots: list[pg.PlotItem] = []
        self.regions: list[pg.LinearRegionItem] = []
        self.scale = 1.0
        self.unit = "s"
        self._syncing = False

    def set_signal(self, sig: AudioSignal):
        self.layout_widget.clear()
        self.plots.clear()
        self.regions.clear()
        self.scale, self.unit = choose_time_unit(sig.duration)
        t = sig.time_axis() * self.scale
        lo, hi = sig.amplitude_range
        names = CHANNEL_NAMES.get(sig.n_channels, [f"Ch {i + 1}" for i in range(sig.n_channels)])
        amp_unit = "float" if sig.fmt == "float" else f"{sig.bits}-bit int"

        title = (f"{sig.name}  |  channels: {sig.n_channels}, sample rate: {sig.sample_rate} Hz, "
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
            p.setLimits(xMin=t[0] if len(t) else 0, xMax=sig.duration * self.scale, yMin=lo * 1.1, yMax=hi * 1.1)
            if self.plots:
                p.setXLink(self.plots[0])
            region = pg.LinearRegionItem(brush=pg.mkBrush(255, 160, 0, 60),
                                         bounds=(0, sig.duration * self.scale))
            region.setZValue(10)
            region.sigRegionChanged.connect(self._on_region_changed)
            p.addItem(region)
            self.plots.append(p)
            self.regions.append(region)
        if self.plots:
            self.plots[-1].setLabel("bottom", "Time", units=None)
            self.plots[-1].getAxis("bottom").setLabel(f"Time [{self.unit}]")
        # Keep each channel readable when there are many of them
        self.layout_widget.setMinimumHeight(160 * sig.n_channels + 40)

    def _on_region_changed(self, region):
        if self._syncing:
            return
        a, b = region.getRegion()
        self.set_region(a / self.scale, b / self.scale, emit=True, source=region)

    def set_region(self, start_s: float, end_s: float, emit=False, source=None):
        self._syncing = True
        for r in self.regions:
            if r is not source:
                r.setRegion((start_s * self.scale, end_s * self.scale))
        self._syncing = False
        if emit:
            self.regionChanged.emit(start_s, end_s)

    def zoom_to(self, start_s: float, end_s: float):
        if self.plots:
            self.plots[0].setXRange(start_s * self.scale, end_s * self.scale, padding=0.02)

    def zoom_all(self):
        if self.plots:
            self.plots[0].enableAutoRange(axis="x")
