# Architecture: `garsas` WAV signal viewer

A desktop tool for opening `.wav` files, plotting them against real time and amplitude units, and cutting out segments (UZD_01). It is laid out so the time-domain analyses of UZD_02 can be added without touching the loading or plotting code.

## Running

```bash
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # numpy, PySide6, pyqtgraph
python -m garsas                   # opens an empty window, use File → Open
python -m garsas path/to/file.wav  # opens the file on start
```

`python -m garsas` tells Python to find the **package** `garsas` on the import path (the current directory is on it) and run its `garsas/__main__.py` as the program. Running it this way matters: the code uses package-relative imports such as `from .gui.main_window import MainWindow`. Those only work when the file is loaded as part of the package. `python garsas/__main__.py` would fail with *"attempted relative import with no known parent package"*.

## Tech stack

| Library | Role | Why |
|---|---|---|
| **numpy** | Signal storage and all math | Samples are one `ndarray`; decoding, slicing and future analysis are vectorised, with no Python loops over samples |
| **PySide6** (Qt 6) | Window, buttons, native file dialogs | The official Qt binding for Python. `QFileDialog` is the OS's standard dialog (requirement 1) |
| **pyqtgraph** | Plotting | Built on Qt and made for large, interactive data. It has built-in peak downsampling and a draggable region item |

The WAV decoding is **our own code**, not `scipy`/`soundfile`, so we keep real sample values for every bit depth (see [`wavio.py`](#coreWaviopy--reading-and-writing-wav)).

## Directory layout

```
garsas/
├── __main__.py          entry point: creates QApplication + MainWindow
├── core/                ── no GUI code; pure Python + numpy ──
│   ├── signal.py        AudioSignal dataclass (the in-memory signal)
│   ├── wavio.py         read_wav() / write_wav(): RIFF/WAVE parser and writer
│   └── timefmt.py       choose_time_unit(): ms / s / min / h
├── analysis/            ── UZD_02, no GUI code ──
│   ├── framing.py       frames(): split a channel into 20–30 ms frames
│   └── snr.py           task 6: noise generation, overall and segmental SNR
└── gui/                 ── Qt code only ──
    ├── signal_plot.py   SignalPlot widget: stacked channel plots + selection region
    ├── main_window.py   MainWindow: menu, toolbar, segment controls, save
    └── snr_window.py    SnrWindow: segmental SNR analysis (task 6)
tests/                   unittest suite: python -m unittest
requirements.txt
garsai#1/                test recordings
```

**Layering rule:** `gui` may import `core` (and later `analysis`). `core` and `analysis` never import `gui`. The signal logic can then be tested from a plain script or a Jupyter notebook without opening a window.

```
 ┌──────────── gui ────────────┐
 │ MainWindow ──► SignalPlot   │
 └──────┬──────────────┬───────┘
        │              │
        ▼              ▼
 ┌── core ─────────────────────┐    ┌── analysis (UZD_02) ──┐
 │ wavio ─► AudioSignal        │◄───│ framing, energy, zcr… │
 │          timefmt            │    └───────────────────────┘
 └─────────────────────────────┘
```

## Data flow

1. **Open**: `MainWindow.open_file()` shows `QFileDialog.getOpenFileName` and calls `load(path)`.
2. **Decode**: `read_wav(path)` reads the whole file into memory and returns one `AudioSignal` (requirement 2: all later work uses this object, not the file).
3. **Plot**: `SignalPlot.set_signal(sig)` builds one plot per channel.
4. **Select**: the user drags the orange region *or* types start/duration. Each one updates the other (see [Segment synchronisation](#segment-synchronisation)).
5. **Save**: `sig.segment(start, duration)` slices the array and `write_wav()` writes it in the same format as the original.

## `core/signal.py`: `AudioSignal`

```python
@dataclass
class AudioSignal:
    samples: np.ndarray   # shape (n_samples, n_channels)
    sample_rate: int      # Hz
    bits: int             # quantisation depth from the header
    fmt: str              # 'pcm' (integers) or 'float'
    name: str             # file name, shown in the plot title
```

- `samples` holds the **native** values: `int16` for 16-bit, `int32` holding the real ±8 388 608 range for 24-bit, `float32` for float files. Nothing is normalised.
- Derived properties: `n_samples`, `n_channels`, `duration` (= `n_samples / sample_rate`).
- `amplitude_range`: theoretical limits for the bit depth, `[-2^(b-1), 2^(b-1)-1]`. Float files use `[-1, 1]` and 8-bit uses `[0, 255]` (8-bit WAV is unsigned). The y-axis uses these limits, so a quiet recording *looks* quiet.
- `time_axis()`: `np.arange(n) / sample_rate`, the time of each sample in seconds.
- `segment(start_s, duration_s)`: converts seconds to sample indices (`round(t · sr)`), clamps them to the signal, and returns a new `AudioSignal` with the same metadata.

This is the one object every other layer passes around. UZD_02 analysis functions will take an `AudioSignal`, or one channel `sig.samples[:, ch]`, as input.

## `core/wavio.py`: reading and writing WAV

### File structure

A WAV file is a **RIFF** container: a 12-byte header followed by *chunks*.

```
"RIFF" <size:u32> "WAVE"
  "fmt " <size> <format tag, channels, sample rate, byte rate, block align, bits, [extension]>
  "LIST" <size> ...metadata... (ignored)
  "data" <size> <interleaved samples: L R L R ...>
```

All numbers are little-endian. `read_wav` walks the chunks in a loop. For each chunk it reads the 4-byte id and 4-byte size, handles `fmt ` and `data`, and skips everything else. Chunks are padded to an even length, hence `pos = body + size + (size & 1)`.

### Format tag

| Tag | Meaning |
|---|---|
| `0x0001` | Integer PCM |
| `0x0003` | IEEE float |
| `0xFFFE` | *WAVE_FORMAT_EXTENSIBLE*: used for >2 channels or >16 bits. The real tag is in the first 2 bytes of the `SubFormat` GUID at offset 24 of the fmt body |

### Decoding samples

`block_align` is the size of one frame (all channels), so `bytes_per_sample = block_align // channels`.

| Bytes | Decoding |
|---|---|
| 1 | `uint8` (unsigned, 128 = silence) |
| 2 | `<i2` (int16) |
| 3 | No numpy type exists, so we build it by hand (below) |
| 4 | `<i4` (int32), or `<f4` if the tag is float |
| 8 | `<f8` (float64) |

**24-bit decoding.** Each sample is 3 bytes `b0 b1 b2` (little-endian). We place them in the *top* three bytes of an int32 and then shift right by 8:

```python
((b0 << 8) | (b1 << 16) | (b2 << 24)) >> 8
```

Because `b2` lands in the sign-bit position, the arithmetic right shift performs **sign extension** for free. The result is the true 24-bit value in the range −8 388 608 … 8 388 607.

Finally, `samples.reshape(n, channels)` de-interleaves `L R L R …` into columns.

`np.frombuffer` makes no copy, which is why reading is fast (≈25 ms for a 23 MB file).

### Writing

`write_wav` does the reverse. For 24-bit it views each int32 as 4 bytes and keeps the lower 3. It writes an `EXTENSIBLE` header when there are >2 channels or >16 bits (as the spec recommends), otherwise a plain one. The round-trip `read → write → read` gives bit-identical samples for every test file.

### A note on the test files

`Music#1` and `Music#2` are named "24-bit", but their headers declare **32-bit** integer PCM (`bits=32`, valid bits = 32), and the lowest byte does hold data. The app correctly reports them as 32-bit.

## `core/timefmt.py`

`choose_time_unit(duration)` returns `(multiplier, label)`:

| Duration | Unit | Multiplier |
|---|---|---|
| < 1 s | ms | 1000 |
| < 120 s | s | 1 |
| < 2 h | min | 1/60 |
| otherwise | h | 1/3600 |

Internally everything stays in **seconds**. The multiplier is applied only when plotting and when showing values in the spin boxes (requirement 4).

## `gui/signal_plot.py`: `SignalPlot`

A widget holding a `pg.GraphicsLayoutWidget` with:

- a **title row** with the file name, channel count, sample rate, bit depth and format (requirement 6);
- **one `PlotItem` per channel**, labelled with standard names (Left/Right, FL/FR/FC/LFE/BL/BR/SL/SR for 5.1/7.1);
- the y-range fixed to `AudioSignal.amplitude_range` (requirement 5), and the x-axis in the chosen time unit;
- `setXLink(first_plot)`: zooming or panning one channel moves all of them.

### Long-signal handling

A 20-minute 48 kHz stereo file has 57.6 million points per channel, but a screen is only about 1 500 pixels wide. Two pyqtgraph options keep drawing fast:

- `setClipToView(True)`: only samples inside the visible x-range are handed to the renderer.
- `setDownsampling(auto=True, mode="peak")`: the visible samples are grouped into one bucket per pixel column, and only each bucket's **min and max** are drawn. The envelope looks the same as the full plot (no peaks are lost, unlike plain decimation). The number of drawn points is about `2 × width` whatever the file length.

Measured on the 20-minute test: reading 0.14 s, building the plot about 2.2 s (a one-off cost), and each zoom or pan repaint about 120 ms.

### Selection region

Every channel has its own `pg.LinearRegionItem` (orange band). When one is dragged, `_on_region_changed` copies its range to the others and emits the Qt signal `regionChanged(start_s, end_s)` in seconds. The `_syncing` flag prevents an infinite loop: updating region B must not re-trigger the update of region A.

`zoom_to(start, end)` and `zoom_all()` set the view range on the first plot. The linked plots follow.

## `gui/main_window.py`: `MainWindow`

- **Toolbar**: Open, segment *start* and *duration* spin boxes (in the current time unit), Zoom to segment, Show all, Save segment. The same actions are in the *File* menu with Ctrl+O / Ctrl+S / Ctrl+Q.
- A **scroll area** around the plot, so 8-channel files get readable plot heights (160 px per channel).
- A **status bar** showing sample count, duration and the measured read and plot times. This is useful for discussing long signals at the defense.
- `self.signal` holds the loaded `AudioSignal` (the in-memory copy).

### Segment synchronisation

```
spin boxes ──valueChanged──► _on_boxes_changed ──► plot.set_region(start_s, end_s)
plot drag  ──regionChanged─► _on_plot_region   ──► update spin boxes (signals blocked)
```

`blockSignals(True)` while the code sets the boxes stops the change from bouncing back to the plot. `_segment_bounds()` converts the box values from display units back to seconds and clamps the end to the signal length.

### Saving

`save_segment` → `signal.segment(start, dur)` → `QFileDialog.getSaveFileName` (with a default name such as `Sinus_125Hz_segment_1.000-3.000s.wav`) → `write_wav`. The result has the same sample rate, bit depth, format and channel count as the source (requirement 7).

## Requirement map (UZD_01)

| # | Requirement | Where |
|---|---|---|
| 1 | Pick file with standard dialog | `MainWindow.open_file` (`QFileDialog`) |
| 2 | Keep in memory, plot from memory | `read_wav` → `MainWindow.signal` |
| 3 | Any channels / rate / depth | `wavio.read_wav` (8/16/24/32 int, 32/64 float, EXTENSIBLE); `SignalPlot` one plot per channel |
| 4 | Continuous time axis, unit by duration | `timefmt.choose_time_unit`, `SignalPlot.set_signal` |
| 5 | Real amplitude values | native dtype in `AudioSignal`, y-range `amplitude_range` |
| 6 | File name + parameters on the plot | title label in `SignalPlot.set_signal` |
| 7 | Highlight segment by start/duration, save it | spin boxes + `LinearRegionItem`, `save_segment` |
| — | Very long signals | clip-to-view + peak downsampling; timings in the status bar |

## Extending for UZD_02

Most UZD_02 analyses run on **overlapping frames** (10–30 ms, 50 % overlap), so the first shared helper is:

```python
# analysis/framing.py
def frames(x, sr, frame_ms=20, overlap=0.5):
    n = int(sr * frame_ms / 1000)
    hop = max(1, int(n * (1 - overlap)))
    w = np.lib.stride_tricks.sliding_window_view(x, n)[::hop]   # (n_frames, n), no copy
    t = (np.arange(len(w)) * hop + n / 2) / sr                   # frame centre times, s
    return w, t
```

Each analysis is then a small numpy function over that 2-D array, for example:

```python
energy    = (w.astype(float) ** 2).sum(axis=1)
energy_db = 10 * np.log10(energy / energy.max())
zcr       = np.abs(np.diff(np.sign(w), axis=1)).sum(axis=1) / (2 * n)
```

On the GUI side, add a panel (for example `gui/analysis_panel.py`) with extra `PlotItem`s under the signal, `setXLink`ed to it. The analysis curves then line up in time with the waveform, and the existing selection region can choose which segment to analyse. Neither `core/` nor `SignalPlot` needs to change.

## UZD_02 task 6: segmental SNR

Opened with **Analysis → Segmental SNR…** (Ctrl+R) or the toolbar button once a file is loaded.

| Step | Where | What |
|---|---|---|
| a. useful signal + noise with a weight | `channel_as_float`, `generate_noise` | The chosen channel (or only the selected segment) is the useful signal. The noise is generated white Gaussian noise, zero mean, scaled to **exactly the signal's RMS**, so the weight `k` is a noise/signal amplitude ratio. Noisy signal = `x + k·n`. The seed makes it reproducible |
| b. SNR of the whole signal | `snr_db` | `10·log10(E_sig / E_tr)`, `E = Σx²`. Because of the scaling, `SNR = −20·log10(k)` exactly (k = 0.1 → 20 dB, k = 1 → 0 dB, k = 2 → −6.02 dB) |
| c. segmental SNR | `segmental_snr` | `frames()` splits signal and noise into 20–30 ms frames (overlap optional: none or 50 %). The SNR of each frame is computed and then averaged. Frames whose SNR is not finite (digital silence) are left out of the mean. Optionally each frame is clipped to the usual [−10, 35] dB first |
| SNR as a function of k | `snr_vs_weight` | Scaling noise by `k` shifts every SNR by `−20·log10(k)`, so frame energies are computed once and reused for the 101 weights of the plot |

The window shows the signal, `k·noise` and the noisy signal (same amplitude scale), the per-frame SNR with the overall and segmental SNR as dashed lines, and both SNRs against `k` (log axis). The result line gives the overall SNR, the segmental SNR and their difference.

What the test recordings show (k = 0.1, 20 ms frames, channel 1):

| Recording | Overall | Segmental | Why |
|---|---|---|---|
| Sinus 125 / 500 Hz, Multitone | 20.00 dB | 20.0 dB | Stationary: every frame has the same energy |
| Pink noise / Brown noise | 20.00 dB | 19.7 / 17.9 dB | Nearly stationary; brown noise has slow level changes |
| Music#1 | 20.00 dB | 16.8 dB | Quieter passages give lower frame SNR |
| DTMF, Pure tones | 20.00 dB | −9.4 dB | Pauses between tones: those frames are near −40 dB and pull the mean down |
| Speech#1 (FC) | 20.00 dB | 20.6 dB raw, −8.2 dB clipped | Mostly digital silence: raw mean only covers the speech frames, with clipping silent frames count as −10 dB |

The overall SNR is dominated by the loud parts (energies are summed before the log). The segmental SNR weights every frame equally (logs are averaged), so it shows how much of the time the signal is buried in noise. Without clipping, both SNRs move by the same amount when `k` changes, so their difference is constant. With clipping the segmental curve flattens at 35 dB and −10 dB.

## Tests

```bash
python -m unittest -v
```

`tests/test_snr.py` checks framing, the noise, the SNR formula, `−20·log10(k)`, hand-computed segmental values, overlap, silence and clipping, plus the expected results on the recordings in `garsai#1/`. `tests/test_snr_gui.py` drives the real `MainWindow` + `SnrWindow` offscreen (`QT_QPA_PLATFORM=offscreen`).

## Platform notes

The code uses only portable APIs (`pathlib`, Qt), so it runs unchanged on Linux, WSL and native Windows. Under **WSL2** the window is shown through WSLg (a remote-desktop bridge). Plotting maths and file reading run at the same speed, but repaints while dragging or zooming are smoother on **native Windows Python**, where Qt draws directly to the Windows desktop. On Windows, create a separate venv; the Linux `.venv` cannot be reused.
