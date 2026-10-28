from PySide6 import QtCore
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer

from ..core.signal import AudioSignal
from ..core.wavio import wav_bytes


class SignalPlayer(QtCore.QObject):
    """Plays an in-memory AudioSignal. Reports the position often enough to drive a smooth playhead."""
    positionChanged = QtCore.Signal(float)  # seconds
    playingChanged = QtCore.Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._player = QMediaPlayer(self)
        self._output = QAudioOutput(self)
        self._player.setAudioOutput(self._output)
        self._buffer: QtCore.QBuffer | None = None
        self._duration = 0.0
        self._timer = QtCore.QTimer(self, interval=30)
        self._timer.timeout.connect(self._emit_position)
        self._player.playbackStateChanged.connect(self._on_state)
        self._player.mediaStatusChanged.connect(self._on_status)

    def set_signal(self, sig: AudioSignal):
        self._player.stop()
        self._player.setSource(QtCore.QUrl())
        if self._buffer is not None:
            self._buffer.deleteLater()
        self._buffer = QtCore.QBuffer(self)
        self._buffer.setData(QtCore.QByteArray(wav_bytes(sig)))
        self._buffer.open(QtCore.QIODevice.ReadOnly)
        self._duration = sig.duration
        self._player.setSourceDevice(self._buffer, QtCore.QUrl("signal.wav"))
        self.positionChanged.emit(0.0)

    @property
    def position(self) -> float:
        return self._player.position() / 1000

    def play(self):
        if self._player.mediaStatus() == QMediaPlayer.EndOfMedia or self.position >= self._duration:
            self._player.setPosition(0)
        self._player.play()

    def stop(self):
        """Pause, keeping the current position."""
        if self._player.playbackState() == QMediaPlayer.PlayingState:
            self._player.pause()
        self._emit_position()

    def rewind(self):
        self.seek(0.0)

    def seek(self, seconds: float):
        seconds = max(0.0, min(seconds, self._duration))
        self._player.setPosition(int(round(seconds * 1000)))
        self.positionChanged.emit(seconds)  # the player applies setPosition asynchronously

    def _emit_position(self):
        self.positionChanged.emit(self.position)

    def _on_state(self, state):
        playing = state == QMediaPlayer.PlayingState
        if playing:
            self._timer.start()
        else:
            self._timer.stop()
        self.playingChanged.emit(playing)

    def _on_status(self, status):
        if status == QMediaPlayer.EndOfMedia:
            self.positionChanged.emit(self._duration)
