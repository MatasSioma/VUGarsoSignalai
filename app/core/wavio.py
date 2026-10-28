"""Minimal RIFF/WAVE reader and writer keeping real sample values for the file's bit depth."""
import struct
from pathlib import Path

import numpy as np

from .signal import AudioSignal

WAVE_FORMAT_PCM = 0x0001
WAVE_FORMAT_IEEE_FLOAT = 0x0003
WAVE_FORMAT_EXTENSIBLE = 0xFFFE


def read_wav(path) -> AudioSignal:
    path = Path(path)
    data = path.read_bytes()
    if data[0:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("Not a RIFF/WAVE file")

    fmt = None
    raw = None
    pos = 12
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        size = struct.unpack_from("<I", data, pos + 4)[0]
        body = pos + 8
        if cid == b"fmt ":
            fmt_tag, channels, rate, _byte_rate, block_align, bits = struct.unpack_from("<HHIIHH", data, body)
            if fmt_tag == WAVE_FORMAT_EXTENSIBLE and size >= 40:
                fmt_tag = struct.unpack_from("<H", data, body + 24)[0]
            fmt = (fmt_tag, channels, rate, block_align, bits)
        elif cid == b"data":
            raw = data[body:body + size]
        pos = body + size + (size & 1)

    if fmt is None or raw is None:
        raise ValueError("Missing fmt or data chunk")
    fmt_tag, channels, rate, block_align, bits = fmt
    bytes_per_sample = block_align // channels
    n = len(raw) // block_align
    raw = raw[:n * block_align]

    if fmt_tag == WAVE_FORMAT_PCM:
        if bytes_per_sample == 1:
            samples = np.frombuffer(raw, dtype=np.uint8)
        elif bytes_per_sample == 2:
            samples = np.frombuffer(raw, dtype="<i2")
        elif bytes_per_sample == 3:
            b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
            samples = ((b[:, 0] << 8) | (b[:, 1] << 16) | (b[:, 2] << 24)) >> 8
        elif bytes_per_sample == 4:
            samples = np.frombuffer(raw, dtype="<i4")
        else:
            raise ValueError(f"Unsupported PCM sample size: {bytes_per_sample} bytes")
        kind = "pcm"
    elif fmt_tag == WAVE_FORMAT_IEEE_FLOAT:
        samples = np.frombuffer(raw, dtype="<f4" if bytes_per_sample == 4 else "<f8")
        kind = "float"
    else:
        raise ValueError(f"Unsupported WAV format code: 0x{fmt_tag:04X}")

    samples = samples.reshape(n, channels)
    return AudioSignal(samples, rate, bits, kind, path.name)


def wav_bytes(sig: AudioSignal) -> bytes:
    ch, rate, bits = sig.n_channels, sig.sample_rate, sig.bits
    bps = (bits + 7) // 8
    block_align = ch * bps
    x = sig.samples

    if sig.fmt == "float":
        fmt_tag = WAVE_FORMAT_IEEE_FLOAT
        raw = x.astype("<f4" if bps == 4 else "<f8").tobytes()
    elif bps == 1:
        fmt_tag, raw = WAVE_FORMAT_PCM, x.astype(np.uint8).tobytes()
    elif bps == 2:
        fmt_tag, raw = WAVE_FORMAT_PCM, x.astype("<i2").tobytes()
    elif bps == 3:
        fmt_tag = WAVE_FORMAT_PCM
        v = x.astype("<i4").reshape(-1, 1).view(np.uint8).reshape(-1, 4)
        raw = v[:, :3].tobytes()  # drop the most significant (sign-extension) byte
    else:
        fmt_tag, raw = WAVE_FORMAT_PCM, x.astype("<i4").tobytes()

    # Use WAVE_FORMAT_EXTENSIBLE for >2 channels or >16 bits, as the spec recommends
    extensible = ch > 2 or bits > 16
    if extensible:
        guid_tail = b"\x00\x00\x00\x00\x10\x00\x80\x00\x00\xAA\x00\x38\x9B\x71"
        fmt_body = struct.pack("<HHIIHH", WAVE_FORMAT_EXTENSIBLE, ch, rate, rate * block_align, block_align, bps * 8)
        fmt_body += struct.pack("<HHI", 22, bits, 0) + struct.pack("<H", fmt_tag) + guid_tail
    else:
        fmt_body = struct.pack("<HHIIHH", fmt_tag, ch, rate, rate * block_align, block_align, bits)

    pad = b"\x00" if len(raw) & 1 else b""
    chunks = b"fmt " + struct.pack("<I", len(fmt_body)) + fmt_body
    chunks += b"data" + struct.pack("<I", len(raw)) + raw + pad
    return b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks


def write_wav(path, sig: AudioSignal) -> None:
    Path(path).write_bytes(wav_bytes(sig))
