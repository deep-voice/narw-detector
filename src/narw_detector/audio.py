"""Audio in: any WAV (or other libsndfile format), first channel, resampled to 2 kHz.

The model was trained on recordings decimated to 2 kHz, so every input is brought to 2 kHz the
same way for detection and for training: scipy.signal.resample_poly (a polyphase FIR with a
Kaiser window) at the reduced ratio 2000 / sr. Long files are resampled in chunks that overlap by
more than the filter's reach, so the result equals resampling the whole file at once.

Audio is float32 in [-1, 1). A 16-bit file read as float is int16 / 32768, the scale the model
was trained on.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SAMPLE_RATE = 2000
CHUNK_S = 600.0                      # input seconds resampled per chunk


def ratio(sr: int) -> tuple[int, int]:
    """(up, down) taking sr to SAMPLE_RATE, reduced."""
    g = math.gcd(SAMPLE_RATE, sr)
    return SAMPLE_RATE // g, sr // g


def _pad(up: int, down: int) -> int:
    """Input samples of context each side of a chunk: past resample_poly's filter half-length
    (10 * max(up, down) upsampled samples), rounded up to a whole number of `down` blocks."""
    reach = math.ceil(10 * max(up, down) / up) + 1
    return math.ceil(reach / down) * down


def resample(x: np.ndarray, sr: int, chunk: int | None = None) -> np.ndarray:
    """1-D float audio at sr -> float32 at SAMPLE_RATE, resampled in overlapping chunks."""
    if sr == SAMPLE_RATE:
        return x.astype(np.float32, copy=False)
    up, down = ratio(sr)
    chunk = chunk or int(CHUNK_S * sr)
    chunk = max(down, chunk // down * down)
    pad = _pad(up, down)
    n_out = math.ceil(len(x) * up / down)
    out = np.empty(n_out, dtype=np.float32)
    for a in range(0, len(x), chunk):
        lo, hi = max(0, a - pad), min(len(x), a + chunk + pad)
        y = resample_poly(x[lo:hi].astype(np.float64), up, down)
        k0 = (a - lo) * up // down                      # a and lo are multiples of down
        k1 = min(n_out, (a + chunk) * up // down)
        start = a * up // down
        out[start:k1] = y[k0:k0 + k1 - start]
    return out


def info(path: Path) -> tuple[int, int]:
    """(sample rate, frames) from the file header."""
    i = sf.info(str(path))
    return i.samplerate, i.frames


def n_samples_2k(path: Path) -> int:
    """The file's length in samples once at 2 kHz."""
    sr, frames = info(path)
    up, down = ratio(sr)
    return math.ceil(frames * up / down)


def read_2k(path: Path) -> np.ndarray:
    """The whole file's first channel at 2 kHz, float32."""
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return resample(x[:, 0], sr)


def window(x: np.ndarray, start: int, n: int) -> np.ndarray:
    """x[start : start + n], zero where it runs past either end of x."""
    out = np.zeros(n, dtype=np.float32)
    lo, hi = max(0, start), min(len(x), start + n)
    if hi > lo:
        out[lo - start:hi - start] = x[lo:hi]
    return out


class AudioStore:
    """Random access to many files at 2 kHz, for training. A file already at 2 kHz is read in
    place; any other is resampled once into cache_dir (a 2 kHz float WAV) and read from there."""

    def __init__(self, files: dict[str, Path], cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.paths = {name: self._prepare(name, Path(p)) for name, p in files.items()}
        self.n_samples = {name: sf.info(str(p)).frames for name, p in self.paths.items()}

    def _prepare(self, name: str, path: Path) -> Path:
        sr, _ = info(path)
        if sr == SAMPLE_RATE and sf.info(str(path)).channels == 1:
            return path
        out = self.cache_dir / f"{name}.wav"
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(".part.wav")
            sf.write(str(tmp), read_2k(path), SAMPLE_RATE, subtype="FLOAT")
            tmp.rename(out)
        return out

    def read(self, name: str, start: int, n: int) -> np.ndarray:
        """n samples from sample `start` of file `name` at 2 kHz, zero outside the file."""
        total = self.n_samples[name]
        lo, hi = max(0, start), min(total, start + n)
        out = np.zeros(n, dtype=np.float32)
        if hi > lo:
            x, _ = sf.read(str(self.paths[name]), dtype="float32", start=lo, stop=hi,
                           always_2d=True)
            out[lo - start:hi - start] = x[:, 0]
        return out
