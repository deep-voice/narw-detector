"""Synthetic audio helpers. Every test uses generated signals: no real recordings."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import chirp

SR = 2000


def upcall(t0: float, sr: int = SR, dur: float = 1.0) -> tuple[np.ndarray, int]:
    """A 100 -> 200 Hz upsweep of `dur` s and its start sample."""
    t = np.arange(int(dur * sr)) / sr
    return 0.3 * chirp(t, 100.0, dur, 200.0).astype(np.float32), int(t0 * sr)


def recording(seconds: float, calls: list[float], sr: int = SR, seed: int = 0) -> np.ndarray:
    """Low noise plus a 1 s upcall starting at each time in `calls`."""
    rng = np.random.default_rng(seed)
    x = (0.01 * rng.standard_normal(int(seconds * sr))).astype(np.float32)
    for c in calls:
        y, s = upcall(c, sr)
        x[s:s + len(y)] += y
    return x


def write_wav(path: Path, x: np.ndarray, sr: int = SR) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), x, sr, subtype="PCM_16")
    return path
