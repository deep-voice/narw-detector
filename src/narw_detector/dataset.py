"""Training items: a 60 s window of 2 kHz audio and its frame targets.

Targets follow the render: every 10 s slot has its own 309 STFT frames timed from the slot's
start, so each slot is painted from its own origin and the six slot targets are joined. A call
overlapping the window is clipped to it (a call the window cuts still teaches its in-window part).
Frames of a call [t0, t1) inside a 10 s slot: round(t0 / 10 * 309) up to round(t1 / 10 * 309),
at least one frame.

Each item also names its target slot - the slot holding its tile - where the loss is taken.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from narw_detector.audio import AudioStore
from narw_detector.spectrogram import RenderSpec


def frame_targets(spans: np.ndarray, slot_s: float, n_frames: int) -> np.ndarray:
    """(n, 2) call spans in seconds from the slot's start -> (n_frames,) 0/1 frame targets."""
    y = np.zeros(n_frames, dtype=np.float32)
    if len(spans) == 0:
        return y
    lo = np.round(spans[:, 0] / slot_s * n_frames).astype(np.int64)
    hi = np.maximum(np.round(spans[:, 1] / slot_s * n_frames).astype(np.int64), lo + 1)
    for a, b in zip(lo, hi, strict=True):
        if b > 0 and a < n_frames:
            y[max(0, a):min(n_frames, b)] = 1.0
    return y


def window_targets(spans: np.ndarray, window_s: float, spec: RenderSpec) -> np.ndarray:
    """(n, 2) call spans in seconds from the window's start -> the window's frame targets,
    built slot by slot."""
    inside = spans[(spans[:, 0] < window_s) & (spans[:, 1] > 0)]
    clipped = np.clip(inside, 0.0, window_s)
    n_slots = round(window_s / spec.slot_s)
    return np.concatenate([frame_targets(clipped - j * spec.slot_s, spec.slot_s,
                                         spec.frames_per_slot) for j in range(n_slots)])


class WindowDataset(Dataset):
    """windows (narw_detector.windows.place_windows) -> items {audio (N,), target (T,), slot}."""

    def __init__(self, windows: pd.DataFrame, events: pd.DataFrame, store: AudioStore,
                 spec: RenderSpec, window_s: float = 60.0):
        self.windows = windows.reset_index(drop=True)
        self.store, self.spec, self.window_s = store, spec, window_s
        self.n_window = int(round(window_s * spec.sample_rate))
        up = events[events.label == "upcall"]
        self.calls = {f: g[["start_s", "end_s"]].to_numpy(float) for f, g in up.groupby("file")}

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int) -> dict:
        w = self.windows.iloc[int(i)]
        start = float(w.start_s)
        spans = self.calls.get(w.file, np.zeros((0, 2))) - start
        audio = self.store.read(w.file, int(self.spec.sample_rate * start), self.n_window)
        return {"audio": torch.from_numpy(audio),
                "target": torch.from_numpy(window_targets(spans, self.window_s, self.spec)),
                "slot": int(round(float(w.slot_start_s) / self.spec.slot_s))}


class EpochShuffleSampler(Sampler[int]):
    """Window indices repeated by weight, in a fresh order each epoch drawn from
    torch.Generator().manual_seed(seed + epoch): reproducible from (seed, epoch)."""

    def __init__(self, weights: np.ndarray, seed: int):
        repeated = np.repeat(np.arange(len(weights)), weights)
        self.indices = torch.as_tensor(repeated, dtype=torch.int64)
        self.seed, self.epoch = seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(self.indices[torch.randperm(len(self.indices), generator=g)].tolist())
