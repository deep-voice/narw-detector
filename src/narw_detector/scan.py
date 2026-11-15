"""Scan a whole recording: every 60 s window at a 30 s hop, decoded and merged.

Window starts are 0, 30, 60, ... up to the last start that still covers the file's end; that last
window is zero-padded past the end. Each window is rendered and run through the network, its
frame probabilities are decoded at the event floor (narw_detector.events), and the events of all
windows are moved to file seconds and merged. The merge comes BEFORE the threshold: thresholding
the merged events at tau gives the detections.
"""
from __future__ import annotations

import numpy as np
import torch

from narw_detector.audio import SAMPLE_RATE, window
from narw_detector.events import EVENT_FLOOR, Event, merge, window_events
from narw_detector.spectrogram import RenderSpec, render

WINDOW_S, HOP_S = 60.0, 30.0
BATCH = 16


def window_starts(n_samples: int, sr: int = SAMPLE_RATE, window_s: float = WINDOW_S,
                  hop_s: float = HOP_S) -> np.ndarray:
    """Window starts in seconds: arange(0, max(duration - window_s, 0) + hop_s, hop_s)."""
    return np.arange(0.0, max(n_samples / sr - window_s, 0.0) + hop_s, hop_s)


@torch.no_grad()
def window_probs(net: torch.nn.Module, audio: torch.Tensor, spec: RenderSpec) -> np.ndarray:
    """(B, N) float32 windows on the net's device -> (B, T') frame probabilities."""
    return torch.sigmoid(net(render(audio, spec))).cpu().numpy()


@torch.no_grad()
def scan_events(net: torch.nn.Module, x: np.ndarray, spec: RenderSpec, device: str,
                floor: float = EVENT_FLOOR, batch: int = BATCH) -> list[Event]:
    """One file's merged events (t0, t1, confidence) in file seconds, every one at or above the
    floor - threshold them at tau for detections. x is the file at 2 kHz, float32."""
    n_window = int(round(WINDOW_S * spec.sample_rate))
    n_in = spec.n_frames(n_window)
    starts = window_starts(len(x), spec.sample_rate)
    events: list[Event] = []
    for i in range(0, len(starts), batch):
        chunk = starts[i:i + batch]
        audio = np.stack([window(x, int(s * spec.sample_rate), n_window) for s in chunk])
        probs = window_probs(net, torch.from_numpy(audio).to(device), spec)
        for s, p in zip(chunk, probs, strict=True):
            events += [(s + a, s + b, c) for a, b, c in window_events(p, n_in, floor)]
    return merge(events)


def detections(events: list[Event], tau: float) -> list[Event]:
    """The merged events with confidence >= tau."""
    return [e for e in events if e[2] >= tau]
