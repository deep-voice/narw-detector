"""From frame probabilities to call events (start, end, confidence).

An event is a run of at least two consecutive output frames with probability above the event
floor (0.4). Its confidence is the run's highest probability. Events are cut at the floor first,
and the operating point tau is applied to their confidence afterwards. tau was chosen with this
floor, so decoding at another floor changes what tau means.

Frame times: within each 10 s slot of n_j output frames (77 or 78), frame i spans
[i / n_j * 10, (i + 1) / n_j * 10) s from the slot's start. A window is decoded as one sequence
over all its slots, so a call that crosses a slot boundary stays one event.
"""
from __future__ import annotations

import functools

import numpy as np

from narw_detector.model import SLOT_FRAMES, slot_lengths

EVENT_FLOOR = 0.4
MIN_FRAMES = 2

Event = tuple[float, float, float]            # (t0_s, t1_s, confidence)


def probs_to_events(probs: np.ndarray, times: tuple[np.ndarray, np.ndarray],
                    floor: float = EVENT_FLOOR, min_frames: int = MIN_FRAMES) -> list[Event]:
    """Runs of frames with prob > floor (at least min_frames long) -> (t0, t1, peak prob);
    times = (t_lo, t_hi) of every frame."""
    above = probs > floor
    events, i, n = [], 0, len(probs)
    while i < n:
        if not above[i]:
            i += 1
            continue
        j = i
        while j < n and above[j]:
            j += 1
        if j - i >= min_frames:
            events.append((float(times[0][i]), float(times[1][j - 1]), float(probs[i:j].max())))
        i = j
    return events


@functools.cache
def frame_times(n_in: int, n_out: int, slot_s: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """[t_lo, t_hi) of each output frame, in seconds from the window's start. Cached per
    geometry, so the arrays are read-only."""
    assert n_in % SLOT_FRAMES == 0, (n_in, SLOT_FRAMES)
    n_slots = n_in // SLOT_FRAMES
    lo, hi = [], []
    for j, n_j in enumerate(slot_lengths(n_in, n_out, n_slots)):
        i = np.arange(n_j, dtype=float)
        lo.append(i / n_j * slot_s + j * slot_s)
        hi.append((i + 1) / n_j * slot_s + j * slot_s)
    t_lo, t_hi = np.concatenate(lo), np.concatenate(hi)
    assert len(t_lo) == n_out, (len(t_lo), n_out)
    t_lo.setflags(write=False)
    t_hi.setflags(write=False)
    return t_lo, t_hi


def window_events(probs: np.ndarray, n_in: int, floor: float = EVENT_FLOOR) -> list[Event]:
    """One window's events in window seconds."""
    return probs_to_events(probs, frame_times(n_in, len(probs)), floor)


def merge(events: list[Event]) -> list[Event]:
    """Union overlapping events, keeping the highest confidence. Scan windows overlap, so the
    same call is usually seen by two windows."""
    out: list[Event] = []
    for t0, t1, c in sorted(events):
        if out and t0 <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], t1), max(out[-1][2], c))
        else:
            out.append((t0, t1, c))
    return out
