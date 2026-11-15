"""Fixed-length shape descriptors for detection-stream events.

To compare a unit with its neighbours, every event is reduced to the same small description.
An LFDCS pitch track (track_dt / track_freq, typically about 6 points) gives:

* shape: the contour's log2 frequency at N_POINTS evenly spaced fractions of the unit's
  duration, minus its mean. It is centred but not scaled, so an upsweep over one octave and one
  over a third of an octave stay different.
* logf_med: the median log2 frequency (the register).
* logdur: log2 of the duration in seconds.

The similarity of two events is the cosine of their shapes times a register term and a duration
term, so it is near 1 only when shape, band and length all match. Frequencies are compared in
octaves, so an interval counts the same at any register.
"""
from __future__ import annotations

import numpy as np

N_POINTS = 16
# Scales of the register and duration terms, in octaves: half an octave of register difference,
# or a factor of 2 in duration, multiplies the similarity by 1/e.
REGISTER_SCALE_OCT = 0.5
DURATION_SCALE_OCT = 1.0
_EPS = 1e-9


def resample_track(dt: np.ndarray, freq: np.ndarray, n_points: int = N_POINTS) -> np.ndarray:
    """One contour -> `n_points` log2-frequencies at uniform fractions of its duration.

    A single-point track (LFDCS writes some) resamples to a constant, which centres to the zero
    vector: a flat shape, matched on register and duration alone.
    """
    dt, freq = np.asarray(dt, dtype=np.float64), np.asarray(freq, dtype=np.float64)
    assert dt.shape == freq.shape and dt.ndim == 1 and len(dt) >= 1
    logf = np.log2(np.maximum(freq, 1.0))
    if len(dt) == 1:
        return np.full(n_points, logf[0])
    order = np.argsort(dt)
    dt, logf = dt[order], logf[order]
    span = dt[-1] - dt[0]
    if span <= 0:
        return np.full(n_points, logf.mean())
    grid = dt[0] + np.linspace(0.0, span, n_points)
    return np.interp(grid, dt, logf)


def descriptor_table(tracks_dt: list, tracks_freq: list, durations: np.ndarray,
                     n_points: int = N_POINTS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-event (shapes, logf_med, logdur) for a whole stream.

    shapes is (n_events, n_points), centred per row. The durations are LFDCS's duration column
    rather than the track span, since LFDCS measures a duration even for single-point tracks.
    """
    assert len(tracks_dt) == len(tracks_freq) == len(durations)
    raw = np.stack([resample_track(dt, f, n_points)
                    for dt, f in zip(tracks_dt, tracks_freq, strict=True)])
    logf_med = np.median(raw, axis=1)
    shapes = raw - raw.mean(axis=1, keepdims=True)
    logdur = np.log2(np.maximum(np.asarray(durations, dtype=np.float64), 1e-3))
    return shapes, logf_med, logdur


def similarity_matrix(shapes_a: np.ndarray, logf_a: np.ndarray, logdur_a: np.ndarray,
                      shapes_b: np.ndarray, logf_b: np.ndarray, logdur_b: np.ndarray
                      ) -> np.ndarray:
    """(n_a, n_b) similarity in [0, 1]: shape cosine x register term x duration term.

    Two flat contours have no shape to compare, so their cosine is set to 1 and they are matched
    on register and duration alone. A flat contour against a shaped one gets 0.
    """
    norm_a = np.linalg.norm(shapes_a, axis=1)
    norm_b = np.linalg.norm(shapes_b, axis=1)
    cos = shapes_a @ shapes_b.T / (np.outer(norm_a, norm_b) + _EPS)
    flat_a, flat_b = norm_a < _EPS, norm_b < _EPS
    both_flat = np.outer(flat_a, flat_b)
    one_flat = np.outer(flat_a, ~flat_b) | np.outer(~flat_a, flat_b)
    cos = np.where(both_flat, 1.0, np.where(one_flat, 0.0, cos))
    cos = np.clip(cos, 0.0, 1.0)          # opposed contours count as no match

    reg = np.exp(-np.abs(logf_a[:, None] - logf_b[None, :]) / REGISTER_SCALE_OCT)
    dur = np.exp(-np.abs(logdur_a[:, None] - logdur_b[None, :]) / DURATION_SCALE_OCT)
    return cos * reg * dur
