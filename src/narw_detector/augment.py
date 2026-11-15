"""Training-time augmentation of the spectrogram, applied to every training batch.

FilterAugment (Nam et al. 2022, the reference code's linear type): per batch 3-5 frequency bands
with random boundaries at least 6 rows apart; per item a gain drawn from U[-6, 6] dB at every
boundary, linear across each band, applied to the magnitude before the dB step.

SpecAugment: per item 2 frequency masks of U[1, 8] rows and 2 time masks per 10 s of U[1, 30]
columns (12 for a 60 s window), zero-filled after the per-clip standardisation.
"""
from __future__ import annotations

import torch
from torch import Tensor

FREQ_MASKS, FREQ_MASK_MAX = 2, 8
TIME_MASKS_PER_10S, TIME_MASK_MAX, FRAMES_PER_10S = 2, 30, 309
FILTER_DB, FILTER_BANDS, FILTER_MIN_BW = 6.0, (3, 6), 6       # db range, n_band in [3, 6), min bw


def _band_mask(n: int, n_masks: int, w_max: int, batch: int, device: torch.device) -> Tensor:
    """(B, n) bool: n_masks random bands of width U[1, w_max] per item, starts in [0, n - w)."""
    w = torch.randint(1, w_max + 1, (batch, n_masks), device=device)
    start = (torch.rand(batch, n_masks, device=device) * (n - w).clamp(min=1)).long()
    pos = torch.arange(n, device=device)
    hit = (pos >= start[..., None]) & (pos < (start + w)[..., None])      # (B, n_masks, n)
    return hit.any(1)


def spec_augment(x: Tensor) -> Tensor:
    """Zero whole frequency rows and time columns of a (B, C, F, T) batch, per item."""
    b, _, f, t = x.shape
    n_time = TIME_MASKS_PER_10S * t // FRAMES_PER_10S
    keep = ~_band_mask(f, FREQ_MASKS, FREQ_MASK_MAX, b, x.device)[:, None, :, None]
    if n_time > 0:
        keep = keep & ~_band_mask(t, n_time, TIME_MASK_MAX, b, x.device)[:, None, None, :]
    return x * keep


def filter_augment_gains(batch: int, n_freq: int, device: torch.device) -> Tensor:
    """FilterAugment's per-item filter -> (B, n_freq) gains in dB, linear between band edges."""
    n_band = int(torch.randint(FILTER_BANDS[0], FILTER_BANDS[1], (1,)))
    min_bw = FILTER_MIN_BW
    while n_freq - n_band * min_bw + 1 < 0:
        min_bw -= 1
    inner = (torch.sort(torch.randint(0, n_freq - n_band * min_bw + 1, (n_band - 1,)))[0]
             + torch.arange(1, n_band) * min_bw)
    edges = [0, *inner.tolist(), n_freq]
    at = torch.rand(batch, n_band + 1, device=device) * 2 * FILTER_DB - FILTER_DB
    gain = torch.empty(batch, n_freq, device=device)
    for i in range(n_band):
        w = torch.linspace(0, 1, edges[i + 1] - edges[i], device=device)
        gain[:, edges[i]:edges[i + 1]] = at[:, i:i + 1] * (1 - w) + at[:, i + 1:i + 2] * w
    return gain
