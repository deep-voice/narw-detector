"""The model's input: a 0-500 Hz spectrogram of 2 kHz audio, rendered per 10 s slot.

One STFT recipe, matching scipy.signal.spectrogram(window="hann", nperseg=256, noverlap=192,
mode="magnitude"): a periodic Hann window, no centring, the mean of every frame subtracted
(scipy's detrend="constant"), density scaling. Rows run from 500 Hz (row 0) down to 0 Hz.

A window is cut into 10 s slots; each slot gets its own STFT (309 frames) and the slots are
joined in time. The dB image (20 log10(magnitude + 1e-6)) is then standardised over the whole
window (per-clip z-score).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import Tensor

DB_FLOOR = 1e-6


@dataclass(frozen=True)
class RenderSpec:
    """The render geometry. The defaults are the released model's; a config file stores them."""
    sample_rate: int = 2000
    n_fft: int = 256
    hop: int = 64
    f_min: float = 0.0
    f_max: float = 500.0
    slot_s: float = 10.0

    @property
    def n_bins(self) -> int:
        """Frequency rows inside [f_min, f_max] (65 for the defaults)."""
        return int(self.band().sum())

    @property
    def n_slot(self) -> int:
        """Samples in one slot (20,000)."""
        return int(round(self.slot_s * self.sample_rate))

    @property
    def frames_per_slot(self) -> int:
        """STFT frames in one slot (309)."""
        return (self.n_slot - self.n_fft) // self.hop + 1

    def n_frames(self, n_samples: int) -> int:
        """Frames of a window of n_samples (a whole number of slots)."""
        assert n_samples % self.n_slot == 0, (n_samples, self.n_slot)
        return n_samples // self.n_slot * self.frames_per_slot

    def band(self, device: torch.device | None = None) -> Tensor:
        freqs = torch.arange(self.n_fft // 2 + 1, device=device) * (self.sample_rate / self.n_fft)
        return (freqs >= self.f_min) & (freqs <= self.f_max)

    def to_dict(self) -> dict:
        return asdict(self)


def stft_mag(x: Tensor, spec: RenderSpec) -> Tensor:
    """(B, N) float32 audio -> (B, F, T) magnitude, high frequency on row 0."""
    assert x.ndim == 2 and x.dtype == torch.float32, (x.shape, x.dtype)
    win = torch.hann_window(spec.n_fft, periodic=True, device=x.device)
    frames = x.unfold(-1, spec.n_fft, spec.hop)
    frames = frames - frames.mean(-1, keepdim=True)
    scale = math.sqrt(1.0 / (spec.sample_rate * float((win ** 2).sum())))
    mag = torch.fft.rfft(frames * win, dim=-1).abs() * scale
    return mag.transpose(1, 2)[:, spec.band(x.device)].flip(1)


def slotted_mag(x: Tensor, spec: RenderSpec) -> Tensor:
    """(B, N) -> (B, F, T): each slot STFT'd on its own, the slots joined in time."""
    b, n = x.shape
    n_slots = n // spec.n_slot
    assert n_slots * spec.n_slot == n, (n, spec.n_slot)
    mag = stft_mag(x.reshape(b * n_slots, spec.n_slot), spec)            # (B*S, F, t)
    return mag.reshape(b, n_slots, *mag.shape[1:]).permute(0, 2, 1, 3).reshape(b, mag.shape[1], -1)


def per_clip_z(db: Tensor) -> Tensor:
    """Standardise each item's (F, T) image over all its pixels."""
    sd = db.std((1, 2), keepdim=True, unbiased=False)
    return (db - db.mean((1, 2), keepdim=True)) / (sd + 1e-5)


def render(audio: Tensor, spec: RenderSpec, gain_db: Tensor | None = None) -> Tensor:
    """audio (B, N) float32, N a whole number of slots -> the model input (B, 1, F, T).

    gain_db (B, F): a per-item filter in dB on the band's rows (row order), applied to the
    magnitude before the dB step - FilterAugment, training only."""
    with torch.autocast(device_type=audio.device.type, enabled=False):
        mag = slotted_mag(audio.float(), spec)
        if gain_db is not None:
            assert gain_db.shape == mag.shape[:2], (gain_db.shape, mag.shape)
            mag = mag * 10.0 ** (gain_db[:, :, None] / 20.0)
        out = per_clip_z(20.0 * torch.log10(mag + DB_FLOOR))[:, None]
    assert out.shape == (audio.shape[0], 1, spec.n_bins, spec.n_frames(audio.shape[1])), out.shape
    return out
