"""The detector network: a dual-stream frame-level sound event detection (SED) model.

Input  (B, 1, 65, T): the spectrogram of a window of 10 s slots (309 frames per slot).
Output (B, T'): per-frame logits of "an upcall is active", T' = (T // 2) // 2 (463 for 60 s).

Two streams read the same window:
  * local   - a frequency-dynamic CNN (FDY) run on each 10 s slot on its own;
  * context - a second FDY CNN plus a bidirectional GRU over the whole window.
Each slot's local features are joined frame by frame to the context features cropped to that
slot, a bidirectional GRU runs over each slot alone, and a linear head gives the logits.

Frequency-dynamic convolution (Nam et al., DCASE 2022): K basis kernels mixed by a softmax
attention that depends on frequency, so the effective kernel changes along the frequency axis.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SLOT_FRAMES = 309                 # input frames of one 10 s slot (2 kHz, n_fft 256, hop 64)


def slot_output_slice(n_in: int, n_out: int, slot_start_in: int, slot_len_in: int) -> slice:
    """The output frames belonging to a slot of input frames [slot_start_in, + slot_len_in).

    Output frame i covers input frames [i*s, i*s + s), s = n_in // n_out (4); it belongs to the
    slot holding the centre i*s + s/2 of that span, so a span split exactly in half goes to the
    later slot and the slots partition the output (77 or 78 frames per 10 s slot)."""
    s = n_in // n_out
    assert s >= 1, (n_in, n_out)
    centre = np.arange(n_out) * s + s / 2
    idx = np.flatnonzero((centre >= slot_start_in) & (centre < slot_start_in + slot_len_in))
    assert len(idx) > 0, (n_in, n_out, slot_start_in, slot_len_in)
    return slice(int(idx[0]), int(idx[-1]) + 1)


def slot_lengths(n_in: int, n_out: int, n_slots: int) -> list[int]:
    """Output frames per slot for n_slots equal slots; the slices must partition [0, n_out)."""
    assert n_in % n_slots == 0, (n_in, n_slots)
    slot_len = n_in // n_slots
    slices = [slot_output_slice(n_in, n_out, j * slot_len, slot_len) for j in range(n_slots)]
    stops = [sl.start for sl in slices[1:]] + [n_out]
    assert slices[0].start == 0, slices
    assert all(sl.stop == stop for sl, stop in zip(slices, stops, strict=True)), slices
    return [sl.stop - sl.start for sl in slices]


def _block(in_ch: int, out_ch: int, pool: tuple[int, int]) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch),
                         nn.ReLU(inplace=True), nn.MaxPool2d(pool))


class FDYConv2d(nn.Module):
    """Frequency-dynamic convolution: n_basis kernels, mixed per frequency row by a softmax
    attention computed from the time-averaged input."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 3, n_basis: int = 4):
        super().__init__()
        self.n_basis, self.out_ch = n_basis, out_ch
        self.conv = nn.Conv2d(in_ch, out_ch * n_basis, k, padding=k // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.att = nn.Sequential(nn.AdaptiveAvgPool2d((None, 1)), nn.Conv2d(in_ch, n_basis, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, freq, t = x.shape
        bases = self.conv(x).view(b, self.n_basis, self.out_ch, freq, t)  # (B, K, out, F, T)
        att = torch.softmax(self.att(x), dim=1)                          # (B, K, F, 1)
        out = (bases * att.unsqueeze(2)).sum(1)                          # (B, out, F, T)
        return F.relu(self.bn(out), inplace=True)


class FreqCNN(nn.Module):
    """Three conv blocks, the first frequency-dynamic: frequency pooled to 1, time pooled by 4
    (floor) -> (B, 4 * width, T // 4)."""

    def __init__(self, width: int, in_channels: int = 1):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            nn.Sequential(FDYConv2d(in_channels, w), nn.MaxPool2d((2, 2))),
            _block(w, 2 * w, (2, 2)),
            _block(2 * w, 4 * w, (2, 1)))
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))
        self.out_ch = 4 * w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.freq_pool(self.features(x)).squeeze(2)              # (B, 4w, T')


class DualStreamSed(nn.Module):
    """The released architecture, described in the module docstring. A slot's 77 local frames
    are resampled (nearest) onto its 77 or 78 output frames, so the output sits on the context
    stream's frame grid."""

    def __init__(self, width: int = 32, in_channels: int = 1, dropout: float = 0.2):
        super().__init__()
        self.local = FreqCNN(width, in_channels)
        self.context = FreqCNN(width, in_channels)
        self.context_gru = nn.GRU(self.context.out_ch, width, batch_first=True, bidirectional=True)
        self.gru = nn.GRU(self.local.out_ch + 2 * width, width, batch_first=True,
                          bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * width, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, f"expected (B, C, F, T), got {tuple(x.shape)}"
        b, c, f, n_in = x.shape
        n_in = int(n_in)                    # a plain int, so the model can be traced
        assert n_in % SLOT_FRAMES == 0, f"{n_in} input frames: not whole {SLOT_FRAMES}-frame slots"
        n_slots = n_in // SLOT_FRAMES
        slots = x.reshape(b, c, f, n_slots, SLOT_FRAMES).permute(0, 3, 1, 2, 4)
        local = self.local(slots.reshape(b * n_slots, c, f, SLOT_FRAMES))   # (B*S, 4w, 77)
        local = local.reshape(b, n_slots, *local.shape[1:])
        context, _ = self.context_gru(self.context(x).transpose(1, 2))      # (B, T', 2w)
        n_out = int(context.shape[1])
        outs = []
        for j in range(n_slots):
            sl = slot_output_slice(n_in, n_out, j * SLOT_FRAMES, SLOT_FRAMES)
            loc = F.interpolate(local[:, j], size=sl.stop - sl.start, mode="nearest")
            out, _ = self.gru(torch.cat([loc.transpose(1, 2), context[:, sl]], -1))
            outs.append(out)
        out = torch.cat(outs, 1)
        assert out.shape[1] == context.shape[1], (out.shape, context.shape)
        return self.head(out).squeeze(-1)
