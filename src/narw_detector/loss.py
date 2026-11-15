"""The training loss: focal binary cross-entropy per output frame, on the target slot only.

Targets live on the input-frame grid (309 frames per 10 s slot); the network outputs about one
frame per four. Targets are max-pooled to the output length and the mask min-pooled (an output
frame touching a masked input frame is masked). The mask is 1 on the item's target slot and 0
elsewhere, so the other five slots are context the model reads but is not scored on.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

FOCAL_GAMMA, FOCAL_ALPHA = 2.0, 0.25


def _pool_to(x: torch.Tensor, n_out: int) -> torch.Tensor:
    return F.adaptive_max_pool1d(x.unsqueeze(1), n_out).squeeze(1)


def target_slot_mask(slots: torch.Tensor, n_in: int, frames_per_slot: int) -> torch.Tensor:
    """(B,) target slot indices -> (B, n_in) input-frame mask, 1 on that slot."""
    frame_slot = torch.arange(n_in, device=slots.device) // frames_per_slot
    return (frame_slot[None, :] == slots[:, None]).float()


def masked_focal_bce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor,
                     gamma: float = FOCAL_GAMMA, alpha: float = FOCAL_ALPHA) -> torch.Tensor:
    """Focal BCE averaged over the unmasked output frames. logits (B, T_out); targets and mask
    (B, T_in), pooled to T_out."""
    n_out = logits.shape[1]
    y = _pool_to(targets, n_out)
    m = 1 - _pool_to(1 - mask, n_out)
    ce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    p = torch.sigmoid(logits)
    pt = torch.where(y == 1, p, 1 - p)
    at = torch.where(y == 1, alpha, 1 - alpha)
    return (at * (1 - pt) ** gamma * ce * m).sum() / m.sum().clamp(min=1)
