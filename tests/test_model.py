"""The network's three tests (shape, one training step, overfit) and its slot geometry."""
import time

import torch

from narw_detector.loss import masked_focal_bce
from narw_detector.model import SLOT_FRAMES, DualStreamSed, slot_lengths
from narw_detector.spectrogram import RenderSpec
from narw_detector.train import training_step


def test_shape_and_size_of_the_released_architecture():
    net = DualStreamSed(width=32)
    assert sum(p.numel() for p in net.parameters()) == 262_481
    out = net(torch.randn(2, 1, 65, 6 * SLOT_FRAMES))
    assert out.shape == (2, 463), out.shape


def test_smoke_one_training_step_gives_a_finite_loss():
    torch.manual_seed(0)
    net, spec = DualStreamSed(width=8), RenderSpec()
    batch = {"audio": 0.01 * torch.randn(2, 120_000),
             "target": (torch.rand(2, 6 * SLOT_FRAMES) > 0.9).float(),
             "slot": torch.tensor([0, 3])}
    loss = training_step(net, batch, spec)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)


def test_overfit_a_tiny_batch_in_under_30_s():
    """Two one-slot windows, one with a bright bar the targets mark: loss to ~0."""
    torch.manual_seed(0)
    net = DualStreamSed(width=8)
    x = torch.rand(2, 1, 33, SLOT_FRAMES) * 0.1
    x[0, 0, 10:20, 100:200] += 1.0
    y = torch.zeros(2, SLOT_FRAMES)
    y[0, 100:200] = 1.0
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    start = time.time()
    for _ in range(200):
        opt.zero_grad()
        loss = masked_focal_bce(net(x), y, torch.ones_like(y))
        loss.backward()
        opt.step()
    assert float(loss) < 0.01, float(loss)
    assert time.time() - start < 30


def test_output_slots_partition_the_frames():
    assert slot_lengths(6 * SLOT_FRAMES, 463, 6) == [77, 77, 78, 77, 77, 77]
    assert slot_lengths(SLOT_FRAMES, 77, 1) == [77]


def test_a_slot_local_stream_never_reads_another_slot():
    """With the context stream silenced, changing slot 0 leaves slot 2's output unchanged."""
    torch.manual_seed(0)
    net = DualStreamSed(width=8).eval()
    for p in net.context_gru.parameters():
        torch.nn.init.zeros_(p)
    x = torch.randn(1, 1, 65, 3 * SLOT_FRAMES)
    y = x.clone()
    y[..., :SLOT_FRAMES] = torch.randn(1, 1, 65, SLOT_FRAMES)
    with torch.no_grad():
        a, b = net(x), net(y)
    assert torch.equal(a[:, 154:], b[:, 154:])
    assert not torch.equal(a[:, :77], b[:, :77])
