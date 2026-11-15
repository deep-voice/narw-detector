"""narw-train: train the detector on your own annotated recordings.

    narw-train --audio TRAIN_DIR --annotations TRAIN_ANN \\
               --val-audio VAL_DIR --val-annotations VAL_ANN --out OUT_DIR \\
               [--hard-negatives HARD.csv --song-positives SONG.csv]

The recipe is the released model's (see the README): 60 s windows of six 10 s slots, each
training tile placed in one slot (narw_detector.windows); focal loss on that slot only;
FilterAugment and SpecAugment; AdamW, learning rate 1e-3 (constant), weight decay 1e-4; batch 64
(32 per step x 2 accumulated); at most 40 epochs.

After every epoch the model scans the validation recordings end to end (they must be fully
annotated) and measures recall at the lowest threshold that keeps false detections to 2 per
scanned hour (narw_detector.evaluate). The epoch with the best recall is kept, and training stops
after 7 epochs without improvement. OUT_DIR/model holds the best epoch's weights, with that
epoch's threshold as the operating point. OUT_DIR/history.csv logs every epoch.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from narw_detector.annotations import durations_s, list_recordings, read_annotations
from narw_detector.audio import AudioStore
from narw_detector.augment import filter_augment_gains, spec_augment
from narw_detector.dataset import EpochShuffleSampler, WindowDataset
from narw_detector.evaluate import score_scan
from narw_detector.hard_negatives import hard_negative_tiles, song_positive_tiles
from narw_detector.loss import masked_focal_bce, target_slot_mask
from narw_detector.model import DualStreamSed
from narw_detector.scan import scan_events
from narw_detector.spectrogram import RenderSpec, render
from narw_detector.weights import pick_device, save_model
from narw_detector.windows import lattice_tiles, merge_tiles, place_windows

WIDTH, LR, WEIGHT_DECAY = 32, 1e-3, 1e-4
BATCH = 64
TILE_SEED = 0                     # the tile lattice's background draw, fixed as in the original


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def training_step(net: DualStreamSed, batch: dict, spec: RenderSpec) -> torch.Tensor:
    """One batch's loss with the training augmentations, in the original's draw order:
    FilterAugment's gains, the render, SpecAugment, the forward pass."""
    audio, target, slot = batch["audio"], batch["target"], batch["slot"]
    gain = filter_augment_gains(len(audio), spec.n_bins, audio.device)
    x = spec_augment(render(audio, spec, gain_db=gain))
    mask = target_slot_mask(slot, target.shape[-1], spec.frames_per_slot)
    return masked_focal_bce(net(x), target, mask)


def train_epoch(net: DualStreamSed, loader: DataLoader, opt: torch.optim.Optimizer,
                spec: RenderSpec, accumulate: int, device: str) -> float:
    """One epoch; an optimizer step every `accumulate` batches (and at the epoch's end)."""
    net.train()
    losses = []
    opt.zero_grad()
    for k, batch in enumerate(tqdm(loader, desc="train", unit="batch", mininterval=30,
                                   ascii=True)):
        batch = {key: v.to(device) for key, v in batch.items()}
        loss = training_step(net, batch, spec)
        (loss / accumulate).backward()
        losses.append(float(loss))
        if (k + 1) % accumulate == 0:
            opt.step()
            opt.zero_grad()
    if len(loader) % accumulate:
        opt.step()
        opt.zero_grad()
    return float(np.mean(losses))


def scan_store(net: DualStreamSed, store: AudioStore, spec: RenderSpec,
               device: str) -> dict[str, list]:
    """Every recording in the store scanned end to end: {file: merged events}."""
    net.eval()
    return {name: scan_events(net, store.read(name, 0, n), spec, device)
            for name, n in tqdm(store.n_samples.items(), desc="val scan", unit="file",
                                mininterval=30, ascii=True)}


def training_windows(audio: Path, annotations: Path, hard_negatives: Path | None,
                     song_positives: Path | None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(placed windows, annotations, recordings) of the training set."""
    recordings = list_recordings(audio)
    events = read_annotations(annotations, recordings)
    durations = durations_s(recordings)
    tiles = [lattice_tiles(events, durations, TILE_SEED)]
    if hard_negatives is not None:
        tiles.append(hard_negative_tiles(hard_negatives, durations))
    if song_positives is not None:
        tiles.append(song_positive_tiles(song_positives, durations))
    windows = place_windows(merge_tiles(*tiles), events, durations)
    print(windows.groupby("role").agg(windows=("weight", "size"), weight=("weight", "sum"))
          .to_string())
    return windows, events, recordings


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audio", type=Path, required=True, help="training recordings folder")
    p.add_argument("--annotations", type=Path, required=True,
                   help="training annotations: Raven selection table(s) or a CSV")
    p.add_argument("--val-audio", type=Path, required=True,
                   help="validation recordings folder (fully annotated, other sites/dates)")
    p.add_argument("--val-annotations", type=Path, required=True)
    p.add_argument("--hard-negatives", type=Path, default=None,
                   help="optional CSV of 10 s hard-negative tiles (file, start, end), e.g. "
                        "narw-mine's")
    p.add_argument("--song-positives", type=Path, default=None,
                   help="optional CSV of 10 s tiles of upcalls inside song, trained at weight 4 "
                        "(narw-mine's)")
    p.add_argument("--out", type=Path, required=True, help="output folder")
    p.add_argument("--epochs", type=int, default=40, help="most epochs (default 40)")
    p.add_argument("--patience", type=int, default=7,
                   help="stop after this many epochs without a better validation recall "
                        "(default 7)")
    p.add_argument("--fp-per-hour", type=float, default=2.0,
                   help="false detections per hour at which validation recall is measured "
                        "(default 2)")
    p.add_argument("--accumulate", type=int, default=2,
                   help="batches of 64 / K accumulated per step (default 2: 32 per step)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto",
                   help="cuda, cpu or mps; auto (the default) is cuda if available, else cpu")
    p.add_argument("--num-workers", type=int, default=4)
    return p


def main(argv: list[str] | None = None) -> None:
    p = build_parser()
    a = p.parse_args(argv)
    if BATCH % a.accumulate:
        p.error(f"--accumulate {a.accumulate} must divide the batch of {BATCH}")
    seed_everything(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    device, spec = pick_device(a.device), RenderSpec()
    windows, events, recordings = training_windows(a.audio, a.annotations,
                                                   a.hard_negatives, a.song_positives)
    val_recordings = list_recordings(a.val_audio)
    overlap = sorted(set(recordings) & set(val_recordings))
    if overlap:
        raise ValueError(f"recordings in both training and validation: {overlap[:5]}")
    val_events = read_annotations(a.val_annotations, val_recordings)
    if not (val_events.label == "upcall").any():
        raise ValueError(f"{a.val_annotations}: the validation annotations hold no upcall")
    store = AudioStore(recordings, a.out / "audio_2k")
    val_store = AudioStore(val_recordings, a.out / "audio_2k")
    val_durations = durations_s(val_recordings)

    ds = WindowDataset(windows, events, store, spec)
    sampler = EpochShuffleSampler(windows.weight.to_numpy(), a.seed)
    loader = DataLoader(ds, batch_size=BATCH // a.accumulate, sampler=sampler, drop_last=True,
                        num_workers=a.num_workers, persistent_workers=a.num_workers > 0)
    net = DualStreamSed(width=WIDTH).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    history, best, best_epoch = [], -1.0, -1
    for epoch in range(a.epochs):
        sampler.set_epoch(epoch)
        loss = train_epoch(net, loader, opt, spec, a.accumulate, device)
        scores = score_scan(scan_store(net, val_store, spec, device), val_events, val_durations)
        val = scores.selection_scores(a.fp_per_hour)
        history.append({"epoch": epoch, "train_loss": loss, **val})
        pd.DataFrame(history).to_csv(a.out / "history.csv", index=False)
        print(f"epoch {epoch}: loss {loss:.4f} | recall {val['recall_iou0.5']:.4f} (IoU 0.5) at "
              f"{val['fp_per_hour']:.2f} false detections/h, tau {val['tau']:.4f}"
              f"{'' if val['reachable'] else ' (rate unreachable: every event counted)'}")
        if val["recall_iou0.5"] > best:
            best, best_epoch = val["recall_iou0.5"], epoch
            save_model(a.out / "model", net, WIDTH, spec, val["tau"], notes={
                "tau_source": f"validation scan, {a.fp_per_hour:g} false detections per hour",
                "best_epoch": epoch, "val_recall_iou0.5": val["recall_iou0.5"],
                "val_fp_per_hour": val["fp_per_hour"]})
        elif epoch - best_epoch >= a.patience:
            print(f"no better validation recall for {a.patience} epochs: stopping")
            break
    print(f"best epoch {best_epoch}: recall {best:.4f} -> {a.out / 'model'}")


if __name__ == "__main__":
    main()
