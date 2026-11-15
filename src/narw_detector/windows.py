"""Which stretches of audio the model trains on: 10 s tiles, each placed in a 60 s window.

Tiles (the original training set's lattice rule). In every recording with at least one upcall,
10 s tiles start every 5 s from 0:
  * pos - the tile holds at least one whole upcall of positive duration;
  * a tile that overlaps an upcall but holds none whole is skipped;
  * background candidates - tiles that overlap no upcall and no possible; as many are drawn as
    there are pos tiles (without replacement, one random generator over the recordings in name
    order).
Hard-negative tiles (optional, e.g. from narw-mine) are added as given.

Windows. The model reads 60 s: six 10 s slots, and the loss is taken on the slot that holds the
tile. Each tile goes to one of the six slots, picked by a fixed hash of the tile among the
placements whose 60 s window lies inside the recording and overlaps no possible; a tile with no
such placement is dropped. The same tile gets the same placement every run.
"""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

TILE_S, STRIDE_S = 10.0, 5.0
WINDOW_S = 60.0
BG_PER_POS = 1.0
TILE_COLUMNS = ["file", "start_s", "role", "weight"]
WINDOW_COLUMNS = ["file", "start_s", "slot_start_s", "role", "weight"]


def _overlaps(start: np.ndarray, end: np.ndarray, ts: np.ndarray, te: np.ndarray) -> np.ndarray:
    """(tiles, events) bool: event [start, end) overlaps tile [ts, te)."""
    return (start[None, :] < te[:, None]) & (end[None, :] > ts[:, None])


def _file_tiles(ev: pd.DataFrame, dur: float, rng: np.random.Generator) -> pd.DataFrame:
    """One recording's positive tiles, then its draw of background tiles."""
    ts = np.arange(0.0, dur - TILE_S + 1e-6, STRIDE_S) if dur > TILE_S else np.zeros(0)
    te = ts + TILE_S
    up, poss = ev[ev.label == "upcall"], ev[ev.label == "possible"]
    s, e = up.start_s.to_numpy(float), up.end_s.to_numpy(float)
    any_up = _overlaps(s, e, ts, te).any(1)
    whole = ((s[None, :] >= ts[:, None]) & (e[None, :] <= te[:, None]) & (e > s)[None, :]).any(1)
    is_pos = any_up & whole
    near_poss = _overlaps(poss.start_s.to_numpy(float), poss.end_s.to_numpy(float), ts, te).any(1)
    cand = ts[~any_up & ~near_poss]
    n_bg = int(round(BG_PER_POS * int(is_pos.sum())))
    bg = (cand[rng.choice(len(cand), size=min(n_bg, len(cand)), replace=False)]
          if n_bg and len(cand) else np.zeros(0))
    starts = np.concatenate([ts[is_pos], bg])
    roles = ["pos"] * int(is_pos.sum()) + ["bg"] * len(bg)
    return pd.DataFrame({"start_s": starts, "role": roles})


def lattice_tiles(events: pd.DataFrame, durations: pd.Series, seed: int = 0) -> pd.DataFrame:
    """Positive and background tiles of every recording with an upcall."""
    rng = np.random.default_rng(seed)
    parts = [_file_tiles(g, float(durations[f]), rng).assign(file=f)
             for f, g in events.groupby("file", sort=True) if (g.label == "upcall").any()]
    tiles = pd.concat(parts, ignore_index=True).assign(weight=1)[TILE_COLUMNS]
    print(f"tiles: {(tiles.role == 'pos').sum():,} pos, {(tiles.role == 'bg').sum():,} background "
          f"from {len(parts):,} recordings with upcalls")
    return tiles


def merge_tiles(*tile_sets: pd.DataFrame) -> pd.DataFrame:
    """Tile tables joined; a tile listed twice is one row, its weights summed and its first role
    kept."""
    t = pd.concat(tile_sets, ignore_index=True)
    return (t.groupby(["file", "start_s"], sort=False, as_index=False)
            .agg(role=("role", "first"), weight=("weight", "sum"))[TILE_COLUMNS])


def _tile_hash(file: str, start_s: float) -> int:
    return zlib.crc32(f"{file}__{int(start_s)}".encode())


def place_windows(tiles: pd.DataFrame, events: pd.DataFrame, durations: pd.Series) -> pd.DataFrame:
    """Places each tile in a 60 s window: rows file, start_s (the window's start), slot_start_s
    (the tile's offset inside the window, a multiple of 10 s), role, weight."""
    n_slots = round(WINDOW_S / TILE_S)
    offsets = np.arange(-(n_slots - 1), 1, dtype=float) * TILE_S
    s = tiles.start_s.to_numpy(float)
    cand = s[:, None] + offsets[None, :]                                   # (tiles, 6) starts
    dur = durations.reindex(tiles.file).to_numpy(float)
    valid = (cand >= 0) & (cand + WINDOW_S <= dur[:, None])
    poss = events[events.label == "possible"]
    files = tiles.file.to_numpy()
    for f, p in poss.groupby("file"):
        rows = np.flatnonzero(files == f)
        c = cand[rows][:, :, None]
        valid[rows] &= ~((c < p.end_s.to_numpy(float)) & (c + WINDOW_S > p.start_s.to_numpy(float))
                         ).any(2)
    n_valid = valid.sum(1)
    pick = np.full(len(tiles), -1)
    for i in np.flatnonzero(n_valid > 0):
        k = np.random.default_rng(_tile_hash(files[i], s[i])).integers(n_valid[i])
        pick[i] = np.flatnonzero(valid[i])[k]
    keep = pick >= 0
    w0 = cand[np.arange(len(tiles)), np.maximum(pick, 0)][keep]
    out = tiles[keep].assign(start_s=w0, slot_start_s=s[keep] - w0)[WINDOW_COLUMNS]
    print(f"windows: {len(out):,} placed, {int((~keep).sum()):,} tiles dropped (no in-file, "
          f"possible-free {WINDOW_S:g} s window)")
    return out.reset_index(drop=True)
