"""Mined song -> training tiles: hard negatives and song positives.

The rules that built the released model's training set:

Hard negatives. Every member of a qualifying run (narw_detector.mining.sequences) whose LFDCS call
group is right-whale-like or humpback (GOM9 groups 5-10, 15-20, 23-25), at any distance, gives
the 10 s tile on the 5 s grid nearest its centre: the tile centre round((c - 5) / 5) * 5 + 5 for a
unit centred at c s into its file. A tile is dropped when any possible lies within 15 s of the
tile centre (on the continuous clock, so across a file boundary too). One tile per grid position,
named by its most stereotyped unit's pool: "rw" (groups 5-10), "flagged" (15, 18: humpback calls
GOM9 describes as right-whale-like) or "hw" (the rest). Mined tiles are not checked for upcalls:
training targets come from the annotations, so an upcall inside a mined tile is still taught.

At most MAX_HARD_NEGATIVES are kept, stratified: the tiles are split into stereotypy terciles,
and each (pool, tercile) group keeps at most MAX_HARD_NEGATIVES // number-of-groups tiles, a
uniform sample (seed 0). The original also stratified by deployment; recordings carry no
deployment here.

Song positives. A labelled upcall whose centre lies inside a qualifying run (any call group) or
within 30 s of one gives the tile starting at round((start - 5) / 5) * 5. narw-train uses these
tiles at weight 4.

Tiles reaching past their file's end are dropped, and a start below 0 becomes 0. A grid position
that is both a hard negative and a song positive is kept as a song positive.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from narw_detector.mining.descriptors import descriptor_table
from narw_detector.mining.sequences import Run, find_runs, group_voices, run_member_stereotypy

TILE_S, GRID_S = 10.0, 5.0
RW_GROUPS = tuple(range(5, 11))
FLAGGED_GROUPS = (15, 18)
HW_GROUPS = (16, 17, 19, 20, 23, 24, 25)
POSSIBLE_HALF_S = 15.0
SONG_WITHIN_S = 30.0
MAX_HARD_NEGATIVES = 30_000


def grid_tile_start(t: np.ndarray) -> np.ndarray:
    """Start of the tile on the 5 s grid whose centre is nearest t."""
    return np.round((t - TILE_S / 2) / GRID_S) * GRID_S


def stream_descriptors(stream: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Descriptors from the pitch tracks; a detection without a track gets a minfreq -> maxfreq
    pseudo-track over its duration."""
    dts, fqs = [], []
    for dt, fq, lo, hi, dur in zip(stream.track_dt, stream.track_freq, stream.minfreq,
                                   stream.maxfreq, stream.duration, strict=True):
        if dt is None or len(dt) == 0:
            dts.append(np.array([0.0, max(dur, 1e-3)]))
            fqs.append(np.array([lo, hi]))
        else:
            dts.append(np.asarray(dt))
            fqs.append(np.asarray(fq))
    return descriptor_table(dts, fqs, stream.duration.to_numpy())


def mine_runs(stream: pd.DataFrame) -> tuple[list[Run], np.ndarray, np.ndarray]:
    """(qualifying runs, each detection's run id or -1, each run member's stereotypy)."""
    shapes, logf, logdur = stream_descriptors(stream)
    times = stream.stream_s.to_numpy()
    runs = find_runs(times, group_voices(times, shapes, logf, logdur))
    run_id, stereo = np.full(len(stream), -1), np.full(len(stream), np.nan)
    for k, r in enumerate(tqdm(runs, desc="stereotypy", unit="run", mininterval=30, ascii=True)):
        run_id[r.members] = k
        stereo[r.members] = run_member_stereotypy(r, times, shapes, logf, logdur)
    print(f"{len(runs):,} qualifying runs; {int((run_id >= 0).sum()):,} of {len(stream):,} "
          "detections in a run")
    return runs, run_id, stereo


def _near_possible(centre_stream: np.ndarray, possibles: pd.DataFrame,
                   offsets: pd.Series) -> np.ndarray:
    """True where a possible's span meets [centre - 15 s, centre + 15 s] (closed), on the
    continuous clock."""
    if possibles.empty:
        return np.zeros(len(centre_stream), dtype=bool)
    base = offsets.reindex(possibles.file).to_numpy()
    p0 = np.sort(base + possibles.start_s.to_numpy())
    p1 = np.sort(base + possibles.end_s.to_numpy())
    w0, w1 = centre_stream - POSSIBLE_HALF_S, centre_stream + POSSIBLE_HALF_S
    return np.searchsorted(p0, w1, "right") > np.searchsorted(p1, w0, "left")


def mined_tiles(stream: pd.DataFrame, run_id: np.ndarray, stereo: np.ndarray,
                annotations: pd.DataFrame, offsets: pd.Series) -> pd.DataFrame:
    """Hard-negative tiles before the cap: rows file, start_s, pool, stereotypy."""
    group = stream.call_group.to_numpy()
    keep = (run_id >= 0) & np.isin(group, RW_GROUPS + FLAGGED_GROUPS + HW_GROUPS)
    s = stream[keep]
    start = grid_tile_start(s.offset_s.to_numpy() + s.duration.to_numpy() / 2)
    tile_centre = start + TILE_S / 2
    near = _near_possible(s.stream_s.to_numpy() + tile_centre - s.offset_s.to_numpy(),
                          annotations[annotations.label == "possible"], offsets)
    g = group[keep]
    pool = np.select([np.isin(g, RW_GROUPS), np.isin(g, FLAGGED_GROUPS)], ["rw", "flagged"], "hw")
    tiles = pd.DataFrame({"file": s.file.to_numpy(), "start_s": start, "pool": pool,
                          "stereotypy": stereo[keep]})[~near]
    print(f"mined tiles: {int(keep.sum()):,} run members -> {int(near.sum()):,} dropped near a "
          f"possible -> {tiles[['file', 'start_s']].drop_duplicates().shape[0]:,} tiles")
    return (tiles.sort_values("stereotypy", ascending=False, kind="stable")
            .drop_duplicates(["file", "start_s"]).reset_index(drop=True))


def cap_tiles(tiles: pd.DataFrame, max_n: int = MAX_HARD_NEGATIVES, seed: int = 0) -> pd.DataFrame:
    """At most max_n tiles, sampled evenly across (pool, stereotypy tercile) groups."""
    if len(tiles) < 3:
        return tiles
    t = tiles.assign(tercile=pd.qcut(tiles.stereotypy.rank(method="first"), 3, labels=False))
    groups = t.groupby(["pool", "tercile"], sort=True)
    per = max(1, max_n // groups.ngroups)
    out = pd.concat([g.sample(min(per, len(g)), random_state=seed) for _, g in groups])
    print(f"hard negatives: {len(out):,} of {len(tiles):,} tiles kept ({groups.ngroups} pool x "
          f"stereotypy groups, at most {per:,} each)")
    return out.drop(columns="tercile").reset_index(drop=True)


def distance_to_runs(t: np.ndarray, runs: list[Run]) -> np.ndarray:
    """Each time's distance in seconds to the nearest run span [t0, t1] (0 inside one)."""
    if not runs:
        return np.full(len(t), np.inf)
    t0 = np.array([r.t0 for r in runs])[None, :]
    t1 = np.array([r.t1 for r in runs])[None, :]
    return np.clip(np.maximum(t0 - t[:, None], t[:, None] - t1), 0.0, None).min(1)


def song_positive_tiles(annotations: pd.DataFrame, runs: list[Run],
                        offsets: pd.Series) -> pd.DataFrame:
    """Song-positive tiles: rows file, start_s."""
    up = annotations[annotations.label == "upcall"]
    centre = offsets.reindex(up.file).to_numpy() + (up.start_s + up.end_s).to_numpy() / 2
    near = distance_to_runs(centre, runs) <= SONG_WITHIN_S
    tiles = pd.DataFrame({"file": up.file.to_numpy()[near],
                          "start_s": grid_tile_start(up.start_s.to_numpy()[near])})
    print(f"song positives: {int(near.sum()):,} of {len(up):,} upcalls within "
          f"{SONG_WITHIN_S:g} s of a run")
    return tiles.drop_duplicates().reset_index(drop=True)


def finish(tiles: pd.DataFrame, durations: pd.Series) -> pd.DataFrame:
    """Drop tiles past their file's end, clamp starts below 0, one row per tile; adds end_s."""
    past = tiles.start_s.to_numpy() + TILE_S > durations.reindex(tiles.file).to_numpy()
    out = tiles[~past].assign(start_s=lambda d: d.start_s.clip(lower=0.0))
    out = out.drop_duplicates(["file", "start_s"]).assign(end_s=lambda d: d.start_s + TILE_S)
    return out.sort_values(["file", "start_s"]).reset_index(drop=True)
