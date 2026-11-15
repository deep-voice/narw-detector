"""Extra training tiles from a table: hard negatives and song positives.

A CSV with columns file, start, end (seconds from the start of the file), one 10 s tile per row
(end = start + 10), inside its recording - narw-mine writes both tables in this format. Hard
negatives are sounds the model should learn to ignore (e.g. humpback song); song positives are
labelled upcalls inside song, trained at weight 4. Either way the tile's targets come from the
annotations: the table only says which audio to train on, and how often.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from narw_detector.windows import TILE_COLUMNS, TILE_S

SONG_POSITIVE_WEIGHT = 4


def read_tiles(path: Path, durations: pd.Series, role: str, weight: int) -> pd.DataFrame:
    """A tile table -> rows file, start_s, role, weight."""
    t = pd.read_csv(path)
    if not {"file", "start", "end"} <= set(t.columns):
        raise ValueError(f"{path}: needs columns file, start, end")
    t["file"] = t.file.astype(str).map(lambda s: Path(s).name)
    unknown = sorted(set(t.file) - set(durations.index))
    if unknown:
        raise ValueError(f"{path}: files not in the training audio folder: {unknown[:5]}")
    if not np.allclose(t.end - t.start, TILE_S):
        raise ValueError(f"{path}: every row must be {TILE_S:g} s long")
    inside = (t.start >= 0) & (t.end <= durations.reindex(t.file).to_numpy())
    if not inside.all():
        raise ValueError(f"{path}: {int((~inside).sum())} rows reach outside their recording")
    print(f"{path}: {len(t):,} {role} tiles at weight {weight}")
    return pd.DataFrame({"file": t.file, "start_s": t.start.astype(float), "role": role,
                         "weight": weight})[TILE_COLUMNS]


def hard_negative_tiles(path: Path, durations: pd.Series) -> pd.DataFrame:
    return read_tiles(path, durations, "hard_neg", 1)


def song_positive_tiles(path: Path, durations: pd.Series) -> pd.DataFrame:
    return read_tiles(path, durations, "song_pos", SONG_POSITIVE_WEIGHT)
