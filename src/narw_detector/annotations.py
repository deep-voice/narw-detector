"""Recordings and their upcall annotations, as the training and threshold tools read them.

Recordings: every audio file under a folder (searched recursively), keyed by file name, so file
names must be unique within the folder.

Annotations: Raven Pro selection tables (a .txt file, or a folder of them; see narw_detector.raven)
or one CSV with columns file, start, end (seconds from the start of the file) - each row one call.
An optional `label` column marks each row "upcall" (the default) or "possible": a possible is a
call the analyst was unsure of. Possibles are never taught as calls nor as background - no training
window holds one - and a detection on a possible is not counted as false.

A call that runs past the end of its file is dropped (its end lies in the next file), as the
original training data dropped calls split across two files.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from narw_detector.audio import SAMPLE_RATE, n_samples_2k
from narw_detector.raven import read_selection_table

AUDIO_SUFFIXES = {".wav", ".flac", ".aif", ".aiff"}
LABELS = ("upcall", "possible")
COLUMNS = ["file", "start_s", "end_s", "label"]


def list_recordings(folder: Path) -> dict[str, Path]:
    """{file name: path} of every audio file under folder."""
    paths = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES)
    if not paths:
        raise ValueError(f"no audio files ({', '.join(sorted(AUDIO_SUFFIXES))}) under {folder}")
    names = pd.Series([p.name for p in paths])
    repeated = sorted(set(names[names.duplicated()]))
    if repeated:
        raise ValueError(f"file names must be unique under {folder}; repeated: {repeated[:5]}")
    return {p.name: p for p in paths}


def durations_s(recordings: dict[str, Path]) -> pd.Series:
    """File name -> duration in seconds at 2 kHz."""
    return pd.Series({name: n_samples_2k(p) / SAMPLE_RATE for name, p in recordings.items()})


def _read_csv(source: Path) -> pd.DataFrame:
    raw = pd.read_csv(source).rename(columns={"start": "start_s", "end": "end_s"})
    if not {"file", "start_s", "end_s"} <= set(raw.columns):
        raise ValueError(f"{source}: a CSV needs columns file, start, end")
    raw["file"] = raw.file.astype(str).map(lambda s: Path(s).name)
    return raw


def _read_raven_tables(source: Path, recordings: dict[str, Path]) -> pd.DataFrame:
    tables = sorted(source.rglob("*.txt")) if source.is_dir() else [source]
    if not tables:
        raise ValueError(f"no Raven selection tables (*.txt) under {source}")
    return pd.concat([read_selection_table(t, set(recordings)) for t in tables],
                     ignore_index=True)


def read_annotations(source: Path, recordings: dict[str, Path]) -> pd.DataFrame:
    """A CSV or Raven selection tables -> rows file, start_s, end_s, label, checked against the
    recordings. Calls running past their file's end are dropped, and the count is printed."""
    source = Path(source)
    if source.suffix.lower() == ".csv":
        raw = _read_csv(source)
    else:
        raw = _read_raven_tables(source, recordings)
    ev = raw.assign(label=raw.get("label", "upcall")).reindex(columns=COLUMNS)
    ev["label"] = ev.label.fillna("upcall").astype(str).str.lower()
    bad = sorted(set(ev.label) - set(LABELS))
    if bad:
        raise ValueError(f"{source}: labels must be {LABELS}, got {bad}")
    unknown = sorted(set(ev.file) - set(recordings))
    if unknown:
        raise ValueError(f"{source}: annotations name files not in the audio folder: "
                         f"{unknown[:5]}")
    if not ((ev.end_s >= ev.start_s).all() and (ev.start_s >= 0).all()):
        raise ValueError(f"{source}: every annotation needs 0 <= start <= end")
    dur = durations_s({f: recordings[f] for f in ev.file.unique()})
    past_end = ev.end_s.to_numpy() > dur.reindex(ev.file).to_numpy()
    if past_end.any():
        print(f"{source}: dropped {int(past_end.sum())} annotations running past their file's end")
    ev = ev[~past_end].sort_values(["file", "start_s"]).reset_index(drop=True)
    print(f"{source}: {(ev.label == 'upcall').sum():,} upcalls, "
          f"{(ev.label == 'possible').sum():,} possibles in {ev.file.nunique():,} files")
    return ev
