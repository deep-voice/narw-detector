"""Raven Pro selection tables (tab-separated text), written and read.

Written: one table per recording, `<recording stem>.selections.txt`, one row per detection. The
detector localises calls in time only, so Low/High Freq is a fixed 50-250 Hz band around the
upcall; Confidence is the detection's peak probability.

Read (annotations for training): the analyst's own tables. Begin/End Time (s) give the call.
A table made over a single recording names it by its file name (`<recording stem>.*.txt`, Raven's
default `<stem>.Table.1.selections.txt`); a table made over a file sequence carries "Begin File"
and "File Offset (s)" columns, and those place each call in its file.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

LOW_HZ, HIGH_HZ = 50.0, 250.0
BEGIN, END = "Begin Time (s)", "End Time (s)"


def write_selection_table(path: Path, events: list[tuple[float, float, float]]) -> None:
    """events (t0_s, t1_s, confidence) in file seconds -> a Raven selection table at path."""
    table = pd.DataFrame({
        "Selection": range(1, len(events) + 1),
        "View": "Spectrogram 1",
        "Channel": 1,
        BEGIN: [round(e[0], 4) for e in events],
        END: [round(e[1], 4) for e in events],
        "Low Freq (Hz)": LOW_HZ,
        "High Freq (Hz)": HIGH_HZ,
        "Confidence": [round(e[2], 4) for e in events],
    })
    table.to_csv(path, sep="\t", index=False)


def read_selection_table(path: Path, recordings: set[str]) -> pd.DataFrame:
    """One analyst table -> rows (file, start_s, end_s[, label]) in file seconds.

    recordings: the file names of the audio, so a single-recording table can be matched to its
    recording by name. A "label" column (any case) is kept when present."""
    t = pd.read_csv(path, sep="\t")
    if not {BEGIN, END} <= set(t.columns):
        raise ValueError(f"{path}: no '{BEGIN}' / '{END}' columns")
    if "View" in t.columns:
        t = t[t["View"].str.startswith("Spectrogram")]
    label = next((c for c in t.columns if c.lower() == "label"), None)
    if "Begin File" in t.columns:
        if "File Offset (s)" not in t.columns:
            raise ValueError(f"{path}: 'Begin File' without 'File Offset (s)'")
        out = pd.DataFrame({"file": t["Begin File"].astype(str).map(lambda s: Path(s).name),
                            "start_s": t["File Offset (s)"].astype(float)})
        out["end_s"] = out.start_s + (t[END] - t[BEGIN]).astype(float).to_numpy()
    else:
        out = pd.DataFrame({"file": _recording_of(path, recordings),
                            "start_s": t[BEGIN].astype(float), "end_s": t[END].astype(float)})
    if label is not None:
        out["label"] = t[label].astype(str).str.strip().str.lower().to_numpy()
    return out.reset_index(drop=True)


def _recording_of(table: Path, recordings: set[str]) -> str:
    """The recording a single-recording table belongs to: the one whose stem starts the table's
    file name (the longest such stem, so 'a1.wav' wins over 'a.wav' for 'a1.Table.1...')."""
    hits = [r for r in recordings if table.name.startswith(Path(r).stem + ".")]
    if not hits:
        raise ValueError(f"{table.name}: names no recording (expected "
                         "'<recording stem>.<...>.txt') and has no 'Begin File' column")
    return max(hits, key=lambda r: len(Path(r).stem))
