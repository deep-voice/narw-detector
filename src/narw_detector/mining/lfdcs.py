"""LFDCS output -> a detection stream: one row per LFDCS detection with its pitch track.

LFDCS writes one netCDF file per detection chunk under `<Outdir>/lfdcs/` (lfdcs_*.nc), plus
`<Outdir>/specaudio/recorder_index.nc`. Per detection: `detection_time`, `duration`, `minfreq`,
`maxfreq`, `call_group_auto` (the GOM9 call group; -32767 when unclassified) and a pitch track
(`call_time`, `call_freq`, sliced per detection by its own 0-based `call_start` / `call_n`; the
segments are not stored in detection order).

Time: `detection_time` is LFDCS's own clock (seconds since its StartDate). LFDCS starts that
clock at the first file it processes and advances it by each file's true duration. So with the
recordings in processing order (by name; the names must sort in time), file i starts at
min(recorder_index stime) + the summed durations of the files before it. The per-file stime
values are not used: they are a nominal grid, which drifted by about 0.8 s per file in our
recordings.

Needs the netCDF4 package (pip install "narw-detector[mine]").
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from narw_detector.audio import info

STREAM_COLUMNS = ["file", "offset_s", "stream_s", "duration", "minfreq", "maxfreq", "call_group",
                  "track_dt", "track_freq"]


def _dataset(path: Path):
    import netCDF4
    ds = netCDF4.Dataset(path)
    ds.set_auto_mask(False)
    return ds


def read_detections(path: Path) -> pd.DataFrame:
    """One LFDCS detection file -> rows time, duration, minfreq, maxfreq, call_group,
    track_dt (seconds from the detection's start), track_freq."""
    ds = _dataset(path)
    try:
        v = ds.variables
        t = v["detection_time"][:].astype(np.float64)
        start, n = v["call_start"][:].astype(np.int64), v["call_n"][:].astype(np.int64)
        if n.sum() != ds.dimensions["ntrack"].size:
            raise ValueError(f"{path.name}: the track lengths do not sum to ntrack")
        # times run to ~5e8 s: subtract in float64, or float32 flattens each track to an instant
        call_time = v["call_time"][:].astype(np.float64)
        call_freq = v["call_freq"][:].astype(np.float32)
        df = pd.DataFrame({
            "time": t, "duration": v["duration"][:].astype(float),
            "minfreq": v["minfreq"][:].astype(float), "maxfreq": v["maxfreq"][:].astype(float),
            "call_group": v["call_group_auto"][:].astype(np.int64),
            "track_dt": [(call_time[s:s + k] - t0).astype(np.float32)
                         for s, k, t0 in zip(start, n, t, strict=True)],
            "track_freq": [call_freq[s:s + k] for s, k in zip(start, n, strict=True)]})
    finally:
        ds.close()
    return df


def read_run(out_dir: Path) -> pd.DataFrame:
    """Every detection file under <Outdir>/lfdcs/ (not the call-library copies or the file
    index), in time order."""
    files = [f for f in sorted((Path(out_dir) / "lfdcs").rglob("lfdcs_*.nc"))
             if "call_library" not in str(f) and "file_index" not in f.name]
    if not files:
        raise ValueError(f"no LFDCS detection files (lfdcs_*.nc) under {Path(out_dir) / 'lfdcs'}")
    return (pd.concat([read_detections(f) for f in files], ignore_index=True)
            .sort_values("time").reset_index(drop=True))


def first_stime(out_dir: Path) -> float:
    ds = _dataset(Path(out_dir) / "specaudio" / "recorder_index.nc")
    try:
        return float(np.min(ds.variables["stime"][:]))
    finally:
        ds.close()


def file_offsets(recordings: dict[str, Path]) -> tuple[pd.Series, np.ndarray]:
    """(file name -> its start on the run's continuous clock, i.e. the summed durations of the
    files before it in name order; the durations in that order)."""
    names = sorted(recordings)
    dur = np.array([frames / sr for sr, frames in (info(recordings[n]) for n in names)])
    return pd.Series(np.concatenate([[0.0], np.cumsum(dur)[:-1]]), index=names), dur


def detection_stream(out_dir: Path, recordings: dict[str, Path]) -> pd.DataFrame:
    """The run's detections anchored to (file, offset_s), with stream_s on the continuous clock;
    one row per (file, offset_s) (LFDCS can start several tracks in one time slice), sorted by
    stream_s. recordings: the files LFDCS processed."""
    offsets, dur = file_offsets(recordings)
    d = read_run(out_dir)
    stream_s = d.time.to_numpy() - first_stime(out_dir)
    i = np.clip(np.searchsorted(offsets.to_numpy(), stream_s, "right") - 1, 0, len(offsets) - 1)
    d = d.assign(file=offsets.index.to_numpy()[i], offset_s=stream_s - offsets.to_numpy()[i],
                 stream_s=stream_s)
    if (d.offset_s < -1).any():
        raise ValueError("a detection precedes its file: are these the recordings this LFDCS "
                         "run processed?")
    past = d.offset_s.to_numpy() > dur[i]
    if past.any():
        print(f"WARNING: dropped {int(past.sum())} detections past their file's end - check "
              "that the recordings are the ones LFDCS processed, in the same order")
    d = d[~past].drop_duplicates(["file", "offset_s"]).sort_values("stream_s", kind="stable")
    print(f"LFDCS stream: {len(d):,} detections over {d.file.nunique():,} files")
    return d[STREAM_COLUMNS].reset_index(drop=True)
