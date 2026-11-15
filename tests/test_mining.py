"""LFDCS output parsing and anchoring, and the mined-tile rules, on synthetic files."""
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd

from conftest import recording, write_wav
from narw_detector.mining.lfdcs import detection_stream, read_detections
from narw_detector.mining.sequences import Run
from narw_detector.mining.tiles import (
    cap_tiles,
    distance_to_runs,
    finish,
    mined_tiles,
    song_positive_tiles,
)

T0 = 4.8e8                                           # LFDCS's clock: seconds since 2006


def write_detection_file(path: Path, times: list[float], groups: list[int],
                         call_n: list[int], layout: list[int]) -> None:
    """A minimal LFDCS detection file; track segments laid down in `layout` order (LFDCS does
    not write them in detection order). Detection i's track is i repeated."""
    n = len(times)
    laid = np.asarray(call_n)[layout]
    call_start = np.empty(n, dtype=np.int64)
    call_start[layout] = np.concatenate([[0], np.cumsum(laid)[:-1]])
    ds = nc.Dataset(path, "w")
    ds.createDimension("n", n)
    ds.createDimension("ntrack", int(sum(call_n)))
    for name, dtype, values in [("detection_time", "f8", times), ("duration", "f4", [0.8] * n),
                                ("minfreq", "f4", [90.0] * n), ("maxfreq", "f4", [180.0] * n),
                                ("call_group_auto", "i4", groups)]:
        ds.createVariable(name, dtype, ("n",))[:] = values
    ds.createVariable("call_start", "i4", ("n",))[:] = call_start
    ds.createVariable("call_n", "i2", ("n",))[:] = call_n
    ds.createVariable("call_time", "f8", ("ntrack",))[:] = np.concatenate(
        [times[i] + 0.01 * np.arange(call_n[i]) for i in layout])
    ds.createVariable("call_freq", "f4", ("ntrack",))[:] = np.concatenate(
        [np.full(call_n[i], 100.0 + i) for i in layout])
    ds.close()


def test_tracks_are_sliced_per_detection_and_times_kept_in_float64(tmp_path):
    f = tmp_path / "lfdcs_test.nc"
    write_detection_file(f, [T0 + 1, T0 + 5, T0 + 9], [5, 18, -32767], [3, 5, 2], [2, 0, 1])
    d = read_detections(f)
    assert [len(t) for t in d.track_freq] == [3, 5, 2]
    assert [float(t[0]) for t in d.track_freq] == [100.0, 101.0, 102.0]
    np.testing.assert_allclose(d.track_dt[1], 0.01 * np.arange(5), atol=1e-6)
    assert d.call_group.tolist() == [5, 18, -32767]


def test_detections_are_anchored_by_true_file_durations(tmp_path):
    audio = tmp_path / "audio"
    a = write_wav(audio / "s.210101000000.wav", recording(100.5, []))
    b = write_wav(audio / "s.210101000140.wav", recording(60, []))
    out = tmp_path / "processed"
    (out / "lfdcs").mkdir(parents=True)
    (out / "specaudio").mkdir()
    idx = nc.Dataset(out / "specaudio" / "recorder_index.nc", "w")
    idx.createDimension("f", 2)
    idx.createVariable("stime", "f8", ("f",))[:] = [T0, T0 + 100.0]       # a nominal grid
    idx.close()
    write_detection_file(out / "lfdcs" / "lfdcs_x.nc", [T0 + 20.0, T0 + 110.0], [6, 15],
                         [2, 2], [0, 1])
    d = detection_stream(out, {a.name: a, b.name: b})
    assert d.file.tolist() == [a.name, b.name]
    np.testing.assert_allclose(d.offset_s, [20.0, 9.5])        # 110 - 100.5, not 110 - 100
    np.testing.assert_allclose(d.stream_s, [20.0, 110.0])


def _stream():
    """Units of a song run 100-160 s in file f.wav, plus one unit outside any run."""
    t = np.arange(100.0, 161.0, 4.0)
    return pd.DataFrame({"file": "f.wav", "offset_s": np.r_[t, 400.0], "stream_s": np.r_[t, 400.0],
                         "duration": 1.0, "call_group": np.r_[[16] * len(t), 6]})


def test_hard_negative_tiles_snap_to_the_grid_and_avoid_possibles():
    s = _stream()
    run_id = np.r_[np.zeros(len(s) - 1, int), -1]
    stereo = np.linspace(0.5, 0.9, len(s))
    ann = pd.DataFrame({"file": "f.wav", "start_s": [150.0], "end_s": [151.0],
                        "label": ["possible"]})
    tiles = mined_tiles(s, run_id, stereo, ann, pd.Series({"f.wav": 0.0}))
    centres = tiles.start_s + 5
    assert set(np.round(centres % 5, 6)) == {0.0}                       # on the 5 s grid
    assert not (np.abs(centres - 150.5) <= 15.5).any()                 # none near the possible
    assert (tiles.pool == "hw").all() and tiles[["file", "start_s"]].duplicated().sum() == 0
    assert 400.0 not in tiles.start_s.tolist()                          # not in a run: no tile


def test_cap_keeps_at_most_n_per_pool_and_stereotypy_group():
    tiles = pd.DataFrame({"file": "f.wav", "start_s": np.arange(90) * 5.0,
                          "pool": ["rw", "hw", "flagged"] * 30, "stereotypy": np.arange(90.0)})
    kept = cap_tiles(tiles, max_n=18)
    assert len(kept) == 18 and kept.groupby("pool").size().tolist() == [6, 6, 6]


def test_song_positives_are_upcalls_within_30_s_of_a_run():
    runs = [Run(np.arange(3), 100.0, 160.0, 4.0)]
    assert distance_to_runs(np.array([120.0, 50.0, 190.0]), runs).tolist() == [0.0, 50.0, 30.0]
    ann = pd.DataFrame({"file": "f.wav", "start_s": [129.0, 189.0, 300.0],
                        "end_s": [130.0, 190.0, 301.0], "label": "upcall"})
    tiles = song_positive_tiles(ann, runs, pd.Series({"f.wav": 0.0}))
    assert tiles.start_s.tolist() == [125.0, 185.0]           # 189.5 is 29.5 s past the run


def test_finish_drops_tiles_past_the_end_and_clamps_negative_starts():
    tiles = pd.DataFrame({"file": "f.wav", "start_s": [-5.0, 0.0, 50.0, 95.0]})
    out = finish(tiles, pd.Series({"f.wav": 100.0}))
    assert out.start_s.tolist() == [0.0, 50.0] and out.end_s.tolist() == [10.0, 60.0]
