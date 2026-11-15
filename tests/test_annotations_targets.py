"""From an analyst's Raven selection table to the frame targets the model trains on."""
import numpy as np
import pandas as pd

from conftest import recording, write_wav
from narw_detector.annotations import durations_s, list_recordings, read_annotations
from narw_detector.audio import AudioStore
from narw_detector.dataset import WindowDataset, frame_targets
from narw_detector.spectrogram import RenderSpec
from narw_detector.windows import lattice_tiles, place_windows

RAVEN_HEADER = ("Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tLow Freq (Hz)\t"
                "High Freq (Hz)\n")


def _raven(path, rows):
    lines = [RAVEN_HEADER] + [f"{i + 1}\t{view}\t1\t{a}\t{b}\t100\t200\n"
                              for i, (view, a, b) in enumerate(rows)]
    path.write_text("".join(lines))
    return path


def test_raven_table_becomes_the_right_frame_targets(tmp_path):
    audio = tmp_path / "audio"
    write_wav(audio / "site1.240101120000.wav", recording(90, [23.0, 51.3]))
    # Raven writes a Spectrogram and a Waveform row per selection when both views are open
    _raven(tmp_path / "site1.240101120000.Table.1.selections.txt",
           [("Spectrogram 1", 23.0, 24.5), ("Waveform 1", 23.0, 24.5),
            ("Spectrogram 1", 51.3, 52.1)])
    recordings = list_recordings(audio)
    ev = read_annotations(tmp_path / "site1.240101120000.Table.1.selections.txt", recordings)
    assert ev[["start_s", "end_s"]].values.tolist() == [[23.0, 24.5], [51.3, 52.1]]
    assert (ev.label == "upcall").all() and (ev.file == "site1.240101120000.wav").all()

    spec = RenderSpec()
    windows = pd.DataFrame({"file": ["site1.240101120000.wav"], "start_s": [20.0],
                            "slot_start_s": [0.0], "role": ["pos"], "weight": [1]})
    ds = WindowDataset(windows, ev, AudioStore(recordings, tmp_path / "cache"), spec)
    item = ds[0]
    assert item["audio"].shape == (120_000,) and item["slot"] == 0
    y = item["target"].numpy()
    assert y.shape == (1854,)
    # call 1 at 3.0-4.5 s into slot 0: frames round(92.7)=93 .. round(139.05)=139
    expect = np.zeros(1854, dtype=np.float32)
    expect[93:139] = 1
    # call 2 at 31.3-32.1 s: slot 3, 1.3-2.1 s in: frames 927 + round(40.17)=40 .. round(64.89)=65
    expect[927 + 40:927 + 65] = 1
    np.testing.assert_array_equal(y, expect)


def test_multi_file_tables_place_calls_by_begin_file(tmp_path):
    audio = tmp_path / "audio"
    for name in ("a.wav", "b.wav"):
        write_wav(audio / name, recording(70, []))
    table = tmp_path / "sequence.selections.txt"
    table.write_text("Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tBegin File\t"
                     "File Offset (s)\tlabel\n"
                     "1\tSpectrogram 1\t1\t10.0\t11.0\ta.wav\t10.0\tupcall\n"
                     "2\tSpectrogram 1\t1\t75.0\t76.5\tb.wav\t5.0\tPossible\n")
    ev = read_annotations(table, list_recordings(audio))
    assert ev.values.tolist() == [["a.wav", 10.0, 11.0, "upcall"], ["b.wav", 5.0, 6.5, "possible"]]


def test_csv_annotations_and_calls_past_the_file_end(tmp_path):
    audio = tmp_path / "audio"
    write_wav(audio / "x.wav", recording(30, []))
    csv = tmp_path / "ann.csv"
    pd.DataFrame({"file": ["x.wav", "x.wav"], "start": [3.0, 29.5],
                  "end": [4.0, 30.7]}).to_csv(csv, index=False)
    ev = read_annotations(csv, list_recordings(audio))
    assert ev.values.tolist() == [["x.wav", 3.0, 4.0, "upcall"]]       # the second runs past 30 s


def test_targets_clip_calls_the_window_cuts():
    y = frame_targets(np.array([[-0.5, 0.4], [9.8, 10.0]]), 10.0, 309)
    assert y[:round(0.4 / 10 * 309)].all() and y[-6:].all() and y.sum() == 12 + 6


def test_training_windows_hold_their_tile_in_one_slot(tmp_path):
    audio = tmp_path / "audio"
    write_wav(audio / "r.wav", recording(120, [20.0, 47.0, 90.0]))
    recordings = list_recordings(audio)
    ev = pd.DataFrame({"file": "r.wav", "start_s": [20.0, 47.0, 90.0, 70.0],
                       "end_s": [21.0, 48.0, 91.0, 71.0],
                       "label": ["upcall", "upcall", "upcall", "possible"]})
    dur = durations_s(recordings)
    tiles = lattice_tiles(ev, dur)
    pos = tiles[tiles.role == "pos"]
    assert len(tiles[tiles.role == "bg"]) == len(pos) == 6        # each call is whole in 2 tiles
    w = place_windows(tiles, ev, dur)
    assert ((w.start_s >= 0) & (w.start_s + 60 <= dur["r.wav"])).all()
    assert ((w.slot_start_s % 10 == 0) & (w.slot_start_s <= 50)).all()
    # no window holds the possible
    assert not ((w.start_s < 71.0) & (w.start_s + 60 > 70.0)).any()
    again = place_windows(tiles, ev, dur)
    pd.testing.assert_frame_equal(w, again)                             # a fixed placement
