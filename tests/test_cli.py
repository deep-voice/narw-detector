"""The command-line tools: each runs as a module, and end to end on tiny synthetic data."""
import subprocess
import sys

import pandas as pd
import pytest
import torch

from conftest import recording, write_wav
from narw_detector import detect, threshold, train
from narw_detector.model import DualStreamSed
from narw_detector.spectrogram import RenderSpec
from narw_detector.weights import load_model, read_config, save_model


@pytest.mark.parametrize("module", ["detect", "train", "threshold", "mining.cli"])
def test_each_cli_runs_as_a_module(module):
    r = subprocess.run([sys.executable, "-m", f"narw_detector.{module}", "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "usage" in r.stdout, r.stderr


@pytest.fixture
def model_dir(tmp_path):
    torch.manual_seed(0)
    return save_model(tmp_path / "model", DualStreamSed(width=32), 32, RenderSpec(), tau=0.5)


def test_model_folder_round_trips_and_checks_its_checksum(model_dir):
    net, spec, tau = load_model(model_dir)
    assert tau == 0.5 and spec == RenderSpec() and not net.training
    (model_dir / "weights.pt").write_bytes((model_dir / "weights.pt").read_bytes()[:-10])
    with pytest.raises(ValueError, match="SHA-256"):
        load_model(model_dir)


def test_detect_writes_csv_and_one_raven_table_per_recording(tmp_path, model_dir):
    write_wav(tmp_path / "in" / "a.wav", recording(75, [20.0]))
    write_wav(tmp_path / "in" / "sub" / "b.wav", recording(65, [5.0], sr=48_000), 48_000)
    # main() returns None: a console script exits with main()'s return value as its status
    assert detect.main([str(tmp_path / "in"), "--model", str(model_dir), "--out",
                        str(tmp_path / "out"), "--device", "cpu"]) is None
    out = pd.read_csv(tmp_path / "out" / "detections.csv")
    assert list(out.columns) == ["file", "start_s", "end_s", "confidence"]
    assert (out.confidence >= 0.5).all()
    for stem in ("a", "b"):
        t = pd.read_csv(tmp_path / "out" / f"{stem}.selections.txt", sep="\t")
        assert list(t.columns) == ["Selection", "View", "Channel", "Begin Time (s)",
                                   "End Time (s)", "Low Freq (Hz)", "High Freq (Hz)",
                                   "Confidence"]


def _dataset(root, calls_by_file):
    rows = []
    for name, calls in calls_by_file.items():
        write_wav(root / "audio" / name, recording(70, calls, seed=len(rows)))
        rows += [(name, c, c + 1.0) for c in calls]
    pd.DataFrame(rows, columns=["file", "start", "end"]).to_csv(root / "ann.csv", index=False)
    return root / "audio", root / "ann.csv"


def test_train_one_epoch_then_pick_a_threshold(tmp_path, capsys):
    audio, ann = _dataset(tmp_path / "train", {"t1.wav": [12.0, 31.0, 55.0],
                                               "t2.wav": [8.0, 40.0]})
    val_audio, val_ann = _dataset(tmp_path / "val", {"v1.wav": [20.0, 45.0]})
    hard = tmp_path / "hard.csv"
    pd.DataFrame({"file": ["t1.wav"], "start": [60.0], "end": [70.0]}).to_csv(hard, index=False)
    assert train.main(["--audio", str(audio), "--annotations", str(ann), "--val-audio",
                       str(val_audio), "--val-annotations", str(val_ann), "--hard-negatives",
                       str(hard), "--out", str(tmp_path / "run"), "--epochs", "1",
                       "--accumulate", "32", "--num-workers", "0", "--device", "cpu"]) is None
    config = read_config(tmp_path / "run" / "model")
    assert config["best_epoch"] == 0 and 0.4 <= config["tau"] <= 1.0
    assert len(pd.read_csv(tmp_path / "run" / "history.csv")) == 1
    capsys.readouterr()
    assert threshold.main(["--model", str(tmp_path / "run" / "model"), "--audio",
                           str(val_audio), "--annotations", str(val_ann), "--device",
                           "cpu"]) is None
    assert "2 calls in" in capsys.readouterr().out
