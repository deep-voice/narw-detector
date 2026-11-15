"""narw-detect: find right whale upcalls in recordings.

    narw-detect RECORDINGS... --model MODEL_DIR --out OUT_DIR

RECORDINGS are audio files or folders (searched recursively for .wav / .flac / .aif / .aiff), at
any sample rate; each is resampled to 2 kHz. Writes OUT_DIR/detections.csv (file, start_s, end_s,
confidence: one row per detected call) and one Raven Pro selection table per recording,
OUT_DIR/<recording stem>.selections.txt. A detection is a merged event whose confidence (its peak
frame probability) is at least the model's operating point tau, from MODEL_DIR/config.json.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from narw_detector.annotations import AUDIO_SUFFIXES
from narw_detector.audio import read_2k
from narw_detector.raven import write_selection_table
from narw_detector.scan import detections, scan_events
from narw_detector.weights import load_model, pick_device


def find_recordings(inputs: list[Path]) -> list[Path]:
    """The audio files given, plus every audio file under the folders given."""
    files = []
    for p in map(Path, inputs):
        if p.is_file():
            files.append(p)
        else:
            files += sorted(q for q in p.rglob("*") if q.suffix.lower() in AUDIO_SUFFIXES)
    if not files:
        raise ValueError(f"no audio files in {[str(p) for p in inputs]}")
    stems = pd.Series([f.stem for f in files])
    repeated = sorted(set(stems[stems.duplicated()]))
    if repeated:
        raise ValueError(f"recording names must be unique (each gets its own Raven table); "
                         f"repeated: {repeated[:5]}")
    return files


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recordings", nargs="+", type=Path, help="audio files or folders")
    p.add_argument("--model", type=Path, required=True,
                   help="model folder holding weights.pt and config.json")
    p.add_argument("--out", type=Path, required=True, help="output folder")
    p.add_argument("--device", default="auto",
                   help="cuda, cpu or mps; auto (the default) is cuda if available, else cpu")
    return p


def main(argv: list[str] | None = None) -> None:
    a = build_parser().parse_args(argv)
    device = pick_device(a.device)
    net, spec, tau = load_model(a.model, device)
    files = find_recordings(a.recordings)
    a.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(files)} recordings, model {a.model} (tau {tau:.4f}), device {device}")
    rows = []
    for f in tqdm(files, unit="file", mininterval=30, ascii=True):
        dets = detections(scan_events(net, read_2k(f), spec, device), tau)
        write_selection_table(a.out / f"{f.stem}.selections.txt", dets)
        rows += [(str(f), t0, t1, c) for t0, t1, c in dets]
    out = pd.DataFrame(rows, columns=["file", "start_s", "end_s", "confidence"]).round(4)
    out.to_csv(a.out / "detections.csv", index=False)
    print(f"{len(out):,} detections -> {a.out / 'detections.csv'}")


if __name__ == "__main__":
    main()
