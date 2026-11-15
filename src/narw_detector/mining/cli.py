"""narw-mine: hard negatives and song positives from an LFDCS run over your training recordings.

    narw-mine --lfdcs-out LFDCS_OUTDIR --audio TRAIN_DIR --annotations TRAIN_ANN --out OUT_DIR

LFDCS_OUTDIR is the Outdir of an LFDCS run over exactly the recordings in TRAIN_DIR (see
docs/lfdcs.md); the recording names must sort in time. Writes
OUT_DIR/hard_negatives.csv and OUT_DIR/song_positives.csv (file, start, end: 10 s tiles), for
narw-train --hard-negatives / --song-positives. The rules are in narw_detector.mining.tiles.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from narw_detector.annotations import durations_s, list_recordings, read_annotations
from narw_detector.mining.lfdcs import detection_stream, file_offsets
from narw_detector.mining.tiles import (
    cap_tiles,
    finish,
    mine_runs,
    mined_tiles,
    song_positive_tiles,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lfdcs-out", type=Path, required=True,
                   help="the LFDCS run's Outdir (holding lfdcs/ and specaudio/)")
    p.add_argument("--audio", type=Path, required=True,
                   help="the recordings that LFDCS run processed")
    p.add_argument("--annotations", type=Path, required=True,
                   help="their upcall annotations: Raven selection table(s) or a CSV")
    p.add_argument("--out", type=Path, required=True)
    return p


def main(argv: list[str] | None = None) -> None:
    a = build_parser().parse_args(argv)
    recordings = list_recordings(a.audio)
    annotations = read_annotations(a.annotations, recordings)
    offsets, _ = file_offsets(recordings)
    durations = durations_s(recordings)
    stream = detection_stream(a.lfdcs_out, recordings)
    runs, run_id, stereo = mine_runs(stream)
    song = finish(song_positive_tiles(annotations, runs, offsets), durations)
    hard = finish(cap_tiles(mined_tiles(stream, run_id, stereo, annotations, offsets)), durations)
    hard = hard.merge(song[["file", "start_s"]], how="left", indicator=True)
    hard = hard[hard._merge == "left_only"].drop(columns="_merge")
    a.out.mkdir(parents=True, exist_ok=True)
    cols = {"start_s": "start", "end_s": "end"}
    hard.rename(columns=cols)[["file", "start", "end", "pool", "stereotypy"]].to_csv(
        a.out / "hard_negatives.csv", index=False)
    song.rename(columns=cols)[["file", "start", "end"]].to_csv(
        a.out / "song_positives.csv", index=False)
    print(f"{len(hard):,} hard negatives, {len(song):,} song positives -> {a.out}")


if __name__ == "__main__":
    main()
