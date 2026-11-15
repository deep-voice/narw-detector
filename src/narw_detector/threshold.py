"""narw-threshold: choose a model's operating point tau on your own validation recordings.

    narw-threshold --model MODEL_DIR --audio VAL_DIR --annotations VAL_ANN \\
                   [--fp-per-hour 2] [--write]

Scans every recording under VAL_DIR end to end (they must be fully annotated: an unannotated
call counts as a false detection) and prints, at each rate, the lowest tau with at most that many
false detections per scanned hour, with the recall there (narw_detector.evaluate). A rate the
model stays below even with every event counted is reported as unreachable. --write stores the
tau of --fp-per-hour in MODEL_DIR/config.json, the operating point narw-detect uses.

Use recordings the model did not train on. The released model's tau was chosen on our own
validation set; this tool is for a model you trained, or to re-pick tau for your recordings.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from narw_detector.annotations import durations_s, list_recordings, read_annotations
from narw_detector.audio import read_2k
from narw_detector.evaluate import ScanScores, score_scan
from narw_detector.scan import scan_events
from narw_detector.weights import load_model, pick_device, write_tau

RATES = (1.0, 2.0, 5.0)


def describe(scores: ScanScores, rate: float) -> str:
    tau = scores.tau_at_rate(rate)
    if tau is None:
        ceiling = scores.at_tau(scores.floor)
        return (f"{rate:g} false detections/h: unreachable. Counting every event gives "
                f"{ceiling['fp_per_hour']:.2f}/h (recall {ceiling['recall_iou0.5']:.4f})")
    r = scores.at_tau(tau)
    return (f"{rate:g} false detections/h: tau {tau:.4f} -> recall {r['recall_iou0.5']:.4f} "
            f"(IoU 0.5), {r['recall_iou0.3']:.4f} (IoU 0.3); {r['false_detections']} false "
            f"({r['fp_per_hour']:.2f}/h); precision {r['precision']:.4f}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--audio", type=Path, required=True, help="validation recordings folder")
    p.add_argument("--annotations", type=Path, required=True,
                   help="their annotations: Raven selection table(s) or a CSV")
    p.add_argument("--fp-per-hour", type=float, default=2.0,
                   help="the rate --write stores tau for (default 2)")
    p.add_argument("--write", action="store_true", help="store that tau in config.json")
    p.add_argument("--device", default="auto",
                   help="cuda, cpu or mps; auto (the default) is cuda if available, else cpu")
    return p


def main(argv: list[str] | None = None) -> None:
    p = build_parser()
    a = p.parse_args(argv)
    device = pick_device(a.device)
    net, spec, tau_now = load_model(a.model, device)
    recordings = list_recordings(a.audio)
    events = read_annotations(a.annotations, recordings)
    scans = {name: scan_events(net, read_2k(p), spec, device)
             for name, p in tqdm(recordings.items(), unit="file", mininterval=30, ascii=True)}
    scores = score_scan(scans, events, durations_s(recordings))
    print(f"{scores.n_calls:,} calls in {scores.hours:.2f} h of {len(recordings)} recordings; "
          f"model tau now {tau_now:.4f}")
    for rate in sorted({*RATES, a.fp_per_hour}):
        print("  " + describe(scores, rate))
    if a.write:
        tau = scores.tau_at_rate(a.fp_per_hour)
        if tau is None:
            p.error(f"--fp-per-hour {a.fp_per_hour:g} is unreachable on these recordings: "
                    "nothing written")
        write_tau(a.model, tau, {"tau_source": f"narw-threshold, {a.fp_per_hour:g} false "
                                              f"detections per hour on {a.audio.name}"})
        print(f"wrote tau {tau:.4f} to {a.model / 'config.json'}")


if __name__ == "__main__":
    main()
