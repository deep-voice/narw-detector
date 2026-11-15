"""Scoring a scan of fully annotated recordings: recall at a false-detection rate.

Each recording is scanned (narw_detector.scan), and its merged events are matched to the
annotated calls by temporal IoU (intersection over union):
  * a call is found at threshold tau when an event with confidence >= tau overlaps it at
    IoU >= 0.5 (recall at IoU >= 0.3 is reported too);
  * an event is a false detection when it overlaps no call at IoU >= 0.3 and no possible. An
    event on a possible counts neither way.

One pass gives every threshold. For a rate R (false detections per hour), tau is the lowest
event confidence that keeps the false detections at or below floor(R * hours). If the model stays
below R even with every event counted, the rate is unreachable: narw-threshold reports that, and
training selection scores the epoch with every event counted.

A real call that the analysts missed counts as a false detection, so on such recordings the
false-detection rate is overstated.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from narw_detector.events import EVENT_FLOOR, Event

HIT_IOU, LOOSE_IOU, FALSE_IOU = 0.5, 0.3, 0.3


def matches(a: np.ndarray, b: np.ndarray, iou: float) -> np.ndarray:
    """(len a, len b): temporal IoU of spans a[i], b[j] >= iou, as intersection >= iou * union."""
    inter = np.clip(np.minimum(a[:, None, 1], b[None, :, 1])
                    - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    union = (a[:, 1] - a[:, 0])[:, None] + (b[:, 1] - b[:, 0])[None, :] - inter
    return inter >= iou * union


@dataclass
class ScanScores:
    """Everything needed to score any threshold: the hours scanned, each call's best matching
    confidence at IoU 0.5 and 0.3 (-1 when nothing matches), and every merged event's confidence
    and class ("real" on a call, "poss" on a possible only, "fp" otherwise)."""
    hours: float = 0.0
    floor: float = EVENT_FLOOR
    call_best: dict = field(default_factory=lambda: {HIT_IOU: [], LOOSE_IOU: []})
    det_conf: list = field(default_factory=list)
    det_class: list = field(default_factory=list)

    def add_file(self, events: list[Event], calls: np.ndarray, possibles: np.ndarray,
                 seconds: float) -> None:
        """One recording: its merged events, its upcalls and possibles (n, 2) in seconds."""
        merged = np.array(events, dtype=float).reshape(-1, 3)
        spans, conf = merged[:, :2], merged[:, 2]
        self.hours += seconds / 3600
        for iou in (HIT_IOU, LOOSE_IOU):
            hit = matches(calls, spans, iou)
            best = np.where(hit, conf[None, :], -1.0).max(1) if len(spans) else -np.ones(len(calls))
            self.call_best[iou] += best.tolist()
        real = matches(spans, calls, FALSE_IOU).any(1)
        on_poss = matches(spans, possibles, FALSE_IOU).any(1)
        self.det_conf += conf.tolist()
        self.det_class += np.where(real, "real", np.where(on_poss, "poss", "fp")).tolist()

    @property
    def n_calls(self) -> int:
        return len(self.call_best[HIT_IOU])

    def at_tau(self, tau: float) -> dict:
        conf, cls = np.asarray(self.det_conf), np.asarray(self.det_class, dtype=object)
        firing = conf >= tau
        n_fp, n_real = int((firing & (cls == "fp")).sum()), int((firing & (cls == "real")).sum())
        judged = n_fp + n_real
        out = {"tau": float(tau), "false_detections": n_fp,
               "fp_per_hour": n_fp / self.hours if self.hours else float("nan"),
               "precision": n_real / judged if judged else float("nan"),
               "n_calls": self.n_calls, "hours": self.hours}
        for iou in (HIT_IOU, LOOSE_IOU):
            best = np.asarray(self.call_best[iou])
            out[f"recall_iou{iou:g}"] = float((best >= tau).mean()) if len(best) else float("nan")
        return out

    def _lowest_tau_within(self, rate: float) -> float:
        """The lowest merged confidence (or 1.0) whose false detections <= floor(rate * hours)."""
        conf, cls = np.asarray(self.det_conf), np.asarray(self.det_class, dtype=object)
        fp = np.sort(conf[cls == "fp"])
        budget = int(np.floor(rate * self.hours))
        candidates = np.unique(np.concatenate([conf, [1.0]]))
        n_fp = len(fp) - np.searchsorted(fp, candidates, side="left")        # fp >= candidate
        return float(candidates[n_fp <= budget].min())

    def reachable(self, rate: float) -> bool:
        """True when counting every event gives at least `rate` false detections per hour."""
        n_fp = sum(c == "fp" for c in self.det_class)
        return self.hours > 0 and n_fp / self.hours >= rate

    def tau_at_rate(self, rate: float) -> float | None:
        """tau at a false-detection rate, or None when the rate is unreachable."""
        return self._lowest_tau_within(rate) if self.reachable(rate) else None

    def selection_scores(self, rate: float) -> dict:
        """The scores a training epoch is selected on: at tau_at_rate(rate), or with every event
        counted when the rate is unreachable."""
        return self.at_tau(self._lowest_tau_within(rate)) | {"reachable": self.reachable(rate)}


def score_scan(events: dict[str, list[Event]], annotations: pd.DataFrame,
               durations: pd.Series) -> ScanScores:
    """Score a scan: events {file name: merged events}, the annotations of those recordings,
    their durations in seconds."""
    scores = ScanScores()
    for f, ev in events.items():
        a = annotations[annotations.file == f]
        up = a[(a.label == "upcall") & (a.end_s > a.start_s)]
        poss = a[a.label == "possible"]
        scores.add_file(ev, up[["start_s", "end_s"]].to_numpy(float).reshape(-1, 2),
                        poss[["start_s", "end_s"]].to_numpy(float).reshape(-1, 2),
                        float(durations[f]))
    return scores
