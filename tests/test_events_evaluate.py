"""Decoding frame probabilities to events, merging, and the scan scores."""
import numpy as np
import pandas as pd
import pytest

from narw_detector.evaluate import ScanScores, score_scan
from narw_detector.events import frame_times, merge, probs_to_events, window_events
from narw_detector.scan import window_starts


def test_runs_above_the_floor_of_two_or_more_frames_are_events():
    p = np.array([0.1, 0.5, 0.9, 0.2, 0.6, 0.1, 0.7, 0.8, 0.45])
    times = (np.arange(9) * 1.0, np.arange(9) * 1.0 + 1.0)
    assert probs_to_events(p, times) == [(1.0, 3.0, 0.9), (6.0, 9.0, 0.8)]


def test_frame_times_follow_each_slot_and_a_seam_call_is_one_event():
    lo, hi = frame_times(6 * 309, 463)
    assert lo[0] == 0 and hi[76] == pytest.approx(10.0) and lo[77] == pytest.approx(10.0)
    assert hi[154 + 77] == pytest.approx(30.0)              # slot 2 has 78 frames
    p = np.zeros(463)
    p[75:80] = 0.9                                          # across the 10 s seam
    assert len(window_events(p, 6 * 309)) == 1


def test_merge_unions_overlaps_and_keeps_the_peak():
    assert merge([(5.0, 6.0, 0.5), (1.0, 2.0, 0.9), (5.5, 7.0, 0.8)]) == [
        (1.0, 2.0, 0.9), (5.0, 7.0, 0.8)]


def test_window_starts_cover_the_file_end():
    assert window_starts(200 * 2000).tolist() == [0, 30, 60, 90, 120, 150]
    assert window_starts(20 * 2000).tolist() == [0]


def _scores():
    """1 h with 4 calls; events: 3 hits, 1 on a possible, false ones at 0.95 / 0.7 / 0.5."""
    events = {"a.wav": [(10.0, 11.0, 0.9), (20.0, 21.0, 0.6), (30.2, 31.0, 0.45),
                        (100.0, 101.0, 0.95), (200.0, 201.0, 0.7), (300.0, 301.0, 0.5),
                        (400.0, 401.0, 0.99)]}
    ann = pd.DataFrame({"file": "a.wav", "start_s": [10.0, 20.0, 30.0, 50.0, 400.0],
                        "end_s": [11.0, 21.0, 31.0, 51.0, 401.0],
                        "label": ["upcall"] * 4 + ["possible"]})
    return score_scan(events, ann, pd.Series({"a.wav": 3600.0}))


def test_recall_and_false_detections_at_tau():
    r = _scores().at_tau(0.55)
    assert r["recall_iou0.5"] == 0.5 and r["false_detections"] == 2 and r["fp_per_hour"] == 2
    assert r["precision"] == 0.5                             # the possible is judged neither way
    assert _scores().at_tau(0.4)["recall_iou0.5"] == 0.75     # 30.2-31.0 vs 30-31: IoU 0.8


def test_tau_at_a_rate_and_an_unreachable_rate():
    s = _scores()
    assert s.tau_at_rate(2.0) == 0.6                          # 0.6 and up: fps at 0.95, 0.7
    assert s.tau_at_rate(1.0) == 0.9                          # only the 0.95 false
    assert s.tau_at_rate(5.0) is None                         # 3 false at most: never 5/h
    read = s.selection_scores(5.0)
    assert not read["reachable"] and read["tau"] == 0.45 and read["recall_iou0.5"] == 0.75


def test_no_events_gives_zero_recall():
    s = ScanScores()
    s.add_file([], np.array([[1.0, 2.0]]), np.zeros((0, 2)), 3600.0)
    assert s.selection_scores(2.0)["recall_iou0.5"] == 0.0
