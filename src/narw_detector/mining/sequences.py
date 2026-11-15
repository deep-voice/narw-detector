"""Voices, runs and stereotypy over a detection stream: finding repeated song.

Song repeats within one singer, while a stream that mixes several sources looks irregular, so
the stream is first split by source:

1. group_voices: a greedy tracker assigns each event to an active voice whose recent members it
   resembles (descriptor similarity) and which was heard recently.
2. find_runs: within a voice, chains of events whose inter-call intervals (ICIs) stay regular.
   Each ICI is compared with the median of the run's recent ICIs, so the tempo may drift slowly
   and one missed unit (an ICI about twice the median) does not break the run. A run qualifies
   with at least RUN_MIN_LEN members and a median ICI of at most RUN_MAX_MEDIAN_ICI_S. That
   limit, in seconds, keeps out right whale counter-calling, which repeats over minutes.
3. stereotypy: an event's median similarity to the run members nearest it in time, itself
   excluded.

The run settings are the ones that built the released model's training set (min length 6,
median ICI at most 8 s).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from narw_detector.mining.descriptors import similarity_matrix

# Voice tracking
VOICE_MAX_GAP_S = 60.0       # a voice with nothing for this long is closed
VOICE_MIN_SIM = 0.35         # join threshold against the voice's recent template
VOICE_TEMPLATE_K = 5         # how many recent members form the template

# Run qualification
RUN_MIN_LEN = 6              # members of a qualifying run
RUN_MAX_MEDIAN_ICI_S = 8.0   # the longest median ICI a run may have
RUN_MAX_START_ICI_S = 15.0   # an ICI longer than this never starts a run
RUN_GAP_FACTOR = 2.5         # one missed unit (~2x local median) does not break the run
RUN_LOCAL_K = 5              # ICIs in the local-median window

STEREOTYPY_K = 4             # temporal neighbours per stereotypy score


@dataclass
class Run:
    """One qualifying rhythmic sequence: member indices into the stream it was mined from."""
    members: np.ndarray          # int indices, time-ordered
    t0: float
    t1: float
    median_ici_s: float


def group_voices(times: np.ndarray, shapes: np.ndarray, logf: np.ndarray, logdur: np.ndarray,
                 max_gap_s: float = VOICE_MAX_GAP_S, min_sim: float = VOICE_MIN_SIM,
                 template_k: int = VOICE_TEMPLATE_K) -> np.ndarray:
    """Assign every event a voice id. Times must be sorted.

    An event joins the best-matching active voice (last member within max_gap_s, median
    similarity to the voice's last template_k members at least min_sim), or starts a new voice.
    When no voice is similar enough the tracker starts a new one rather than merging, because a
    voice that mixes two singers would mix their units in one stereotypy score.
    """
    n = len(times)
    assert shapes.shape[0] == n and len(logf) == n and len(logdur) == n
    assert np.all(np.diff(times) >= 0), "times must be sorted"

    voice_id = np.full(n, -1, dtype=np.int64)
    # The active voices, as parallel lists: last event time, member indices, voice id.
    active_last: list[float] = []
    active_members: list[list[int]] = []
    active_ids: list[int] = []
    next_id = 0

    for i in tqdm(range(n), desc="group_voices", unit="ev", mininterval=30, ascii=True):
        t = times[i]
        keep = [j for j, last in enumerate(active_last) if t - last <= max_gap_s]
        active_last = [active_last[j] for j in keep]
        active_members = [active_members[j] for j in keep]
        active_ids = [active_ids[j] for j in keep]

        best_j, best_sim = -1, min_sim
        if active_members:
            tails = [m[-template_k:] for m in active_members]
            flat = [k for tail in tails for k in tail]
            sims = similarity_matrix(shapes[i:i + 1], logf[i:i + 1], logdur[i:i + 1],
                                     shapes[flat], logf[flat], logdur[flat])[0]
            pos = 0
            for j, tail in enumerate(tails):
                s = float(np.median(sims[pos:pos + len(tail)]))
                pos += len(tail)
                if s >= best_sim:
                    best_j, best_sim = j, s

        if best_j >= 0:
            voice_id[i] = active_ids[best_j]
            active_last[best_j] = t
            active_members[best_j].append(i)
        else:
            voice_id[i] = next_id
            active_last.append(t)
            active_members.append([i])
            active_ids.append(next_id)
            next_id += 1
    return voice_id


def find_runs(times: np.ndarray, voice_id: np.ndarray,
              min_len: int = RUN_MIN_LEN,
              max_median_ici_s: float = RUN_MAX_MEDIAN_ICI_S,
              max_start_ici_s: float = RUN_MAX_START_ICI_S,
              gap_factor: float = RUN_GAP_FACTOR,
              local_k: int = RUN_LOCAL_K) -> list[Run]:
    """Qualifying runs within each voice.

    A chain accepts the next ICI while it is at most gap_factor times the median of the last
    local_k accepted ICIs; the first ICI must be at most max_start_ici_s. A chain qualifies as a
    run with at least min_len members and a median ICI of at most max_median_ici_s.
    """
    assert len(times) == len(voice_id)
    runs: list[Run] = []
    for v in np.unique(voice_id):
        idx = np.flatnonzero(voice_id == v)
        if len(idx) < min_len:
            continue
        chain = [idx[0]]
        icis: list[float] = []
        for k in range(1, len(idx) + 1):
            ici = times[idx[k]] - times[idx[k - 1]] if k < len(idx) else np.inf
            limit = (gap_factor * float(np.median(icis[-local_k:]))
                     if icis else max_start_ici_s)
            if ici <= limit:
                chain.append(idx[k])
                icis.append(float(ici))
                continue
            if len(chain) >= min_len:
                med = float(np.median(icis))
                if med <= max_median_ici_s:
                    m = np.asarray(chain)
                    runs.append(Run(m, float(times[m[0]]), float(times[m[-1]]), med))
            if k < len(idx):
                chain, icis = [idx[k]], []
    return runs


def stereotypy(t_query: np.ndarray, shapes_q: np.ndarray, logf_q: np.ndarray,
               logdur_q: np.ndarray, run: Run,
               times: np.ndarray, shapes: np.ndarray, logf: np.ndarray, logdur: np.ndarray,
               k: int = STEREOTYPY_K, self_tol_s: float = 0.05) -> np.ndarray:
    """Each query event's median similarity to the k members of `run` nearest it in time.

    times, shapes, logf and logdur describe the stream the run was mined from. Members within
    self_tol_s of the query are skipped, so a run member is not compared with itself.
    """
    m = run.members
    sim = similarity_matrix(shapes_q, logf_q, logdur_q, shapes[m], logf[m], logdur[m])
    dt = np.abs(t_query[:, None] - times[m][None, :])
    sim = np.where(dt <= self_tol_s, np.nan, sim)
    order = np.argsort(dt, axis=1)
    ranked = np.take_along_axis(sim, order, axis=1)
    scores = np.full(len(t_query), np.nan)
    for i in range(len(t_query)):
        row = ranked[i][~np.isnan(ranked[i])][:k]
        if len(row):
            scores[i] = float(np.median(row))
    return scores


def run_member_stereotypy(run: Run, times: np.ndarray, shapes: np.ndarray, logf: np.ndarray,
                          logdur: np.ndarray, k: int = STEREOTYPY_K) -> np.ndarray:
    """Stereotypy of each run member against its own run (self excluded)."""
    m = run.members
    return stereotypy(times[m], shapes[m], logf[m], logdur[m], run,
                      times, shapes, logf, logdur, k=k)
