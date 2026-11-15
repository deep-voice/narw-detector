# AGENTS.md

Guidance for coding agents, and people, working in this repository. It holds rules and the facts
you need to follow them. What the detector does and how to use it is in the [README](README.md).
`CLAUDE.md` imports this file, so Claude Code reads it too.

## Repository map

- `src/narw_detector/`, the package:
  - **Commands:** `detect.py` (`narw-detect`), `train.py` (`narw-train`), `threshold.py`
    (`narw-threshold`), `mining/cli.py` (`narw-mine`).
  - **Inference:** `audio.py` reads any audio file and resamples it to 2 kHz. `spectrogram.py`
    makes the model's input. `model.py` is the network. `scan.py` scans a whole recording in
    60 s windows at a 30 s hop. `events.py` turns frame probabilities into call events.
    `raven.py` reads and writes Raven Pro selection tables. `weights.py` loads a model folder and
    checks its checksum.
  - **Training:** `annotations.py`, `windows.py`, `dataset.py`, `augment.py`, `loss.py`,
    `hard_negatives.py`. `evaluate.py` holds the scoring rules that training and `narw-threshold`
    use.
  - **Mining:** `mining/` turns an LFDCS run into hard-negative and song-positive tiles.
- `tests/`: one test file per area of the package.
- `docs/training.md` (the training recipe) and `docs/lfdcs.md` (LFDCS settings for `narw-mine`).
- `notebooks/demo.ipynb`: runs the released model on a public NOAA recording.
- The trained model is not in git. It is two assets of the GitHub Release `v0.1.0`: `weights.pt`
  (a PyTorch state_dict) and `config.json` (the architecture settings, the threshold τ with its
  source, and the SHA-256 of `weights.pt`). The README's Setup downloads them into `model/`.

## Setup and workflow

- `uv sync --all-extras` installs everything: the package, the test tools and the demo's Jupyter.
- `uv run pytest` and `uv run ruff check src tests` must pass. CI runs both on Python 3.10 and
  3.12, on CPU.
- Commit notebooks with their outputs cleared.
- Keep the README and `docs/` in step with the code: commands, flags, defaults and file formats.

### What never goes in git

- Audio, annotations, datasets, weights (`*.pt`, `*.ckpt`, `model/`), detections, training runs
  (`run/`) and caches. `.gitignore` covers them. Weights ship as release assets, with their
  checksum in `config.json`.
- Before each commit, check the staged files and their sizes. If anything large, binary or
  data-like is staged, stop.

## Code

- Small single-purpose functions with clear names. Readable beats clever.
- Write for the use in front of you: no parameters, branches or abstractions for hypothetical
  uses. No defensive code for inputs that cannot occur.
- pandas: vectorized built-ins (`groupby`, `merge`, boolean masks) over `iterrows` and loops.
  Build DataFrames column by column.
- **Errors.** A mistake in what the user gave (command-line arguments, audio files, annotation
  tables, a model folder) raises `ValueError` or calls the parser's `error()`, with a message
  that says what to fix. An assumption about our own data (shapes, equal lengths, expected
  columns) is an `assert`.
- Console scripts: `main(argv=None)` returns `None`. A returned value becomes the exit status.
- Python 3.10 compatible: for example `datetime.timezone.utc`, not `datetime.UTC`.
- Devices: `pick_device("auto")` picks CUDA if available, else CPU. Apple's MPS is used only on
  request: some PyTorch versions (2.15 on macOS 14) crash running the model's GRUs on it.
- Ruff settings are in `pyproject.toml` (line length 100).

## Tests

- A new module comes with tests in `tests/`.
- A network gets three tests (see `tests/test_model.py`):
  - shape: a forward pass on a dummy batch gives the exact output shape;
  - smoke: one training step gives a finite loss;
  - overfit: one tiny synthetic batch is fitted to near-zero loss in under 30 s on CPU.
- Tests use synthetic tensors and small audio files written to `tmp_path`. Never real
  recordings, annotations, the network or the released weights.
- Assert shapes where data changes form (audio to spectrogram, spectrogram to frames).

## The released model

- Changes to `audio.py`, `spectrogram.py`, `model.py`, `events.py` or `scan.py` can change what
  the released weights compute. Run `narw-detect` on the same recording before and after such a
  change, compare the detections, and report the result.
- Do not edit a released `config.json`: its τ and checksum belong to the release.

## Evaluation and claims

- **Thresholds come from validation recordings only:** fully annotated recordings, separate from
  the training data. Never set a threshold on the recordings you report results on. A threshold
  set there only gives a comparison point, and must be labelled so (as in the README's
  fixed-rate columns).
- **Every reported number states** the recordings (calls and hours), the unit (one annotated
  call), the match rule and the false-detection rate or threshold.
  - The rules are in `evaluate.py`: a detection finds a call at IoU 0.5 or more; a false
    detection overlaps no call, and no possible call, at IoU 0.3 or more. Other rules (such as
    any overlap) are named where they are used.
  - Do not change the rules silently.
- **One training run is a screen, not a finding.** Compare training recipes over at least three
  runs (seeds) and report the mean ± standard deviation.
- **Possible calls** (label `possible`) are never trained on as calls or as background, and a
  detection on one counts neither way.
- **Check a number before reporting it.** Could it be wrong? Does it make sense next to the
  others?
- The README's results and the released τ come from the maintainers' evaluations on data that is
  not distributed. Do not edit them.
- In docs and comments, state what was measured, with its numbers. No unmeasured claims ("fast",
  "robust"), and no hedging of what was measured.

## Domain facts

- **Calls.** North Atlantic right whale upcalls rise through about 90–210 Hz in roughly one
  second. Humpback song units can look like them; they repeat every few seconds, which is what
  the 60 s context and the mined song tiles are for.
- **Model input.** The first channel only, resampled to 2 kHz. A 0–500 Hz spectrogram (`n_fft`
  256, hop 64), a 60 s window of six 10 s slots.
- **Decoding.** An event is a run of at least 2 frames above 0.4; its confidence is its highest
  frame probability. A detection is an event whose confidence reaches τ.
- **Annotations.** Raven Pro selection tables, or a CSV (`file`, `start`, `end`, optional
  `label`): see the README's Input formats. Output tables give every selection a fixed 50–250 Hz
  band, because the detector localises calls in time only.
- **LFDCS** is WHOI's licensed software. This repository only reads its output. Do not add LFDCS
  code, binaries, call libraries or run instructions beyond `docs/lfdcs.md`.
- **Data.** The demo reads a public NOAA recording. When you use NOAA data, cite NOAA's dataset
  (see the README's References). Do not add downloads of data that is not public.

## Spectrograms and audio in figures

- **The display transform is not the model's.** For display at 2 kHz: 512-sample windows, 448
  overlap, zero-padded to 1024 (about 2 Hz per row). Only a figure of the model's input matches
  `spectrogram.py`.
- **Frequency increases upward** (`origin="lower"`, rows from low to high). Say so in the
  plotting function's docstring: a flipped axis turns an upcall into a downcall.
- **Levels.** Never scale a picture by its own peak. Put black at the noise floor (the median dB)
  and choose the range above it.
  - Comparing over time (a long overview): one level for the whole stretch.
  - Judging one call: each picture gets its own levels.
- **Wide views:** combine columns by their maximum, so a one-second call stays visible.
- **Detections:** brackets at the start and end and a bar along the top. No fill: it hides the
  call.
- **A figure that shows detections states τ** and where τ came from.
- **Style:** `sns.set_theme()` once per module or notebook; spectrograms use `magma`.
- **Audio in a browser:** resample to 4 kHz or more, because browsers do not play 2 kHz. A faster
  player raises the pitch (the demo uses 2×).
