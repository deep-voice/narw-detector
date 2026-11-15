# narw-detector

[![CI](https://github.com/deep-voice/narw-detector/actions/workflows/ci.yml/badge.svg)](https://github.com/deep-voice/narw-detector/actions/workflows/ci.yml)
[![License: AGPL v3](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

Detects North Atlantic right whale upcalls in audio recordings. Each detection has a start time,
an end time and a confidence, and the output opens in [Raven Pro](#references).

The detector is a sound event detection network. It reads 60 s of audio at a time, as a
0–500 Hz spectrogram, and gives each frame the probability that an upcall is sounding. The 60 s
context is there to help it tell a lone upcall from a humpback song unit that repeats every few
seconds.

This repository holds the detector and the code to train it on your own annotated recordings.
The trained weights are on the [releases page](https://github.com/deep-voice/narw-detector/releases).

The model was trained on recordings and annotations from the NOAA Northeast Fisheries Science
Center (NEFSC) Passive Acoustics Branch. The work was funded by the National Fish and Wildlife
Foundation (NFWF).

## Results

The share of annotated upcalls that each detector misses, at the same number of false detections
per hour of audio. Lower is better.

### Held-out NOAA NEFSC deployments

Two deployments from the training data's region, held out from training
(`NEFSC_MA-RI_202110_NS01` and `NEFSC_MA-RI_202103_NS02` in NOAA's archive): 6,710 annotated
upcalls in 2,325 hours.

| Detector | Missed at 1 false detection per hour | Missed at 2 false detections per hour | Missed at its own threshold |
|---|---|---|---|
| This model | **30%** | **18%** | **18%**, at 1.9 false detections per hour |
| LFDCS 2.0 | 86% | 80% | 68%, at 7.2 false detections per hour |

- **This model:** a detection finds a call when the two overlap in time with an intersection
  over union (IoU) of at least 0.5. A false detection overlaps no annotated call, and no call the
  analysts marked as possible, at IoU 0.3 or more. Its own threshold is the released τ, 0.5926,
  chosen on our validation set.
- **LFDCS 2.0** ([Baumgartner and Mussoline, 2011](https://doi.org/10.1121/1.3562166)) is scored
  with its own match rule, which is looser: a detection finds a call when its midpoint falls
  inside the call, padded by 0.25 s on each side. Its own threshold is WHOI's preset
  (Mahalanobis distance of 3 or less). It ran on the same 2 kHz audio as the model, and may do
  better on the original recordings.
- **At 1 and 2 false detections per hour,** each detector's threshold was set on these
  recordings to reach that rate. These columns compare the detectors at equal rates; they are not
  thresholds to use.

### Eight NOAA days at six sites, 2008–2018

NOAA NEFSC's eight annotated days at six sites from Georgia to Stellwagen Bank
([NOAA NEFSC, 2023](https://doi.org/10.25921/2c09-ng58)), the set of Table 3 in
[Hyer et al. (2025)](https://doi.org/10.1016/j.ecoinf.2025.103130): 646 upcalls in 192 hours.
The sites are Georgia, North Carolina, the Northeast offshore, Georges Bank (three days),
Nantucket and Stellwagen Bank, recorded on MARU, HARP and AUH recorders. This model was trained
on SoundTrap recordings from 2021–2022 and has seen none of these recorders.

| Detector | Missed at 1 false detection per hour | Missed at 2 false detections per hour | Missed at 5 false detections per hour |
|---|---|---|---|
| This model's recipe, 3 training runs | **19% ± 1.5** | **11% ± 1.5** | 5% ± 0.8 |
| NARWnet (Hyer et al., 2025) | 22% | 13% | **4%** |
| LFDCS 2.0 | 60% | 52% | 41% |

- Every detector uses NARWnet's match rule: a detection finds a call when the two overlap at all.
  The 517 calls the analysts marked as possible count neither way.
- The thresholds were set on these days to reach each rate, so each value is a point on a
  detector's curve, for comparison only. To reach these rates, this model kept events down to a
  score of 0.1 instead of 0.4.
- Our row is the mean ± standard deviation of three training runs of the released model's
  recipe; the released weights are one of them. Its gaps to NARWnet are about one standard
  deviation, so the two are level. NARWnet was trained on MARU recordings from Stellwagen Bank in
  2009; MARU is the recorder at four of these six sites, Stellwagen Bank among them.
- NARWnet's row comes from the predictions released with its code, scored with its own rule; at
  its published threshold they give the paper's Table 3. LFDCS 2.0 ran with WHOI's call library
  and the vendor's MARU parameters at every site. This model and LFDCS read the Northeast offshore
  recordings (10 kHz) resampled to 2 kHz.

## Setup

You need Python 3.10 or later and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/deep-voice/narw-detector
cd narw-detector
uv sync --extra demo
```

This creates the environment in `.venv`: the detector, its commands (`narw-detect`,
`narw-train`, `narw-threshold`, `narw-mine`) and, from `--extra demo`, Jupyter and the plotting
libraries for the demo notebook. Run the commands with `uv run` (for example
`uv run narw-detect ...`), or activate the environment first with `source .venv/bin/activate`.

Then download the model from the
[release](https://github.com/deep-voice/narw-detector/releases/tag/v0.1.0) into `model/`:

```bash
mkdir -p model
curl -L -o model/config.json https://github.com/deep-voice/narw-detector/releases/download/v0.1.0/config.json
curl -L -o model/weights.pt https://github.com/deep-voice/narw-detector/releases/download/v0.1.0/weights.pt
```

`config.json` records the SHA-256 of `weights.pt`
(`cf80622ab71826f772341626059933b40c3ee6dae7eb3c8c6803ec4e0750a320`), and the detector checks it
on load.

For development, `uv sync --all-extras` adds the test tools: `uv run pytest`,
`uv run ruff check src tests`. To use the commands without cloning the repository:
`pip install "narw-detector @ git+https://github.com/deep-voice/narw-detector"`.

## Try the demo

```bash
uv run jupyter lab notebooks/demo.ipynb
```

The notebook runs the detector on 20 minutes of a NOAA recording that was held out from
training (about 40 MB, from NOAA's public archive), and lets you look at and listen to each
detection.

## Detect upcalls

```bash
narw-detect recordings/ --model model/ --out detections/
```

`recordings/` can be one audio file or a folder; folders are searched recursively for `.wav`,
`.flac`, `.aif` and `.aiff` files. Any sample rate works: every file is resampled to 2 kHz, the rate
the model was trained on. Only the first channel is read. Detection runs on a CUDA GPU when
there is one, and otherwise on the CPU (`--device` to choose).
Apple silicon (`--device mps`) is used only when asked for: some PyTorch versions crash running
the model on it.

### Output

- `detections/detections.csv`: one row per detected call:

  | Column | Meaning |
  |---|---|
  | `file` | the recording |
  | `start_s`, `end_s` | the call, in seconds from the start of the file |
  | `confidence` | the peak frame probability of the detection, 0–1 |

- `detections/<recording>.selections.txt`: one Raven Pro selection table per recording. Open the
  recording in Raven, then *File > Open Selection Table*. Each selection spans the call in time.
  The detector does not measure frequency, so every selection gets a fixed 50–250 Hz band. A
  `Confidence` column holds each detection's confidence.

### The operating point

A detection is reported when its confidence reaches the model's threshold τ, stored in
`config.json`. The released model's τ is 0.5926, chosen on our internal validation set (not
distributed). To trade missed calls against false detections on your own recordings, see
[Choose the threshold](#choose-the-threshold-for-your-recordings).

## Limitations

- **Training data.** The model was trained on SoundTrap recordings from southern New England
  (NOAA NEFSC, 2021–2022). On other recorders, sites and noise conditions it may miss more calls
  or raise more false detections. Check it on a few annotated days of your own before relying on
  it.
- **Time only.** The detector gives each call's start and end, not its frequency range.

## Train it on your own recordings

You need recordings with upcalls annotated, and a separate set of fully annotated validation
recordings. **Our own training data (NOAA NEFSC's recordings and annotations) is not
distributed**, so retraining starts from your data.

```bash
narw-train --audio train/audio --annotations train/selections/ \
           --val-audio val/audio --val-annotations val/selections/ --out run/
```

### Input formats

- **Recordings:** a folder of audio files at any sample rate (searched recursively). File names
  must be unique within the folder. Files must be at least 60 s long.
- **Annotations:** Raven Pro selection tables: one `.txt` file, or a folder of them.
  - A table made over one recording must be named after it, as Raven names it by default:
    `<recording name>.Table.1.selections.txt`.
  - A table made over a file sequence must have the `Begin File` and `File Offset (s)` columns,
    which place each call in its file.
  - `Begin Time (s)` and `End Time (s)` give each call.
  - Or one CSV with columns `file`, `start` and `end`, in seconds from the start of the file.
  - Optional `label` column, in either format: `upcall` (the default) or `possible`, for a call
    you are unsure of. A possible is never used as a call or as background, and a detection on a
    possible is not counted as false.
- **Validation recordings:** recordings with **every** upcall annotated, from other days or
  sites than the training set. After each epoch, the model scans them end to end. Any
  unannotated call there counts against the model.
- **Optional, for training on hard examples:** two CSVs of 10 s tiles (`file`, `start`, `end`),
  usually written by `narw-mine` from humpback song (see below):
  - `--hard-negatives`: sounds the model should learn to ignore;
  - `--song-positives`: upcalls inside song, trained at four times the weight.

### What training does

The recipe is the released model's:

- **Training audio.** 10 s tiles are taken on a 5 s grid: every tile that holds a whole
  annotated upcall, and as many call-free tiles. Each tile is placed in one of the six 10 s slots
  of a 60 s window.
- **Loss.** Focal loss (Lin et al., 2017) on that slot only.
- **Augmentation.** FilterAugment (Nam, Kim and Park, 2022) and SpecAugment (Park et al., 2019).
- **Optimiser.** AdamW (Loshchilov and Hutter, 2019) at learning rate 10⁻³ and weight decay 10⁻⁴, with a batch of 64.
- **Selection.** After each epoch, the model scans the validation recordings and measures the
  share of calls found at the threshold that allows 2 false detections per hour. The best epoch
  is kept. Training stops after 7 epochs without improvement, or at 40 epochs.

`run/model/` holds the best epoch's weights. Its `config.json` holds the threshold that epoch was
read at. `run/history.csv` logs every epoch. Use the trained model with
`narw-detect ... --model run/model/`.

**Retraining won't reproduce the released checkpoint exactly.** The released checkpoint was
selected on our internal validation set, with our own selection rule. The public code selects on
your validation recordings, so even with the same training data it won't pick the same epoch.

More detail is in [docs/training.md](docs/training.md).

### Choose the threshold for your recordings

```bash
narw-threshold --model run/model/ --audio val/audio --annotations val/selections/ \
               --fp-per-hour 2 --write
```

This scans fully annotated recordings. At 1, 2 and 5 false detections per hour (and at
`--fp-per-hour`), it prints the threshold and the share of calls found there. `--write` stores
the `--fp-per-hour` threshold in the model's `config.json`. Use recordings the model did not
train on.

### Hard examples from humpback song (optional)

Humpback song units can look like upcalls. The released model was also trained on song mined
from LFDCS pitch tracks: runs of near-identical units repeating every few seconds.
`narw-mine` turns an LFDCS run over your training recordings into the two hard-example tables.
LFDCS is WHOI's software and is not included. [docs/lfdcs.md](docs/lfdcs.md) gives the settings
we ran it with, what `narw-mine` needs from the run, and how to mine its output.

## References

- Baumgartner, M. F., and Mussoline, S. E. (2011). A generalized baleen whale call detection and
  classification system. *The Journal of the Acoustical Society of America* 129(5), 2889–2902.
  https://doi.org/10.1121/1.3562166
- Hyer, M. D., Anderson, A. T., Mann, D. A., Mooney, T. A., Aoki, N., and Jensen, F. H. (2025).
  Robust real-time detection of right whale upcalls using neural networks on the edge.
  *Ecological Informatics* 89, 103130. https://doi.org/10.1016/j.ecoinf.2025.103130
- K. Lisa Yang Center for Conservation Bioacoustics at the Cornell Lab of Ornithology. *Raven Pro:
  Interactive Sound Analysis Software.* https://ravensoundsoftware.com/
- Lin, T.-Y., Goyal, P., Girshick, R., He, K., and Dollár, P. (2017). Focal loss for dense object
  detection. *ICCV 2017.* https://arxiv.org/abs/1708.02002
- Loshchilov, I., and Hutter, F. (2019). Decoupled weight decay regularization. *ICLR 2019.*
  https://arxiv.org/abs/1711.05101
- Nam, H., Kim, S.-H., Ko, B.-Y., and Park, Y.-H. (2022). Frequency dynamic convolution:
  frequency-adaptive pattern recognition for sound event detection. *Interspeech 2022.*
  https://arxiv.org/abs/2203.15296
- Nam, H., Kim, S.-H., and Park, Y.-H. (2022). FilterAugment: an acoustic environmental data
  augmentation method. *ICASSP 2022.* https://arxiv.org/abs/2110.03282
- NOAA Northeast Fisheries Science Center. NEFSC Southern New England bottom mounted PAM audio data.
  NOAA National Centers for Environmental Information. https://doi.org/10.25921/znjw-at57
- NOAA Northeast Fisheries Science Center (2023). The North Atlantic right whale annotations from
  passive acoustic data collected in the western North Atlantic Ocean. NOAA National Centers for
  Environmental Information. https://doi.org/10.25921/2c09-ng58
- Park, D. S., Chan, W., Zhang, Y., Chiu, C.-C., Zoph, B., Cubuk, E. D., and Le, Q. V. (2019).
  SpecAugment: a simple data augmentation method for automatic speech recognition. *Interspeech
  2019*, 2613–2617. https://arxiv.org/abs/1904.08779

## Citation

Naama Marom-Yochai, Michael Faran, Michael Michelachvili, Doron Shachar (Deep Voice Foundation).
*80,000 Upcalls Under the Sea: Teaching AI to Hear North Atlantic Right Whales.* North Atlantic
Right Whale Consortium Annual Meeting, 2026.

GitHub's *Cite this repository* button gives the same citation, from [CITATION.cff](CITATION.cff).

## Acknowledgments

The NOAA Northeast Fisheries Science Center (NEFSC) Passive Acoustics Branch, for the recordings
([NOAA NEFSC, NCEI](https://doi.org/10.25921/znjw-at57)) and annotations the model was trained
on, and the eight-day annotated set ([NOAA NEFSC, 2023](https://doi.org/10.25921/2c09-ng58)). The National Fish and Wildlife Foundation (NFWF), for
funding this work through a vessel strike reduction grant. Hyer et al. (2025), for releasing
their model's predictions with their code.

## Contact

naama@deepvoicefoundation.com

## License

The code and the released model weights are licensed under the GNU Affero General Public
License v3.0 (see [LICENSE](LICENSE)).
