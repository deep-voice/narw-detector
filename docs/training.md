# Training on your own recordings

`narw-train` reproduces the recipe that trained the released model. Every step below is the
released recipe, except where it says otherwise.

## Data

### Annotations

Raven Pro selection tables, or one CSV (`file`, `start`, `end`, optional `label`); the README
lists the rules. A few details:

- A Raven table that has both a *Spectrogram* and a *Waveform* row for each selection is read
  once per selection (the Spectrogram rows).
- A call that runs past the end of its file is dropped: its end is in the next file. The
  original training data dropped calls split across two files in the same way.
- A zero-length annotation is kept as a one-frame call in training. In validation it is not
  counted as a call.

### Which audio the model trains on

1. **Tiles.** In every recording with at least one upcall, 10 s tiles start every 5 s from 0.
   - A tile that holds a whole upcall is a positive tile.
   - A tile that overlaps an upcall but holds none whole is skipped.
   - Tiles that overlap no upcall and no possible are background candidates. As many are drawn
     as there are positive tiles (seed 0).
   - Recordings without upcalls give no tiles. To train on them, list stretches of them as hard
     negatives.
2. **Hard negatives and song positives** (optional) are added as given. A tile listed twice is
   trained at the sum of its weights.
3. **Windows.** The model reads 60 s: six 10 s slots. Each tile goes to one of the six slots.
   - Only placements whose 60 s window lies inside the recording and holds no possible are
     allowed.
   - A fixed hash of the tile picks the slot among them, so the placement is the same every run.
   - A tile with no allowed placement is dropped, so recordings under 60 s give no training
     windows.
4. **Targets.** Each 10 s slot has 309 spectrogram frames. Every frame inside an annotated upcall
   is 1. A call that the window cuts keeps its in-window part.
   - The loss is taken on the tile's slot only. The other five slots are context the model reads
     but is not scored on.
   - Targets always come from the annotations, whatever list a tile came from, so an upcall
     inside a hard-negative tile is still taught as an upcall.

### Audio

Every recording is resampled to 2 kHz with `scipy.signal.resample_poly`, the same way as for
detection. Files at another rate are resampled once into `OUT/audio_2k/`. Files already at 2 kHz
are read in place.

## Recipe

| | |
|---|---|
| Input | 60 s windows, six 10 s slots, each slot its own spectrogram (n_fft 256, hop 64 at 2 kHz, 0–500 Hz); the window's dB image standardised as a whole |
| Network | dual-stream SED: a frequency-dynamic CNN (Nam et al., 2022) on each slot, a second one plus a BiGRU over the whole window, a BiGRU per slot over both, a linear head; width 32, 262,481 parameters |
| Loss | focal binary cross-entropy (Lin et al., 2017; γ 2, α 0.25) per output frame, on the target slot only |
| Augmentation | FilterAugment (Nam, Kim and Park, 2022; 3–5 bands, ±6 dB); SpecAugment (Park et al., 2019; 2 frequency masks of up to 8 rows, 2 time masks of up to 30 frames per 10 s) |
| Optimiser | AdamW (Loshchilov and Hutter, 2019), learning rate 10⁻³ constant, weight decay 10⁻⁴ |
| Batch | 64: 32 windows per step, gradients accumulated over 2 steps (`--accumulate 4` gives 16 per step for a small GPU, with the same batch of 64) |
| Length | at most 40 epochs; stop after 7 without a better validation recall |

## Validation and selection

After each epoch the model scans every validation recording end to end:

1. Windows are 60 s long with a 30 s hop, and the last one is zero-padded.
2. Frame probabilities above 0.4 for at least two frames form an event. Its confidence is the
   peak probability.
3. Events from overlapping windows are merged before any threshold is applied.

Then:

- A call is **found** when an event reaching the threshold overlaps it with temporal IoU ≥ 0.5.
- An event is a **false detection** when it overlaps no call, and no possible, at IoU ≥ 0.3.
- The epoch's score is the share of calls found at the lowest threshold that keeps false
  detections to at most 2 per scanned hour (`--fp-per-hour`). If the model stays under that rate
  even with every event counted, the score counts every event.

The best epoch is kept. Its threshold becomes the model's operating point.

## Differences from the released model's training

- **Validation set and selection rule.** We selected the released checkpoint on our internal
  validation set, with our own rule. The public code selects on your recordings, so retraining
  won't pick the same epoch.
- **Hard-negative sampling.** `narw-mine` caps the song tiles at 30,000, stratified by pool and
  stereotypy. The original also stratified by deployment; recordings have no deployment here.
- **Positive tiles.** The original lattice kept at most 3,000 positive tiles per deployment (a
  seeded sample) and matched the background to them. With no deployments there is no cap: every
  positive tile is used, with as many background tiles.
