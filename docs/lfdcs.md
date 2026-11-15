# Hard examples from humpback song: LFDCS, then `narw-mine`

Humpback song units can look like upcalls. The released model also trained on song mined from
LFDCS's pitch tracks:

- song units became "no upcall" tiles;
- upcalls sung over song were trained four times per epoch.

This page covers both steps:

1. Run LFDCS over your training recordings.
2. Run `narw-mine` on its output.

## 1. Run LFDCS

LFDCS (the Low-Frequency Detection and Classification System; Baumgartner & Mussoline, 2011) is
WHOI's software. It is not part of this repository, and not ours to redistribute. Ask its
developers at WHOI for the distribution (LFDCS 2.0) and the GOM9 call library, and run it as the
documentation that comes with the distribution describes.

### Parameter file

The settings we used for SoundTrap recordings at 2 kHz, with the GOM9 call libraries. Replace
the `<...>` fields:

```text
             Experiment: <any name>
                  Indir: <absolute path to your training recordings>
                 Outdir: <absolute path for LFDCS output>
              StartDate: 01/01/06 00:00:00
           PlatformType: Mooring
             PlatformID: <recorder id>
         InstrumentType: Soundtrap
           FileDateSpec: .YYMMDDhhmmss.
          FileExtension: wav
             FileFormat: WAV
           InstrumentID: 1
             Originator: <you>
               Location: <site>
               TimeZone: UTC
           FileDuration: 3600.0
                  Frame: 512
                Overlap: 0.75
                 Smooth: 1

        CallLibraryFile: <LFDCS_HOME>/supplemental/call_library/clnb_gom9/call_library_gom9.nc
      CallLibraryFileBB: <LFDCS_HOME>/supplemental/call_library/clbb_gom/call_library_broadband_gom.nc

  DetectionFileDuration: 5.0

                  Scale: 0.0030
                 Offset: -100.0

    SpectrogramDuration: 30.0
    PitchTrackingWindow: 25.0
   NoiseReductionWindow: 60.0
     AvgFFTLowThreshold: -999.0
    AvgFFTHighThreshold:  999.0
    AvgFFTDurationLimit: 300.0

        BBP_InThreshold: 5.0
         BBP_InDuration: 1.5
       BBP_OutThreshold: 5.0
        BBP_OutDuration: 4.0
        BBP_MaxDuration: 300.0

  BB_DetectionThreshold: 9.6
      BB_MinSegmentSpan: 150.0
        BB_MinTotalSpan: 200.0
BB_MinBroadbandDuration: 0.125

     DetectionThreshold: 10.0
  CostGradientThreshold: 15.0
      DistanceWeighting: 20.0

        MinCallDuration: 0.25
        MinAvgAmplitude: 10.0
           BlankingTime: 0.25
           BlankingFreq: 20.0
```

### What `narw-mine` needs from the run

- **The GOM9 call libraries**, as in the parameter file: `narw-mine` selects song units by their
  GOM9 call group.
- **File names that sort in time.** `narw-mine` maps LFDCS's clock back to files in name order.
  `FileDateSpec` tells LFDCS where the timestamp sits in each file name: `.YYMMDDhhmmss.` matches
  SoundTrap names such as `1234.210417200407.wav`. Change it to match yours.
- **One run per continuous recording period**, over exactly the files you then give to
  `narw-mine`.

## 2. Mine its output

```bash
narw-mine --lfdcs-out <Outdir> --audio train/audio --annotations train/selections/ --out mined/
narw-train ... --hard-negatives mined/hard_negatives.csv --song-positives mined/song_positives.csv
```

Install the netCDF reader first: `pip install "narw-detector[mine] @ git+https://github.com/deep-voice/narw-detector"`.

### What `narw-mine` does

1. **Reads LFDCS's detections.** For each detection it reads the pitch track, the GOM9 call
   group, the duration and the frequency range. It places each detection in its file from LFDCS's
   clock and the files' true durations.
2. **Finds song.** Similar units are grouped into *voices*. A voice yields a *run* when it holds
   at least 6 units and the median interval between units is at most 8 s. Each unit gets a
   stereotypy score: its similarity to its neighbours in the run.
3. **Hard negatives.** Every unit in a run with a right-whale-like or humpback call group (GOM9
   groups 5–10, 15–20 and 23–25) gives the 10 s tile on the 5 s grid nearest the unit.
   - A tile is dropped when an annotated possible lies within 15 s of its centre.
   - One tile per grid position.
   - At most 30,000 tiles are kept, sampled evenly across call-group pools and stereotypy
     terciles.
   - Tiles are not checked for upcalls: training targets come from the annotations.
4. **Song positives.** An annotated upcall whose centre lies in a run, or within 30 s of one,
   gives the 10 s tile starting near it. Training uses it at weight 4.

Both tables have columns `file`, `start`, `end` (10 s tiles). `hard_negatives.csv` also lists
each tile's pool and stereotypy.
