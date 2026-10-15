# tornado-onset: project notes

The write-up is `report.qmd` at the repo root (Quarto, PDF target; `quarto render report.qmd`).
This file is the running log: results, decisions, open questions.

## The question

Given the radar scans observed so far, will a tornado appear in the next 5 and 10 minutes when
none is visible yet?

That last clause defines the evaluation. A **pre-onset case** is a tornado sample in which no
input scan is tornadic and the target scan is. Scoring restricts the positives to those cases
and keeps every null, so the number answers "will one appear", not "will one persist". The
contrast that matters is warned storms vs confirmed tornadoes (the WC view): clear air is easy.

## Data

TorNet v1.1. Each sample is four radar scans about five minutes apart, each 120 azimuths by 240
range gates by two tilts for six products (DBZ, VEL, KDP, RHOHV, ZDR, WIDTH), a range-folded mask
for the two Doppler products, a tornado label per scan, and event metadata. Inputs are the two
oldest scans; targets are the two newest, at about +5 and +10 minutes from the last input.

The data comes from the Hub dataset `jakeryderv/tornet` (Parquet, every netCDF field, one
config per year plus a `catalog` config of the scalar fields; published 2026-10-14 from the
original Zenodo release, CC BY 4.0). `shard_paths` fetches the shards of `train_years` /
`test_years`, or explicit `data_files` patterns, into the Hub file cache, and `OnsetDataset`
streams them with pyarrow one shard at a time (a shard is 256 samples, 170 MB compressed,
0.8 GB decoded), shuffling shard order per epoch and rows within a shard; DataLoader workers
take disjoint shards. The `datasets` library was tried first and dropped: it materialises an
uncompressed Arrow copy, about 3 MB per sample, 75 GB for 2015 alone. No local netCDF, no
`TORNET_ROOT`.

Partitions are the authors' `train` and `test` splits. Validation is held out from the training
partition by whole days, never by random sample, because consecutive scans of one storm are near
duplicates. The first slice is 2015: 19,721 training files (16,717 train / 3,004 validation over
180 / 32 days) and 3,902 test files.

Pre-onset positives are rare by construction, roughly 200 per test year, so every pre-onset
number carries an interval of about ±0.025 AUC. Differences under about 0.03 are not resolvable
on one year.

## Model

The published TorNet detector supplies both the starting point and the judge. Its Keras
checkpoint was converted once to a torch state dict (h5py only, no Keras; the conversion
reproduces the published logits to 2e-5, see the model card) and published as
`jakeryderv/tornet-detector-torch` (MIT, as the original). The conversion code lived in this
repo until the trim of 2026-10-14 and is in the git history; `tornet.resolve_weights` fetches
the published state dict unless a local `weights/tornet_detector_v1.pt` exists.

```
two scans (14 channels each: 12 normalised radar channels + 2 range-folded)
    |
TorNet encoder (frozen or fine-tuned per step), run per scan; bottleneck features concatenated
    |
the authors' head: 1x1 convs to a per-cell logit map, max over cells
    -> P(tornado at +5), P(tornado at +10)
```

**Dropped 2026-10-14: the extrapolate combiner.** A fixed time rule (features extrapolated
along the real scan spacing, one shared head started from the published head) was coded as an
alternative to concatenation but never run. It was a sideways step rather than a rung on the
ladder below, so it went with the trim; the natural temporal step later is a longer history,
not a different combiner.

4.9M parameters. One loss: weighted cross-entropy on the head's per-horizon logits against the
labels of the future scans (the authors' sample weights: EF2+ ×2, warnings ×0.5). AdamW, the
encoder at a tenth of the learning rate, cosine schedule, mixed precision, early stopping on
the mean validation head AUC (`patience` 2, cap `epochs` 8); the best checkpoint is scored.

The frozen published detector (**the judge**) never trains. At evaluation it reads the newest
observed scan (**persistence**, the floor: assume the future looks like the present) and the
real future scan (**truth**, the ceiling: a perfect forecast).

**Dropped 2026-10-14: the radar-drawing decoder.** The first version carried a U-Net decoder
drawing the future velocity and reflectivity, trained by a heat-map-weighted L1 and the judge's
cross-entropy on the drawn chip, as an auxiliary signal for the shared encoder. On 2015 the
head-only model matched or beat it (0.739 / 0.735 vs 0.716 / 0.729 pre-onset WC AUC at +5 /
+10 min; +0.023 [+0.004, +0.043] paired at +5) with half the parameters and 30% less time per
epoch, so it went, together with the segmentation_models_pytorch dependency. The last version
with it is commit `82567d5`; its run directories were removed (numbers kept below).

## Evaluation

Per horizon, four probabilities per test sample: head, judge on persistence, judge on forecast,
judge on truth. Each is written as a predictions CSV with the label pattern (e.g. `0011`) and
`pre_onset` / `ongoing` flags. `summary.json` holds AUC and AUCPR per view (NC, RC, WC) with
bootstrap intervals, for all cases and for the pre-onset subset, paired bootstrap differences
against persistence over the same files.

Reading a run: the head sits somewhere between the floor and the ceiling. The floor is what the
published detector already gives you; the ceiling is what it would give with the true future
scan. The paired delta vs persistence with its interval is the result.

## Runs

A ladder: the published detector first, then two trained runs. One seed each, 2014 and 2015
(the authors' train and test partitions of both years; validation 15% of days), defined at the
top of `run.py` and run by `uv run run.py all` (which ends with the summary table).

| step | run | what changes | question |
|---|---|---|---|
| 0 | `detector` | nothing trained: the published detector as designed, one scan, scored on the test set | what the detector already gives on the onset question. On the newest observed scan it is the floor; on the real future scan, the ceiling |
| 1 | `frozen` | encoder frozen, both observed scans in, features concatenated, a new head trained on the +5 / +10 min labels | can the detector's own features, read over two scans, be re-aimed at "will it appear" |
| 2 | `finetuned` | encoder unfrozen at a tenth of the head's learning rate | does letting the features adapt to the target buy more |

Decided 2026-10-14: the one-scan frozen rung and the extrapolate combiner are out, so step 1
changes two things at once (the target and the second scan); the detector row shows what one
scan with the original head gives. Two years instead of one because the 2015-only runs
overfit after epoch 2-3 and the test set gains two thirds more pre-onset cases.

Every invocation of `run.py` writes one self-contained directory, `results/<stamp>/`: step
zero at `detector/` (floor and ceiling predictions), one directory per trained run (params,
history, checkpoints, head predictions and scores), `summary.csv` / `summary.txt`, and
`figures/` (the ladder of head AUCs between floor and ceiling, the paired gains with
intervals, ROC and PR curves on the pre-onset WC set, training curves). `summary.csv` also
carries per-test-year rows. Every trained run
writes only its head predictions and is paired against the cached floor on the same test
files; `run.py summary <dir>` rebuilds the summary and figures from the files alone.

With one seed per config, differences inside about ±0.03 are read as ties; the paired bootstrap
on the same test files is the interval reported.

### Final result (2026-10-15): the ladder on 2014-2015

Test partition: 6,448 samples, 495 pre-onset cases, 215 ongoing. Pre-onset cases vs warned
storms (WC AUC), 95% bootstrap intervals; paired bootstrap against the floor on the same files.
Run directory `results/final/` (git-ignored; figures copied to `figures/`).

| step | +5 min | paired vs floor | +10 min | paired vs floor | best / epochs |
|---|---|---|---|---|---|
| 0 floor (detector, newest scan) | 0.731 [0.697, 0.767] | | 0.726 [0.701, 0.750] | | |
| 0 ceiling (detector, future scan) | 0.769 [0.734, 0.806] | +0.038 [+0.016, +0.063] | 0.797 [0.772, 0.822] | +0.071 [+0.048, +0.093] | |
| 1 frozen | 0.699 [0.662, 0.737] | -0.032 [-0.045, -0.020] | 0.710 [0.683, 0.736] | -0.017 [-0.024, -0.010] | 2 / 4 |
| 2 finetuned | 0.724 [0.689, 0.762] | -0.007 [-0.031, +0.015] | 0.728 [0.701, 0.752] | +0.001 [-0.016, +0.018] | 5 / 5 (cap) |

Per year (pre-onset WC AUC, floor / finetuned / frozen): 2014 +5 0.752 / 0.725 / 0.719,
+10 0.732 / 0.732 / 0.722; 2015 +5 0.709 / 0.721 / 0.678, +10 0.722 / 0.725 / 0.700.

- **The frozen head is below the floor**, significantly: a new head on the frozen features
  over two scans does not match the published head carried forward.
- **Fine-tuning recovers the floor and no more.** +0.025 / +0.018 over frozen, a tie with the
  detector at both horizons; validation was still rising at the epoch-5 cap (train AUC 0.93 vs
  val 0.88), so a longer fine-tune might add a little, bounded by the ceiling gap of
  0.045 / 0.07.
- Reading: for the ten minutes before onset, two scans read by this CNN carry about what the
  latest scan already carries. Written up in `report.qmd` together with the coauthor's
  detection-task steps (frozen features, linear head, current scan vs two-scan mean; notebook
  in `references/`).

Earlier (2015 only, netCDF era, decoder vs head-only): head-only 0.739 / 0.735 vs the decoder
model's 0.716 / 0.729 (floor 0.709 / 0.722, ceiling 0.764 / 0.796); head-only minus decoder
+0.023 [+0.004, +0.043] at +5. Both overfit after epoch 2-3, hence early stopping.

## Open questions

- Seed variance on 2015: is 0.739 a number or a draw? Out of scope for this round (one seed per
  run); differences within about ±0.03 are read as ties.
- Overfitting on one year: lower encoder learning rate, or more years once the data comes from
  the Hub.
- More signal than two radar chips hold: environmental fields (HRRR) joined on the per-sample
  predictions, and longer scan histories.

## Conventions

- Torch only; uv, ruff, ty. No tests and no CLI since the trim of 2026-10-14: `run.py` at the
  root is the entry point, the package under `src/` holds the importable parts (detector port,
  data, model, training loop, scorer), about 1,100 lines in all.
- Results land under `results/<stamp>/` (git-ignored; a finished directory is copied in by
  hand, minus checkpoints). The shared defaults are the `Config` dataclass in `config.py`;
  the runs are a few overrides each in `run.py`.
