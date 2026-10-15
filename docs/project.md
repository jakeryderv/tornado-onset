# tornado-onset: project notes

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
original Zenodo release, CC BY 4.0). `load_split` fetches the per-year configs of
`train_years` / `test_years`, or explicit `data_files` patterns; `datasets` caches and
memory-maps the shards, and the dataset wrapper builds the detector's tensors per row. No local
netCDF, no `TORNET_ROOT`.

Partitions are the authors' `train` and `test` splits. Validation is held out from the training
partition by whole days, never by random sample, because consecutive scans of one storm are near
duplicates. The first slice is 2015: 19,721 training files (16,717 train / 3,004 validation over
180 / 32 days) and 3,902 test files.

Pre-onset positives are rare by construction, roughly 200 per test year, so every pre-onset
number carries an interval of about ±0.025 AUC. Differences under about 0.03 are not resolvable
on one year.

## Model

The published TorNet detector supplies both the starting point and the judge. Its Keras
checkpoint is converted once to a torch state dict (`onset-convert-weights`), with no Keras in
the dependency tree; the conversion reproduces the published logits to 2e-5.

```
two scans (14 channels each: 12 normalised radar channels + 2 range-folded)
    |
TorNet encoder, fine-tuned, run per scan; bottleneck features concatenated across scans
    |
the authors' head: 1x1 convs to a per-cell logit map, max over cells
    -> P(tornado at +5), P(tornado at +10)
```

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
with it is commit `82567d5`; its runs are in `results/y2015/`.

## Evaluation

Per horizon, four probabilities per test sample: head, judge on persistence, judge on forecast,
judge on truth. Each is written as a predictions CSV with the label pattern (e.g. `0011`) and
`pre_onset` / `ongoing` flags. `summary.json` holds AUC and AUCPR per view (NC, RC, WC) with
bootstrap intervals, for all cases and for the pre-onset subset, paired bootstrap differences
against persistence over the same files, and the drawing's pixel error per product against
truth and against persistence. `maps_h<k>.npz` keeps every 8×15 likelihood map; `features.npz`
keeps the head logits and the max- and mean-pooled bottleneck features per sample, so a later
model can take them as inputs.

Reading a run: the head sits somewhere between the floor and the ceiling. The floor is what the
published detector already gives you; the ceiling is what it would give with the true future
scan. The paired delta vs persistence with its interval is the result.

## Runs

| config | what | status |
|---|---|---|
| `y2015` | encoder + head on 2015, early stopping | ready; the seed is a `--set seed=N` away |
| `smoke_2015` | `y2015` on 6,000 files | utility |

`onset-summary [roots]` tabulates every finished run (head vs floor and ceiling per config and
horizon, mean ± sd over seeds); `--compare A B` gives the paired head-vs-head delta per seed.

### 2015 result (2026-10-14, before the decoder was dropped)

Test partition: 3,902 samples, 275 pre-onset cases, 82 ongoing. Pre-onset cases vs warned
storms (WC AUC), paired bootstrap against persistence in brackets. Runs in `results/y2015/`.

| horizon | floor | ceiling | decoder + judge losses, head | head-only |
|---|---|---|---|---|
| +5 min | 0.709 | 0.764 | 0.716 (+0.007 [-0.030, +0.040]) | 0.739 (+0.030 [-0.006, +0.064]) |
| +10 min | 0.722 | 0.796 | 0.729 (+0.007 [-0.015, +0.031]) | 0.735 (+0.013 [-0.011, +0.038]) |

Head-only minus the other head on the same files, pre-onset WC: +0.023 [+0.004, +0.043] at
+5 min, +0.006 [-0.010, +0.021] at +10 min. Both overfit after epoch 2 to 3 on one year of
training data (training AUC keeps rising to 0.95 while validation falls), which is what early
stopping now addresses.

- **The head beats the floor, and only just.** +0.030 / +0.013 over the published detector on
  the newest scan, intervals touching zero; the ceiling stays 0.055 / 0.074 above the floor.
  Fine-tuning on one year buys a small, real-looking gain that one test year cannot resolve.
- **One seed each.** The next runs are more seeds of `y2015`, and `encoder_lr_scale` 0.03
  against the overfitting, 15 minutes each.

## Open questions

- Seed variance on 2015: is 0.739 a number or a draw? Three seeds and `onset-summary` answer it.
- Overfitting on one year: lower encoder learning rate, or more years once the data comes from
  the Hub.
- More signal than two radar chips hold: environmental fields (HRRR) joined on `features.npz`,
  and longer scan histories.

## Conventions

- Torch only; uv, ruff, ty, pytest. `uv run pytest` includes two tiny runs on 24 samples of one
  Hub shard (skipped offline).
- Runs land under `$EXP_DIR` (default `experiments/`, git-ignored). Any `Config` key can be set
  in the JSON or with `--set key=value`; keys starting with `_` are notes.
- `onset-score <checkpoint>` rescores any checkpoint with the config it was trained with.
