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
TorNet encoder, fine-tuned, run per scan; features concatenated across scans at every scale
    |
    +--> classification head (the authors' design: 1x1 convs to a per-cell logit map, max)
    |        -> P(tornado at +5), P(tornado at +10)            <- the product
    |
    +--> U-Net decoder -> drawn future VEL + DBZ at each horizon  <- auxiliary signal
```

The frozen published detector (**the judge**) never trains. It reads three chips at evaluation:
the newest observed scan (**persistence**, the floor: assume the future looks like the present),
the real future scan (**truth**, the ceiling: a perfect forecast), and the drawn forecast (a
health check on the auxiliary task). It also sits inside the loss.

**Losses**, one total; each term updates only the parts that produced its output:

| term | what | reaches |
|---|---|---|
| `cls` | weighted cross-entropy on the head's per-horizon logits | head, encoder |
| `field` | L1 on drawn VEL + DBZ, times (1 + `heatmap_weight` × the judge's normalised heat map on the real target scan) | decoder, encoder |
| `detector` | the judge's weighted cross-entropy on the drawn chip (drawn products placed into the last observed scan) | decoder, encoder |

Defaults: `cls_weight` 1, `field_weight` 1, `heatmap_weight` 2, `detector_weight` 0.5; the
authors' sample weights (EF2+ ×2, warnings ×0.5). With `decoder: false` the model is head-only:
encoder and head, `cls` alone.

Checkpoints are selected on the mean validation head AUC, because the head is the product.

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
| `y2015` | the model above on 2015 | running 2026-10-14 |
| `y2015_head_only` | encoder + head, no decoder, no drawing losses | queued 2026-10-14 |
| `smoke_2015` | `y2015` on 6,000 files, 4 epochs | utility |

Results go here as they land, with the run directory copied to `results/` without checkpoints.

## Open questions

- Does the drawing earn its place? `y2015` vs `y2015_head_only` answers it. If head-only ties,
  the decoder goes. If it loses, the auxiliary signal matters and cheaper conduits are worth
  trying: supervising the head's own 8×15 map with the judge's heat map on the real future scan,
  or predicting the judge's bottleneck features on the future scan from the past scans.
- More years, and a longer history than two scans.
- Environmental fields (HRRR) joined on `features.npz`.

## Conventions

- Torch only; uv, ruff, ty, pytest. `uv run pytest` includes a one-epoch run on 24 real files
  when `TORNET_ROOT` is set.
- Runs land under `$EXP_DIR` (default `experiments/`, git-ignored). Any `Config` key can be set
  in the JSON or with `--set key=value`; keys starting with `_` are notes.
- `onset-score <checkpoint>` rescores any checkpoint with the config it was trained with.
