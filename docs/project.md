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
| `y2015` | the model above on 2015 | done 2026-10-14, `results/y2015/onset_2015_261014_151335` |
| `y2015_head_only` | encoder + head, no decoder, no drawing losses | done 2026-10-14, `results/y2015/onset_2015_head_only_261014_153239` |
| `smoke_2015` | `y2015` on 6,000 files, 4 epochs | utility |

### 2015 result (2026-10-14)

Test partition: 3,902 samples, 275 pre-onset cases, 82 ongoing. Pre-onset cases vs warned
storms (WC AUC), paired bootstrap against persistence in brackets.

| horizon | floor | ceiling | baseline head | head-only | baseline forecast (judge) |
|---|---|---|---|---|---|
| +5 min | 0.709 | 0.764 | 0.716 (+0.007 [−0.030, +0.040]) | 0.739 (+0.030 [−0.006, +0.064]) | 0.696 |
| +10 min | 0.722 | 0.796 | 0.729 (+0.007 [−0.015, +0.031]) | 0.735 (+0.013 [−0.011, +0.038]) | 0.711 |

Head-only minus the baseline head on the same files, pre-onset WC: +0.023 [+0.004, +0.043] at
+5 min, +0.006 [−0.010, +0.021] at +10 min. Baseline: 10.1M parameters, 181 s per epoch, best
epoch 2. Head-only: 4.9M parameters, 129 s per epoch, best epoch 3. Both overfit after epoch 2
to 3 on one year of training data (training AUC keeps rising to 0.95 while validation falls).

- **The decoder does not earn its place on 2015.** Head-only is as good at +10 min and better
  at +5 min, with less than half the parameters and 30% less time per epoch. The drawing losses
  were supposed to lift the head through the shared encoder; here they do not.
- **The head beats the floor, and only just.** Head-only is +0.030 / +0.013 over the published
  detector on the newest scan, intervals touching zero; the ceiling stays 0.055 / 0.074 above
  the floor. Fine-tuning on one year buys a small, real-looking gain that one test year cannot
  resolve.
- **The drawing route is unchanged from before:** the judge on the forecast sits just under the
  floor (−0.013 / −0.011) while the pixels beat persistence.

**Decision pending.** The evidence says drop the decoder (and segmentation_models_pytorch with
it) and build on head-only. Before that, two cheap checks are worth a look: a second seed of
each, since the two runs differ by 0.006 to 0.023 and the seed noise is unknown; and the
overfitting, which argues for fewer epochs or a lower encoder learning rate on one year.

## Open questions

- Does the drawing earn its place? On 2015, no (see the result above). Open: seed variance, and
  whether the judge can help the head some cheaper way, e.g. supervising the head's own 8×15 map
  with the judge's heat map on the real future scan.
- Overfitting on one year: fewer epochs, a lower encoder learning rate, or more years.
- More years, and a longer history than two scans.
- Environmental fields (HRRR) joined on `features.npz`.

## Conventions

- Torch only; uv, ruff, ty, pytest. `uv run pytest` includes a one-epoch run on 24 real files
  when `TORNET_ROOT` is set.
- Runs land under `$EXP_DIR` (default `experiments/`, git-ignored). Any `Config` key can be set
  in the JSON or with `--set key=value`; keys starting with `_` are notes.
- `onset-score <checkpoint>` rescores any checkpoint with the config it was trained with.
