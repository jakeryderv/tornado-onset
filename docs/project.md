# tornado-onset: project notes

## Where things stand (2026-10-14)

Carried over from the TorNet fork (`~/dev/projects/tornet`, branch `jake/dev`, `docs/project.md`
there has the full record):

- **Target.** Pre-onset cases: tornado samples where no input scan is tornadic and the target scan
  is. Scored against warned storms (WC) with paired bootstraps. With two scans in, that is about
  220 cases per test year, so every number carries a ±0.025 interval.
- **What worked.** Fine-tuning the TorNet backbone on the prediction target, with a radar
  forecaster decoder trained by a heat-map-weighted L1 on velocity and reflectivity plus the
  frozen detector's cross-entropy on the drawn chip. On fold B (train 2013+2015, test 2014) the
  classification head scored 0.757 WC AUC at +10 min against a persistence floor of 0.732 and a
  ceiling of 0.798; at +5 min nothing beats the floor (0.753). Without the detector-aware drawing
  loss the head drops to 0.741 / 0.721, so the decoder stays as an auxiliary signal.
- **What did not.** The drawn forecast read by the frozen detector never beats copying the last
  scan (+0.004 at +10 min, n.s.), at four times the probe data. Drawing all twelve channels made it
  worse and let the detector term bend KDP away from reality. The indirect route is retired as a
  predictor; its MAE and judge score remain as a health check.
- **Biggest external signal so far.** HRRR environment fields added +0.054 WC AUC at lead 10 in
  the fork's Experiment 7 meta model. HRRR coverage starts 2014-07-30, so that combination only
  pays from 2015 on. Not part of this slice.

## Plan

1. **Weights.** Convert the published `.keras` checkpoint (a zip holding an HDF5 weights file) to
   a torch state dict with h5py, no Keras. Verify against the fork's saved reference logits
   (`results/predictions_pretrained_2013_2014.csv`).
2. **Thin slice on 2015.** Port the data pipeline (catalog, reader, normalisation, CoordConv
   coordinates, two scans in / two targets out), the torch backbone, the U-Net decoder
   (segmentation_models_pytorch) and the authors' head; the training loop with the frozen loss
   design; the scorer (floor / forecast / ceiling / head, pre-onset scoping, paired bootstraps,
   per-sample table of head logits and pooled features). Validation held out from the 2015
   training partition by day. Smoke on ~6,000 files, then the full 2015 partition (~25 min
   locally).
3. Later, not now: more years, HRRR join on the per-sample table, longer scan sequences.

## Decisions

- Package `tornado_onset`, torch only, uv + ruff + ty + pytest.
- Loss design frozen at the fork's `fold_b_detector` config. No channel-scope, anchor or
  bottleneck options.
- Checkpoint selection on validation head AUC (the head is the product).
- Runs land under `$EXP_DIR` (default `experiments/`, git-ignored); results worth keeping are
  copied to `results/` without checkpoints.
