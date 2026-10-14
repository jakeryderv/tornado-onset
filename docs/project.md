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

1. **Weights (done 2026-10-14).** `onset-convert-weights` reads `model.weights.h5` out of the
   published `.keras` zip with h5py and writes `weights/tornet_detector_v1.pt` (26 tensors,
   4,665,409 parameters), no Keras. `--verify` compares logits with
   `results/reference/predictions_pretrained_2013_2014_fp32.csv`, the Keras model run in fp32 on
   CPU over the 3,119-sample 2013–2014 test partition: max difference 1.2e-5. The fork's
   original reference CSV was produced on the GPU with TF32 convolutions and differs from fp32
   by up to 1e-2, which is why a fresh fp32 reference was made; the verification turns TF32 off.
2. **Thin slice on 2015 (code built 2026-10-14; runs pending).** `onset-train <config.json>`
   trains and scores; `onset-score <checkpoint>` rescores any checkpoint (score-only mode).
   Modules: `data` (catalog, reader, normalisation, coordinates, day split, dataset), `tornet`
   (the published model and its weights), `model` (encoder + U-Net decoder + head, the losses),
   `train` (loop, checkpoint on best validation head AUC), `score` (floor / forecast / ceiling /
   head, pre-onset scoping, paired bootstraps, likelihood maps, `features.npz` with head logits
   and max/mean-pooled bottleneck features per test sample), `metrics`, `config`, `cli`.
   Configs: `configs/smoke_2015.json` (6,000 files, 4 epochs) and `configs/y2015.json` (the full
   2015 training partition, 19,721 files, 6 epochs; test partition 3,902 files). The end-to-end
   test trains one epoch on 24 real files and scores 12.
3. Later, not now: more years, HRRR join on the per-sample table, longer scan sequences.

## Decisions

- Package `tornado_onset`, torch only, uv + ruff + ty + pytest.
- Loss design frozen at the fork's `fold_b_detector` config. No channel-scope, anchor or
  bottleneck options.
- Checkpoint selection on validation head AUC (the head is the product).
- Runs land under `$EXP_DIR` (default `experiments/`, git-ignored); results worth keeping are
  copied to `results/` without checkpoints.
