# tornado-onset

Predict tornado onset from radar: given the scans observed so far, will a tornado appear in the
next 5 and 10 minutes when none is visible yet?

The model is the published [TorNet](https://github.com/mit-ll/tornet) backbone
([weights on Hugging Face](https://huggingface.co/tornet-ml/tornado_detector_baseline_v1))
fine-tuned on that target. A radar forecaster decoder, trained with a detector-aware loss, is the
auxiliary signal that keeps the encoder honest; the frozen published detector is the judge that
sets the persistence floor (detector on the newest real scan) and the ceiling (detector on the
real future scan). Everything is PyTorch; the published Keras checkpoint is converted once to a
state dict.

See `docs/project.md` for the design, the evaluation protocol and the run log.

## Setup

```sh
uv sync
export TORNET_ROOT=~/data/tornet   # catalog.csv plus train/<year>, test/<year>
uv run onset-convert-weights --verify results/reference/predictions_pretrained_2013_2014_fp32.csv
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run ty check
```

The first command downloads the published Keras checkpoint from Hugging Face, writes
`weights/tornet_detector_v1.pt` (git-ignored) and checks its logits against the reference.

## Runs

```sh
export TORNET_ROOT=~/data/tornet
uv run onset-train configs/smoke_2015.json                 # 6,000 files, 4 epochs
uv run onset-train configs/y2015.json                      # the 2015 slice
uv run onset-score experiments/<run>/checkpoints/last.pt   # rescore any checkpoint
```

Each run lands under `$EXP_DIR` (default `experiments/`): `params.json`, `history.csv`,
`checkpoints/{best,last}.pt`, `scores/` (per-view predictions CSVs, `summary.json`, likelihood
maps, `features.npz`) and `results.json`. Any key in `tornado_onset.config.Config` can be set in
the JSON or on the command line with `--set key=value`.

## Data

TorNet v1.1 (Zenodo). The first slice uses 2015 only: the authors' train and test partitions
within that year, with validation held out from the training partition by day.
