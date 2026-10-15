# tornado-onset

Predict tornado onset from radar: given the scans observed so far, will a tornado appear in the
next 5 and 10 minutes when none is visible yet?

The model is the published [TorNet](https://github.com/mit-ll/tornet) backbone
([weights on Hugging Face](https://huggingface.co/tornet-ml/tornado_detector_baseline_v1))
fine-tuned on that target, with the authors' head on top. The frozen published detector is the
judge that sets the persistence floor (detector on the newest real scan) and the ceiling
(detector on the real future scan). Everything is PyTorch; the published Keras checkpoint is
converted once to a state dict.

See `docs/project.md` for the design, the evaluation protocol and the run log.

## Setup

```sh
uv sync
uv run onset-convert-weights --verify results/reference/predictions_pretrained_2013_2014_fp32.csv
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run ty check
```

The second command downloads the published Keras checkpoint from Hugging Face, writes
`weights/tornet_detector_v1.pt` (git-ignored) and checks its logits against the reference,
reading the 2013 and 2014 test samples from the Hub dataset. The tests fetch one shard of each
2013 partition (about 225 MB, cached).

## Runs

```sh
uv run onset-train configs/smoke_2015.json                 # 6,000 samples
uv run onset-train configs/y2015.json                      # the 2015 slice
uv run onset-train configs/y2015.json --set seed=2          # another seed
uv run onset-score experiments/<run>/checkpoints/last.pt   # rescore any checkpoint
uv run onset-summary                                       # every finished run, one table
```

Each run lands under `$EXP_DIR` (default `experiments/`): `params.json`, `history.csv`,
`checkpoints/{best,last}.pt`, `scores/` (per-view predictions CSVs, `summary.json`, likelihood
maps, `features.npz`) and `results.json`. Any key in `tornado_onset.config.Config` can be set in
the JSON or on the command line with `--set key=value`.

## Data

TorNet v1.1 from the Hub dataset [`jakeryderv/tornet`](https://huggingface.co/datasets/jakeryderv/tornet):
one config per year with the authors' `train` and `test` splits, cached and memory-mapped by
`datasets` on first use (2015 is about 12 GB). `train_years` / `test_years` pick years;
`data_files` in a config takes explicit shard patterns instead, for a fold or a subset.
Validation is held out from the training partition by day. The first slice uses 2015 only.
