# tornado-onset

Predict tornado onset from radar: given the scans observed so far, will a tornado appear in the
next 5 and 10 minutes when none is visible yet?

The model is the published [TorNet](https://github.com/mit-ll/tornet) backbone
([weights on Hugging Face](https://huggingface.co/tornet-ml/tornado_detector_baseline_v1))
fine-tuned on that target, with the authors' head on top. The frozen published detector is the
judge that sets the persistence floor (detector on the newest real scan) and the ceiling
(detector on the real future scan). Everything is PyTorch, and model and data both come from the
Hugging Face Hub.

See `docs/project.md` for the design, the evaluation protocol and the run log.

## Setup

```sh
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run ty check
```

Nothing to download by hand: the detector weights come from
[`jakeryderv/tornet-detector-torch`](https://huggingface.co/jakeryderv/tornet-detector-torch)
(the published Keras checkpoint converted once to a torch state dict, 18 MB) and the data from
the Hub dataset below, both cached on first use. The tests fetch one shard of each 2013
partition, about 225 MB.

To redo the weight conversion yourself and check it against the Keras model's fp32 logits:

```sh
uv run onset-convert-weights --verify tests/data/predictions_pretrained_2013_2014_fp32.csv
```

A local `weights/tornet_detector_v1.pt` takes precedence over the Hub copy.

## Runs

```sh
uv run onset-train baseline                                # the model on 2015
uv run onset-train baseline --set max_files=6000 --set epochs=3   # a quick check
uv run onset-train all                                     # baseline + the three experiments, then the table
uv run onset-score experiments/<run>/checkpoints/last.pt   # rescore any checkpoint
uv run onset-summary                                       # every finished run, one table
```

The experiments are `baseline`, `frozen_encoder`, `one_scan` and `extrapolate`, defined in
`tornado_onset.config.EXPERIMENTS`; any `Config` key can be changed with `--set key=value`.
Each run lands under `$EXP_DIR` (default `experiments/`): `params.json`, `history.csv`,
`checkpoints/{best,last}.pt`, `scores/` (per-view predictions CSVs, `summary.json`, likelihood
maps, `features.npz`) and `results.json`.

## Data

TorNet v1.1 from the Hub dataset [`jakeryderv/tornet`](https://huggingface.co/datasets/jakeryderv/tornet):
Parquet shards per partition and year, fetched into the Hub file cache on first use (2015 is
about 12 GB) and streamed one shard at a time with pyarrow, so no uncompressed Arrow cache is
ever built. `train_years` / `test_years` pick years; `data_files` in a config takes explicit
shard patterns instead, for a fold or a subset. Validation is held out from the training
partition by day. The first slice uses 2015 only.
