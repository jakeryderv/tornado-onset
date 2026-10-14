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

This repo continues the work in the `jake/dev` branch of the TorNet fork at
`~/dev/projects/tornet` (Experiments 1–7 and the forecaster fold B runs live there, with their
results and notes). See `docs/project.md` for where things stand and what is being built.

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

## Data

TorNet v1.1 (Zenodo). The first slice uses 2015 only: the authors' train and test partitions
within that year, with validation held out from the training partition by day.
