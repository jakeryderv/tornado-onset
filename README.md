# tornado-onset

Predict tornado onset from radar: given the scans observed so far, will a tornado appear in the
next 5 and 10 minutes when none is visible yet?

The model is the published [TorNet](https://github.com/mit-ll/tornet) detector's encoder
fine-tuned on that target, run on the two most recent scans, with the authors' head on top. The
frozen published detector is the judge that sets the persistence floor (detector on the newest
real scan) and the ceiling (detector on the real future scan). Everything is PyTorch; model and
data both come from the Hugging Face Hub:

- weights: [`jakeryderv/tornet-detector-torch`](https://huggingface.co/jakeryderv/tornet-detector-torch),
  the published Keras checkpoint converted once to a torch state dict (18 MB)
- data: [`jakeryderv/tornet`](https://huggingface.co/datasets/jakeryderv/tornet), TorNet v1.1 as
  Parquet shards per partition and year (2015 is about 12 GB), streamed one shard at a time

Both are cached on first use; nothing to download by hand.

## Run

```sh
uv sync
uv run run.py all                # baseline + three experiments on 2015, then the summary table
uv run run.py baseline --smoke   # a few thousand files, one epoch: does the path run?
```

The four runs are defined at the top of `run.py`:

| run | change | question |
|---|---|---|
| `baseline` | none | fine-tuned encoder on two scans |
| `frozen_encoder` | encoder frozen, head only trains | does fine-tuning on the onset target matter |
| `one_scan` | the model sees only the newest scan | does the second scan buy anything |
| `extrapolate` | features extrapolated by the real scan timing, shared head from the published head | a fixed time rule instead of a learned one |

Each lands under `experiments/<name>_<timestamp>/`: `params.json`, `history.csv`,
`checkpoints/{best,last}.pt`, `scores/` (predictions CSVs for head, floor and ceiling, and
`summary.json`) and `results.json`. The shared defaults are the `Config` dataclass in
`src/tornado_onset/config.py`; the package holds the detector port, the data pipeline, the
model, the training loop and the scorer.

`docs/project.md` is the running log; `report.qmd` the writeup.

```sh
uv run ruff check . && uv run ruff format --check . && uv run ty check
```
