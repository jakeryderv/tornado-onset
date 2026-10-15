# tornado-onset

Predict tornado onset from radar: given the scans observed so far, will a tornado appear in the
next 5 and 10 minutes when none is visible yet?

The model is the published [TorNet](https://github.com/mit-ll/tornet) detector's encoder, frozen
or fine-tuned, run on the most recent scans with a new copy of the authors' head trained on that
target. The published detector itself, as designed, is step zero: on the newest real scan it is
the persistence floor, on the real future scan the ceiling. Everything is PyTorch; model and
data both come from the Hugging Face Hub:

- weights: [`jakeryderv/tornet-detector-torch`](https://huggingface.co/jakeryderv/tornet-detector-torch),
  the published Keras checkpoint converted once to a torch state dict (18 MB)
- data: [`jakeryderv/tornet`](https://huggingface.co/datasets/jakeryderv/tornet), TorNet v1.1 as
  Parquet shards per partition and year (2015 is about 12 GB), streamed one shard at a time

Both are cached on first use; nothing to download by hand.

## Run

```sh
uv sync
uv run run.py all                      # the published detector, then the three runs, then the table
uv run run.py frozen_one_scan --smoke  # a few thousand files, one epoch: does the path run?
```

The runs form a ladder, each changing one thing, defined at the top of `run.py`:

| step | run | what changes | question |
|---|---|---|---|
| 0 | `detector` | nothing trained: the published detector as designed, one scan, scored on the 2015 test set | what the detector already gives on the onset question. Newest observed scan: the floor. Real future scan: the ceiling |
| 1 | `frozen_one_scan` | a new head trained on the onset target; same single scan, encoder frozen | the scope change alone |
| 2 | `frozen_two_scans` | the second observed scan added, features concatenated | does time help when the features are fixed |
| 3 | `finetuned` | encoder unfrozen at a tenth of the head's learning rate | does letting the features adapt buy more |

Step zero lands at `experiments/detector/` (floor and ceiling predictions, `summary.json`) and is
made once; each trained run lands under `experiments/<name>_<timestamp>/` with `params.json`,
`history.csv`, `checkpoints/{best,last}.pt`, `scores/` (head predictions, the floor and ceiling
copied in, `summary.json`) and `results.json`. The shared defaults are the `Config` dataclass in
`src/tornado_onset/config.py`; the package holds the detector port, the data pipeline, the
model, the training loop and the scorer.

`docs/project.md` is the running log; `report.qmd` the writeup.

```sh
uv run ruff check . && uv run ruff format --check . && uv run ty check
```
