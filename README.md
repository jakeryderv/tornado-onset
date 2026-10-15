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
  Parquet shards per partition and year (2014 and 2015, about 23 GB), streamed one shard at a time

Both are cached on first use; nothing to download by hand.

## Run

```sh
uv sync
uv run run.py all                  # the published detector, then the two runs, into results/<stamp>/
uv run run.py summary results/X    # rebuild the summary and figures of a finished directory
```

The runs form a ladder, each changing one thing, defined at the top of `run.py`:

| step | run | what changes | question |
|---|---|---|---|
| 0 | `detector` | nothing trained: the published detector as designed, one scan, scored on the test set | what the detector gives today. Newest observed scan: the floor. Real future scan: the ceiling |
| 1 | `frozen` | encoder frozen, both observed scans in, features concatenated, a new head trained on the +5 / +10 min labels | can the detector's own features be re-aimed at onset |
| 2 | `finetuned` | encoder unfrozen at a tenth of the head's learning rate | does adapting the features to the target buy more |

Every invocation writes one self-contained directory, `results/<stamp>/`:

```
detector/            floor and ceiling predictions of step zero, summary.json
<run>/               params.json, history.csv, checkpoints/, scores/ (head predictions,
                     floor and ceiling copied in, summary.json), results.json
summary.csv/.txt     one row per run and horizon, then per test year
figures/             ladder.png, delta.png, roc_pr.png, training.png
```

Set `ONSET_WORKERS` to raise the loader workers from the laptop default of 4 on a bigger
machine. The shared defaults are the `Config` dataclass in
`src/tornado_onset/config.py`; the package holds the detector port, the data pipeline, the
model, the training loop and the scorer.

`docs/project.md` is the running log; `report.qmd` the writeup.

```sh
uv run ruff check . && uv run ruff format --check . && uv run ty check
```
