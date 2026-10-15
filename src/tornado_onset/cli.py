"""Console entry points."""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch

from tornado_onset.config import DEFAULT_WEIGHTS, EXPERIMENTS, config_from_dict, experiment


def _log() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")


def last_scans(paths: list[str], filenames: list[str]):
    """Yield (x, coords) of the newest scan for the given filenames, in that order, reading
    each shard once."""
    from tornado_onset.data import read_shard, row_of, sample_from_row

    wanted = set(filenames)
    found: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for path in paths:
        shard = read_shard(path)
        for i, f in enumerate(shard["filename"]):
            if f in wanted:
                s = sample_from_row(row_of(shard, i))
                found[f] = (torch.from_numpy(s.fields[-1]), torch.from_numpy(s.coords))
    for f in filenames:
        yield found[f]


def verify_reference(
    weights: str, reference_csv: str, batch_size: int, years: list[int] | None = None
) -> float:
    """Run the converted detector over the samples in a reference predictions CSV (read from
    the Hub's test shards) and return the largest absolute logit difference."""
    from tornado_onset.data import shard_paths
    from tornado_onset.metrics import roc_auc
    from tornado_onset.tornet import load_detector

    ref = pd.read_csv(reference_csv)
    if years:
        ref = ref[ref["year"].isin(years)].reset_index(drop=True)
    paths = shard_paths("test", sorted(ref["year"].unique().tolist()))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_detector(weights, device)
    torch.backends.cudnn.allow_tf32 = False  # compare in true fp32
    torch.backends.cuda.matmul.allow_tf32 = False
    logits, xs, cs = [], [], []

    def flush() -> None:
        if xs:
            with torch.no_grad():
                out = model(torch.stack(xs).to(device), torch.stack(cs).to(device))
            logits.append(out.squeeze(1).float().cpu().numpy())
            xs.clear()
            cs.clear()

    for x, c in last_scans(paths, ref["filename"].tolist()):
        xs.append(x)
        cs.append(c)
        if len(xs) == batch_size:
            flush()
    flush()
    out = np.concatenate(logits)
    diff = np.abs(out - ref["logit"].to_numpy())
    logging.info(
        "%d samples: max |dlogit| %.2e, mean %.2e; AUC torch %.4f, reference %.4f",
        len(out),
        diff.max(),
        diff.mean(),
        roc_auc(ref["label"].to_numpy(bool), out),
        roc_auc(ref["label"].to_numpy(bool), ref["logit"].to_numpy()),
    )
    return float(diff.max())


def convert_weights(argv: list[str] | None = None) -> None:
    from tornado_onset.tornet import convert

    p = argparse.ArgumentParser(
        description="Convert the published TorNet .keras checkpoint to a torch state dict (no Keras)."
    )
    p.add_argument("--keras", help="path to a .keras file; default downloads the published one")
    p.add_argument("--out", default=DEFAULT_WEIGHTS)
    p.add_argument(
        "--verify",
        metavar="CSV",
        help="reference predictions CSV to compare logits against; "
        "tests/data/predictions_pretrained_2013_2014_fp32.csv is the Keras model in fp32",
    )
    p.add_argument("--tolerance", type=float, default=1e-4, help="max |logit difference| allowed")
    p.add_argument("--years", nargs="*", type=int, help="restrict the verification to these years")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args(argv)
    _log()
    sd = convert(a.keras, a.out)
    logging.info(
        "wrote %s: %d tensors, %d parameters", a.out, len(sd), sum(v.numel() for v in sd.values())
    )
    if a.verify:
        worst = verify_reference(a.out, a.verify, a.batch_size, a.years)
        if worst > a.tolerance:
            logging.error("max logit difference %.3e exceeds %.1e", worst, a.tolerance)
            sys.exit(1)
        logging.info("verified within %.1e", a.tolerance)


def train(argv: list[str] | None = None) -> None:
    from tornado_onset.summary import main as summary
    from tornado_onset.train import run

    names = [*EXPERIMENTS, "all"]
    p = argparse.ArgumentParser(
        description="Train and score one of the experiments, or all four back to back."
    )
    p.add_argument("experiment", choices=names, help="; ".join(names))
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=JSON",
        help="override a Config key, e.g. --set max_files=6000 --set epochs=3",
    )
    a = p.parse_args(argv)
    _log()
    chosen = list(EXPERIMENTS) if a.experiment == "all" else [a.experiment]
    for name in chosen:
        run(experiment(name, **_overrides(a.set)))
    if a.experiment == "all":
        summary([os.environ.get("EXP_DIR", "experiments")])


def score(argv: list[str] | None = None) -> None:
    """Score an existing checkpoint with the config it was trained with (score-only mode)."""
    from tornado_onset.model import build_model
    from tornado_onset.score import evaluate

    p = argparse.ArgumentParser(
        description="Score a checkpoint with the config it was trained with (params.json)."
    )
    p.add_argument("checkpoint", help="a state dict saved by onset-train (best.pt or last.pt)")
    p.add_argument("--out", help="output directory; default: <run>/scores_<checkpoint name>")
    p.add_argument("--set", action="append", default=[], metavar="KEY=JSON")
    a = p.parse_args(argv)
    _log()
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(a.checkpoint)))
    import json

    with open(os.path.join(run_dir, "params.json")) as f:
        cfg = config_from_dict({**json.load(f)["config"], **_overrides(a.set)})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(a.checkpoint, map_location=device))
    stem = os.path.splitext(os.path.basename(a.checkpoint))[0]
    out = a.out or os.path.join(run_dir, f"scores_{stem}")
    evaluate(cfg, model, device, out)
    logging.info("wrote %s", out)


def _overrides(items: list[str]) -> dict:
    import json

    out = {}
    for item in items:
        key, _, value = item.partition("=")
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out
