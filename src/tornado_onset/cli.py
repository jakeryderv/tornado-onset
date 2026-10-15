"""Console entry points."""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from tornado_onset.config import DEFAULT_WEIGHTS, load_config


def _log() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")


class _LastScan(Dataset):
    """The newest scan of each sample, as the published detector was scored."""

    def __init__(self, ds, indices: list[int]):
        self.ds, self.indices = ds.with_format("numpy"), indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        from tornado_onset.data import sample_from_row

        s = sample_from_row(self.ds[self.indices[i]])
        return torch.from_numpy(s.fields[-1]), torch.from_numpy(s.coords)


def verify_reference(
    weights: str, reference_csv: str, batch_size: int, workers: int, years: list[int] | None = None
) -> float:
    """Run the converted detector over the samples in a reference predictions CSV (read from
    the Hub's test splits) and return the largest absolute logit difference."""
    from tornado_onset.data import load_split
    from tornado_onset.metrics import roc_auc
    from tornado_onset.tornet import load_detector

    ref = pd.read_csv(reference_csv)
    if years:
        ref = ref[ref["year"].isin(years)].reset_index(drop=True)
    ds = load_split("test", sorted(ref["year"].unique().tolist()))
    where = {f: i for i, f in enumerate(ds["filename"])}
    indices = [where[f] for f in ref["filename"]]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_detector(weights, device)
    torch.backends.cudnn.allow_tf32 = False  # compare in true fp32
    torch.backends.cuda.matmul.allow_tf32 = False
    loader = DataLoader(_LastScan(ds, indices), batch_size=batch_size, num_workers=workers)
    logits = []
    with torch.no_grad():
        for x, c in loader:
            logits.append(model(x.to(device), c.to(device)).squeeze(1).float().cpu().numpy())
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
        "results/reference/predictions_pretrained_2013_2014_fp32.csv is the Keras model in fp32",
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
        worst = verify_reference(a.out, a.verify, a.batch_size, a.workers, a.years)
        if worst > a.tolerance:
            logging.error("max logit difference %.3e exceeds %.1e", worst, a.tolerance)
            sys.exit(1)
        logging.info("verified within %.1e", a.tolerance)


def train(argv: list[str] | None = None) -> None:
    from tornado_onset.train import run

    p = argparse.ArgumentParser(
        description="Train the onset model from a JSON config and score the best checkpoint."
    )
    p.add_argument(
        "config", nargs="?", help="JSON file overriding the defaults in tornado_onset.config"
    )
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=JSON",
        help="override one key, e.g. epochs=1",
    )
    a = p.parse_args(argv)
    _log()
    run(load_config(a.config, **_overrides(a.set)))


def score(argv: list[str] | None = None) -> None:
    """Score an existing checkpoint with the config it was trained with (score-only mode)."""
    from tornado_onset.model import build_model
    from tornado_onset.score import evaluate

    p = argparse.ArgumentParser(description="Score a checkpoint on the config's test years.")
    p.add_argument("checkpoint", help="a state dict saved by onset-train (best.pt or last.pt)")
    p.add_argument("--config", help="JSON config; default: params.json next to the checkpoint")
    p.add_argument("--out", help="output directory; default: <run>/scores_<checkpoint name>")
    p.add_argument("--set", action="append", default=[], metavar="KEY=JSON")
    a = p.parse_args(argv)
    _log()
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(a.checkpoint)))
    if a.config:
        cfg = load_config(a.config, **_overrides(a.set))
    else:
        import json

        with open(os.path.join(run_dir, "params.json")) as f:
            cfg = load_config(None, **{**json.load(f)["config"], **_overrides(a.set)})
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
