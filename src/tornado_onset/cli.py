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

DEFAULT_WEIGHTS = "weights/tornet_detector_v1.pt"


class _LastScan(Dataset):
    """The newest scan of each file, as the published detector was scored."""

    def __init__(self, root: str, files: list[str]):
        self.root, self.files = root, files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        from tornado_onset.data import read_sample

        s = read_sample(os.path.join(self.root, self.files[i]))
        return torch.from_numpy(s.fields[-1]), torch.from_numpy(s.coords)


def verify_reference(
    weights: str, reference_csv: str, root: str, batch_size: int, workers: int
) -> float:
    """Run the converted detector over the files in a reference predictions CSV and return the
    largest absolute logit difference."""
    from sklearn.metrics import roc_auc_score

    from tornado_onset.tornet import load_detector

    ref = pd.read_csv(reference_csv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_detector(weights, device)
    torch.backends.cudnn.allow_tf32 = False  # compare in true fp32
    torch.backends.cuda.matmul.allow_tf32 = False
    loader = DataLoader(
        _LastScan(root, ref["filename"].tolist()), batch_size=batch_size, num_workers=workers
    )
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
        roc_auc_score(ref["label"], out),
        roc_auc_score(ref["label"], ref["logit"]),
    )
    return float(diff.max())


def convert_weights(argv: list[str] | None = None) -> None:
    from tornado_onset.data import data_root
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
    p.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="max |logit difference| allowed against the fp32 reference",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sd = convert(a.keras, a.out)
    logging.info(
        "wrote %s: %d tensors, %d parameters", a.out, len(sd), sum(v.numel() for v in sd.values())
    )
    if a.verify:
        worst = verify_reference(a.out, a.verify, data_root(), a.batch_size, a.workers)
        if worst > a.tolerance:
            logging.error("max logit difference %.3e exceeds %.1e", worst, a.tolerance)
            sys.exit(1)
        logging.info("verified within %.1e", a.tolerance)


def train(argv: list[str] | None = None) -> None:
    raise SystemExit("not built yet: see docs/project.md, step 2")


def score(argv: list[str] | None = None) -> None:
    raise SystemExit("not built yet: see docs/project.md, step 2")
