"""The converted detector reproduces the published model's logits (fp32 reference)."""

import numpy as np
import pandas as pd
import torch

from tornado_onset.cli import _LastScan
from tornado_onset.config import resolve_weights
from tornado_onset.tornet import load_detector

REFERENCE = "results/reference/predictions_pretrained_2013_2014_fp32.csv"


def test_logits_match_reference(hub_files: dict) -> None:
    """Eight 2013 reference samples, read from the Hub's 2013 test split (0.26 GB, cached)."""
    from tornado_onset.data import load_split

    ref = pd.read_csv(REFERENCE)
    ref = ref[ref["year"] == 2013].sample(8, random_state=0)
    ds = load_split("test", [2013])
    where = {f: i for i, f in enumerate(ds["filename"])}
    det = load_detector(resolve_weights())
    scan = _LastScan(ds, [where[f] for f in ref["filename"]])
    with torch.no_grad():
        out = np.array(
            [det(x[None], c[None]).item() for x, c in (scan[i] for i in range(len(scan)))]
        )
    assert np.abs(out - ref["logit"].to_numpy()).max() < 1e-4
