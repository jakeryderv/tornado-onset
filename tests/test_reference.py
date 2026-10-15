"""The converted detector reproduces the published model's logits (fp32 reference)."""

import numpy as np
import pandas as pd
import torch

from tornado_onset.cli import last_scans
from tornado_onset.config import resolve_weights
from tornado_onset.tornet import load_detector

REFERENCE = "tests/data/predictions_pretrained_2013_2014_fp32.csv"


def test_logits_match_reference(hub_files: dict) -> None:
    """Eight 2013 reference samples, read from the Hub's 2013 test shards (0.26 GB, cached)."""
    from tornado_onset.data import shard_paths

    ref = pd.read_csv(REFERENCE)
    ref = ref[ref["year"] == 2013].sample(8, random_state=0)
    paths = shard_paths("test", [2013])
    det = load_detector(resolve_weights())
    with torch.no_grad():
        out = np.array(
            [det(x[None], c[None]).item() for x, c in last_scans(paths, ref["filename"].tolist())]
        )
    assert np.abs(out - ref["logit"].to_numpy()).max() < 1e-4
