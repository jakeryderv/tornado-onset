"""The converted detector reproduces the published model's logits (fp32 reference)."""

import os

import numpy as np
import pandas as pd
import pytest
import torch

from tornado_onset.cli import DEFAULT_WEIGHTS, _LastScan
from tornado_onset.tornet import load_detector

REFERENCE = "results/reference/predictions_pretrained_2013_2014_fp32.csv"


def test_logits_match_reference(tornet_root: str) -> None:
    if not os.path.exists(DEFAULT_WEIGHTS):
        pytest.skip(f"{DEFAULT_WEIGHTS} missing: run onset-convert-weights")
    ref = pd.read_csv(REFERENCE).sample(8, random_state=0)
    det = load_detector(DEFAULT_WEIGHTS)
    ds = _LastScan(tornet_root, ref["filename"].tolist())
    with torch.no_grad():
        out = np.array([det(x[None], c[None]).item() for x, c in (ds[i] for i in range(len(ds)))])
    assert np.abs(out - ref["logit"].to_numpy()).max() < 1e-4
