import os

import numpy as np
import pandas as pd
import torch

from tornado_onset.data import (
    ALL_VARIABLES,
    BACKGROUND,
    N_INPUT_CHANNELS,
    OnsetDataset,
    channel_indices,
    collate,
    coordinates,
    denormalize_fields,
    load_catalog,
    normalize_fields,
    read_sample,
    select_files,
    split_by_day,
)


def test_coordinates_shape_and_range() -> None:
    c = coordinates((218.0, 278.0), (136680.0, 196680.0), (120, 240))
    assert c.shape == (2, 120, 240)
    r, rinv = c
    assert np.all(np.diff(r, axis=1) > 0)  # range grows along the range axis
    assert np.allclose(r[0], r[-1])  # and is the same for every azimuth
    assert np.allclose(rinv, 1 / r)


def test_normalise_round_trip() -> None:
    rng = np.random.default_rng(0)
    frames = {v: rng.uniform(-1, 1, (4, 6, 5, 2)).astype(np.float32) for v in ALL_VARIABLES}
    frames["VEL"][0, 0, 0, 0] = np.nan
    x = normalize_fields(frames)
    assert x.shape == (4, 6, 5, 12)
    assert x[0, 0, 0, channel_indices(["VEL"])[0]] == BACKGROUND
    back = denormalize_fields(x)
    for v in ALL_VARIABLES:
        mask = ~np.isnan(frames[v])
        assert np.allclose(back[v][mask], frames[v][mask], atol=1e-5)


def test_split_by_day_keeps_days_whole() -> None:
    rows = pd.DataFrame(
        {"day": ["d1"] * 5 + ["d2"] * 3 + ["d3"] * 2 + ["d4"] * 4, "filename": range(14)}
    )
    tr, va = split_by_day(rows, 0.5, seed=1)
    assert len(tr) + len(va) == 14
    assert not set(tr["day"]) & set(va["day"])
    assert len(set(va["day"])) == 2


def test_read_sample_and_dataset(tornet_root: str) -> None:
    cat = load_catalog(tornet_root)
    rows = select_files(cat, "test", [2015])
    assert len(rows) > 0
    f = rows["filename"].iloc[0]
    s = read_sample(os.path.join(tornet_root, f))
    assert s.fields.shape == (4, N_INPUT_CHANNELS, 120, 240)
    assert s.coords.shape == (2, 120, 240)
    assert s.labels.shape == (4,)
    assert np.isfinite(s.fields).all()
    ds = OnsetDataset(tornet_root, [f, f], n_in=2, horizons=2)
    b = collate([ds[0], ds[1]])
    assert b["x"].shape == (2, 2, N_INPUT_CHANNELS, 120, 240)
    assert b["future"].shape == (2, 2, N_INPUT_CHANNELS, 120, 240)
    assert b["minutes"].shape == (2, 2)
    assert torch.all(b["minutes"] > 0)
    assert b["meta"][0]["filename"] == f
