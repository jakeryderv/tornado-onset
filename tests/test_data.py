import numpy as np
import torch
from datasets import Dataset

from tornado_onset.data import (
    ALL_VARIABLES,
    BACKGROUND,
    N_INPUT_CHANNELS,
    OnsetDataset,
    channel_indices,
    collate,
    coordinates,
    days_of,
    denormalize_fields,
    load_split,
    normalize_fields,
    sample_from_row,
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
    day = 86400
    times = [[d * day + 60 * i] * 4 for d in (1, 1, 1, 2, 2, 3, 4, 4, 4, 4) for i in (0,)]
    ds = Dataset.from_dict({"time": times, "filename": [str(i) for i in range(len(times))]})
    idx = np.arange(len(ds))
    tr, va = split_by_day(ds, idx, 0.5, seed=1)
    assert len(tr) + len(va) == len(ds)
    assert not set(days_of(ds, tr)) & set(days_of(ds, va))
    assert len(set(days_of(ds, va))) == 2


def test_hub_rows_and_dataset(hub_files: dict) -> None:
    ds = load_split("test", [], data_files=hub_files)
    assert len(ds) == 256 and ds.features["VEL"].shape == (4, 120, 240, 2)
    s = sample_from_row(ds.with_format("numpy")[0])
    assert s.fields.shape == (4, N_INPUT_CHANNELS, 120, 240)
    assert s.coords.shape == (2, 120, 240) and s.labels.shape == (4,)
    assert np.isfinite(s.fields).all() and s.filename.startswith("test/2013/")
    od = OnsetDataset(ds, n_in=2, horizons=2, indices=[0, 1])
    b = collate([od[0], od[1]])
    assert b["x"].shape == (2, 2, N_INPUT_CHANNELS, 120, 240)
    assert b["future"].shape == (2, 2, N_INPUT_CHANNELS, 120, 240)
    assert b["minutes"].shape == (2, 2) and torch.all(b["minutes"] > 0)
    assert torch.all(b["dt_in"] > 0)
    assert b["meta"][0]["filename"] == ds[0]["filename"]
