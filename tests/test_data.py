import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from tornado_onset.data import (
    ALL_VARIABLES,
    BACKGROUND,
    N_INPUT_CHANNELS,
    OnsetDataset,
    channel_indices,
    collate,
    coordinates,
    denormalize_fields,
    normalize_fields,
    read_shard,
    row_of,
    sample_from_row,
    shard_paths,
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
    df = pd.DataFrame(
        {
            "filename": [str(i) for i in range(14)],
            "day": ["d1"] * 5 + ["d2"] * 3 + ["d3"] * 2 + ["d4"] * 4,
        }
    )
    tr, va = split_by_day(df, 0.5, seed=1)
    assert len(tr) + len(va) == 14 and not tr & va
    days = df.set_index("filename")["day"]
    assert not set(days.loc[list(tr)]) & set(days.loc[list(va)])
    assert len(set(days.loc[list(va)])) == 2


def test_hub_shard_rows_and_dataset(hub_files: dict) -> None:
    paths = shard_paths("test", [], patterns=hub_files["test"])
    assert len(paths) == 1
    shard = read_shard(paths[0])
    assert len(shard["filename"]) == 256 and shard["VEL"].shape == (256, 4, 120, 240, 2)
    assert shard["VEL"].dtype == np.float16 and shard["range_folded_mask"].dtype == np.uint8
    s = sample_from_row(row_of(shard, 0))
    assert s.fields.shape == (4, N_INPUT_CHANNELS, 120, 240)
    assert s.coords.shape == (2, 120, 240) and s.labels.shape == (4,)
    assert np.isfinite(s.fields).all() and s.filename.startswith("test/2013/")
    keep = set(shard["filename"][:6])
    ds = OnsetDataset(paths, n_in=2, horizons=2, keep=keep, shuffle=True, seed=0)
    items = list(DataLoader(ds, batch_size=4, num_workers=2, collate_fn=collate))
    assert sum(len(b["meta"]) for b in items) == 6
    b = items[0]
    assert b["x"].shape[1:] == (2, N_INPUT_CHANNELS, 120, 240)
    assert b["future"].shape[1:] == (2, N_INPUT_CHANNELS, 120, 240)
    assert b["minutes"].shape[1] == 2 and torch.all(b["minutes"] > 0) and torch.all(b["dt_in"] > 0)
    assert {m["filename"] for it in items for m in it["meta"]} == keep
