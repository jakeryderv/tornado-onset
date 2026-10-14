import json
import os

import numpy as np
import pandas as pd
import pytest
import torch

from tornado_onset.config import Config, load_config
from tornado_onset.data import N_INPUT_CHANNELS, OnsetDataset, collate, load_catalog, select_files
from tornado_onset.model import Losses, OnsetModel
from tornado_onset.summary import compare, find_runs, run_rows, table

AZ, RNG = 32, 48  # small chips keep the tests quick; the pyramid logic is size-agnostic


def tiny_batch(b: int = 2, n_in: int = 2, horizons: int = 2, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {
        "x": torch.randn(b, n_in, N_INPUT_CHANNELS, AZ, RNG, generator=g),
        "coords": torch.rand(b, 2, AZ, RNG, generator=g),
        "future": torch.randn(b, horizons, N_INPUT_CHANNELS, AZ, RNG, generator=g),
        "labels": torch.tensor([[0.0, 1.0], [1.0, 1.0]])[:b],
        "frame_labels": torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]])[:b],
        "minutes": torch.tensor([[5.0, 10.0]] * b),
        "weight": torch.ones(b),
        "meta": [{"filename": f"f{i}"} for i in range(b)],
    }


def test_forward_shapes() -> None:
    m = OnsetModel(n_in=2, horizons=2, start_filters=8).eval()
    b = tiny_batch()
    out = m(b["x"], b["coords"])
    assert out["cls_map"].shape == (2, 2, -(-AZ // 16), -(-RNG // 16))
    assert out["logits"].shape == (2, 2)
    assert out["features"].shape[:2] == (2, 2 * 8 * 8)
    assert torch.equal(out["logits"], out["cls_map"].amax(dim=(2, 3)))


def test_loss_reaches_encoder_and_head() -> None:
    m = OnsetModel(n_in=2, horizons=2, start_filters=8)
    b = tiny_batch()
    total, parts = Losses(Config(), torch.device("cpu"))(m(b["x"], b["coords"]), b)
    assert set(parts) == {"cls"} and torch.isfinite(total)
    total.backward()
    assert all(p.grad is not None for p in m.encoder.parameters())
    assert all(p.grad is not None for p in m.head.parameters())


def test_config_rejects_unknown_keys(tmp_path) -> None:
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"epochs": 1, "bogus": 2}))
    with pytest.raises(ValueError, match="bogus"):
        load_config(p)
    p.write_text(json.dumps({"epochs": 1, "_note": "ignored"}))
    assert load_config(p).epochs == 1
    cfg = load_config(None, epochs=1, sample_weights={"w2": 3.0})
    assert cfg.sample_weights["w2"] == 3.0 and cfg.sample_weights["wW"] == 0.5


def _tiny_root(tornet_root: str, tmp_path) -> tuple[str, pd.DataFrame]:
    """A data root of 24 training and 12 test files from 2015, symlinked from the real one."""
    cat = load_catalog(tornet_root)
    test_rows = select_files(cat, "test", [2015]).head(12)
    train_rows = select_files(cat, "train", [2015]).head(24)
    small = tmp_path / "data"
    small.mkdir()
    rows = pd.concat([train_rows, test_rows])
    rows.drop(columns=["year", "day"]).to_csv(small / "catalog.csv", index=False)
    for sub in ("train/2015", "test/2015"):
        (small / sub).mkdir(parents=True)
    for f in rows["filename"]:
        os.symlink(os.path.join(tornet_root, f), small / f)
    return str(small), test_rows


def test_end_to_end_tiny_run(tornet_root: str, tmp_path) -> None:
    """Two epochs on a handful of 2015 files with the real weights, scoring, and the summary."""
    if not os.path.exists(Config().weights):
        pytest.skip("run onset-convert-weights first")
    from tornado_onset.train import run

    small, test_rows = _tiny_root(tornet_root, tmp_path)
    base = dict(
        max_files=24, epochs=2, patience=1, batch_size=4, workers=0, bootstrap=5, val_fraction=0.3
    )
    results = run(load_config(None, exp_dir=str(tmp_path / "exp"), exp_name="t", **base), small)
    assert results["epochs_run"] in (1, 2) and os.path.exists(results["best_checkpoint"])
    scores = os.path.join(results["expdir"], "scores")
    assert {
        f"predictions_{w}_h{h}.csv" for w in ("head", "persistence", "truth") for h in (0, 1)
    } <= set(os.listdir(scores))
    feats = np.load(os.path.join(scores, "features.npz"))
    assert feats["features_max"].shape == (12, 2 * 384) and feats["head_logits"].shape == (12, 2)
    with open(os.path.join(scores, "summary.json")) as f:
        s = json.load(f)
    assert len(s["horizons"]) == 2 and "mae_forecast" not in s["horizons"][0]
    assert set(results["pre_onset_WC_AUC"]["h0"]) == {"persistence", "truth", "head"}
    # a second config with another seed, then the summary over both
    run(load_config(None, exp_dir=str(tmp_path / "exp"), exp_name="u", seed=7, **base), small)
    runs = find_runs([str(tmp_path / "exp")])
    assert len(runs) == 2
    rows = [r for run_dir in runs for r in run_rows(run_dir)]
    t = table(rows)
    assert set(t["exp_name"]) == {"t", "u"} and len(t) == 4
    c = compare(rows, "t", "u", n_boot=5)
    assert len(c) == 2 and "WC" in c[0]
    ds = OnsetDataset(small, test_rows["filename"].tolist()[:2])
    assert collate([ds[0], ds[1]])["x"].shape[1] == 2
