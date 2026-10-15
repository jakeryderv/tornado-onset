import json
import os

import numpy as np
import pytest
import torch

from tornado_onset.config import Config, load_config
from tornado_onset.data import N_INPUT_CHANNELS
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


def test_end_to_end_tiny_run(hub_files: dict, tmp_path) -> None:
    """Two epochs on 24 samples of one Hub shard with the real weights, scoring on 12, and the
    summary over two such runs."""
    if not os.path.exists(Config().weights):
        pytest.skip("run onset-convert-weights first")
    from tornado_onset.train import run

    base = dict(
        data_files=hub_files,
        max_files=24,
        epochs=2,
        patience=1,
        batch_size=4,
        workers=0,
        bootstrap=5,
        val_fraction=0.3,
        exp_dir=str(tmp_path / "exp"),
    )
    results = run(load_config(None, exp_name="t", **base))
    assert results["epochs_run"] in (1, 2) and os.path.exists(results["best_checkpoint"])
    scores = os.path.join(results["expdir"], "scores")
    names = {f"predictions_{w}_h{h}.csv" for w in ("head", "persistence", "truth") for h in (0, 1)}
    assert names <= set(os.listdir(scores))
    feats = np.load(os.path.join(scores, "features.npz"))
    assert feats["features_max"].shape == (256, 2 * 384) and feats["head_logits"].shape == (256, 2)
    with open(os.path.join(scores, "summary.json")) as f:
        s = json.load(f)
    assert len(s["horizons"]) == 2 and s["horizons"][0]["n"] == 256
    assert set(results["pre_onset_WC_AUC"]["h0"]) == {"persistence", "truth", "head"}
    run(load_config(None, exp_name="u", seed=7, **base))
    runs = find_runs([str(tmp_path / "exp")])
    assert len(runs) == 2
    rows = [r for run_dir in runs for r in run_rows(run_dir)]
    t = table(rows)
    assert set(t["exp_name"]) == {"t", "u"} and len(t) == 4
    c = compare(rows, "t", "u", n_boot=5)
    assert len(c) == 2 and "WC" in c[0]
