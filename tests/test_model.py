import json
import os

import numpy as np
import pytest
import torch

from tornado_onset.config import EXPERIMENTS, Config, config_from_dict, experiment
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
        "dt_in": torch.tensor([5.0] * b),
        "weight": torch.ones(b),
        "meta": [{"filename": f"f{i}"} for i in range(b)],
    }


@pytest.mark.parametrize("combiner", ["concat", "extrapolate"])
def test_forward_shapes(combiner: str) -> None:
    m = OnsetModel(n_in=2, horizons=2, start_filters=8, combiner=combiner).eval()
    b = tiny_batch()
    out = m(b["x"], b["coords"], b["minutes"], b["dt_in"])
    assert out["cls_map"].shape == (2, 2, -(-AZ // 16), -(-RNG // 16))
    assert out["logits"].shape == (2, 2)
    assert out["features"].shape[:2] == (2, 2 * 8 * 8)
    assert torch.equal(out["logits"], out["cls_map"].amax(dim=(2, 3)))
    if combiner == "extrapolate":
        assert m.head.heatmap.out_channels == 1  # one shared head, the published shape


def test_extrapolate_is_linear_in_lead() -> None:
    m = OnsetModel(n_in=2, horizons=2, start_filters=8, combiner="extrapolate").eval()
    b = tiny_batch()
    per_scan = m.scan_features(b["x"], b["coords"])
    f = m.extrapolate(per_scan, b["minutes"], b["dt_in"])
    new, old = per_scan[:, -1], per_scan[:, -2]
    assert torch.allclose(f[:, 0], new + (new - old) * (5.0 / 5.0), atol=1e-6)
    assert torch.allclose(f[:, 1], new + (new - old) * (10.0 / 5.0), atol=1e-6)


def test_single_scan_model() -> None:
    m = OnsetModel(n_in=1, horizons=2, start_filters=8).eval()
    b = tiny_batch()
    out = m(b["x"][:, -1:], b["coords"], b["minutes"], b["dt_in"])
    assert out["logits"].shape == (2, 2) and out["features"].shape[1] == 8 * 8


@pytest.mark.parametrize("combiner", ["concat", "extrapolate"])
def test_loss_reaches_encoder_and_head(combiner: str) -> None:
    m = OnsetModel(n_in=2, horizons=2, start_filters=8, combiner=combiner)
    b = tiny_batch()
    out = m(b["x"], b["coords"], b["minutes"], b["dt_in"])
    total, parts = Losses(Config(combiner=combiner), torch.device("cpu"))(out, b)
    assert set(parts) == {"cls"} and torch.isfinite(total)
    total.backward()
    assert all(p.grad is not None for p in m.encoder.parameters())
    assert all(p.grad is not None for p in m.head.parameters())


def test_config_rules() -> None:
    with pytest.raises(ValueError, match="combiner"):
        Config(combiner="lstm")
    with pytest.raises(ValueError, match="extrapolate"):
        Config(combiner="concat", head_init="pretrained")
    with pytest.raises(ValueError, match="two input scans"):
        Config(combiner="extrapolate", input_scans=1)
    with pytest.raises(ValueError, match="input_scans"):
        Config(input_scans=3)
    Config(combiner="extrapolate", head_init="pretrained")
    Config(input_scans=1)
    with pytest.raises(ValueError, match="bogus"):
        config_from_dict({"epochs": 1, "bogus": 2})
    assert config_from_dict({"epochs": 1, "_note": "ignored"}).epochs == 1
    cfg = config_from_dict({"epochs": 1, "sample_weights": {"w2": 3.0}})
    assert cfg.sample_weights["w2"] == 3.0 and cfg.sample_weights["wW"] == 0.5
    for name in EXPERIMENTS:  # every experiment builds a valid config named after itself
        assert experiment(name).exp_name == name
    with pytest.raises(ValueError, match="unknown experiment"):
        experiment("nope")


def test_end_to_end_tiny_run(hub_files: dict, tmp_path) -> None:
    """Two epochs on 24 samples of one Hub shard with the real weights, scoring on a shard, and
    the summary over two such runs (the second one single-scan)."""
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
    results = run(experiment("baseline", exp_name="t", **base))
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
    run(experiment("one_scan", exp_name="u", seed=7, **base))
    runs = find_runs([str(tmp_path / "exp")])
    assert len(runs) == 2
    rows = [r for run_dir in runs for r in run_rows(run_dir)]
    t = table(rows)
    assert set(t["exp_name"]) == {"t", "u"} and len(t) == 4
    c = compare(rows, "t", "u", n_boot=5)
    assert len(c) == 2 and "WC" in c[0]
