import json
import os

import numpy as np
import pytest
import torch

from tornado_onset.config import Config, load_config
from tornado_onset.data import N_INPUT_CHANNELS, OnsetDataset, collate, load_catalog, select_files
from tornado_onset.model import TARGET_CHANNELS, Losses, OnsetModel, full_chip
from tornado_onset.tornet import TorNetDetector

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


@pytest.mark.parametrize("decoder", [True, False])
def test_forward_shapes(decoder: bool) -> None:
    m = OnsetModel(
        n_in=2, horizons=2, start_filters=8, decoder_channels=[32, 16, 8, 8], decoder=decoder
    ).eval()
    b = tiny_batch()
    out = m(b["x"], b["coords"])
    assert m.draws == decoder == ("fields" in out)
    if decoder:
        assert out["fields"].shape == (2, 2, len(TARGET_CHANNELS), AZ, RNG)
    assert out["cls_map"].shape == (2, 2, -(-AZ // 16), -(-RNG // 16))
    assert out["logits"].shape == (2, 2)
    assert out["features"].shape[:2] == (2, 2 * 8 * 8)
    assert torch.equal(out["logits"], out["cls_map"].amax(dim=(2, 3)))


def test_full_chip_persists_other_channels() -> None:
    b = tiny_batch()
    last = b["x"][:, -1]
    drawn = torch.zeros(2, len(TARGET_CHANNELS), AZ, RNG)
    chip = full_chip(drawn, last)
    other = [i for i in range(N_INPUT_CHANNELS) if i not in TARGET_CHANNELS]
    assert torch.equal(chip[:, other], last[:, other])
    assert torch.equal(chip[:, TARGET_CHANNELS], drawn)
    assert not torch.equal(last[:, TARGET_CHANNELS], drawn)  # the clone did not touch the input


@pytest.mark.parametrize("heatmap_weight,detector_weight", [(0.0, 0.0), (2.0, 0.5)])
def test_losses_route_gradients(heatmap_weight: float, detector_weight: float) -> None:
    cfg = Config(heatmap_weight=heatmap_weight, detector_weight=detector_weight, bootstrap=0)
    judge = TorNetDetector(start_filters=4).eval().requires_grad_(False)
    m = OnsetModel(n_in=2, horizons=2, start_filters=8, decoder_channels=[32, 16, 8, 8])
    losses = Losses(cfg, torch.device("cpu"), judge)
    b = tiny_batch()
    out = m(b["x"], b["coords"])
    total, parts = losses(out, b)
    assert set(parts) == ({"field", "cls", "detector"} if detector_weight else {"field", "cls"})
    assert torch.isfinite(total)
    total.backward()
    assert all(p.grad is not None for p in m.encoder.parameters())
    assert all(p.grad is not None for p in m.classification.parameters())
    assert all(p.grad is None for p in judge.parameters())  # the judge never learns


def test_head_only_trains_without_drawing_losses() -> None:
    cfg = Config(decoder=False, field_weight=0.0, heatmap_weight=0.0, detector_weight=0.0)
    m = OnsetModel(n_in=2, horizons=2, start_filters=8, decoder=False)
    b = tiny_batch()
    total, parts = Losses(cfg, torch.device("cpu"), None)(m(b["x"], b["coords"]), b)
    assert set(parts) == {"cls"}
    total.backward()
    assert all(p.grad is not None for p in m.encoder.parameters())
    with pytest.raises(ValueError, match="head-only"):
        Config(decoder=False)


def test_cls_weight_zero_leaves_head_untrained() -> None:
    cfg = Config(cls_weight=0.0, heatmap_weight=0.0, detector_weight=0.0)
    m = OnsetModel(n_in=2, horizons=2, start_filters=8, decoder_channels=[32, 16, 8, 8])
    b = tiny_batch()
    total, _ = Losses(cfg, torch.device("cpu"), None)(m(b["x"], b["coords"]), b)
    total.backward()
    assert all(p.grad is None or torch.all(p.grad == 0) for p in m.classification.parameters())


def test_config_rejects_unknown_keys(tmp_path) -> None:
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"epochs": 1, "bogus": 2}))
    with pytest.raises(ValueError, match="bogus"):
        load_config(p)
    p.write_text(json.dumps({"epochs": 1, "_note": "ignored"}))
    assert load_config(p).epochs == 1
    cfg = load_config(None, epochs=1, sample_weights={"w2": 3.0})
    assert cfg.sample_weights["w2"] == 3.0 and cfg.sample_weights["wW"] == 0.5


def test_end_to_end_tiny_run(tornet_root: str, tmp_path) -> None:
    """One epoch on a handful of 2015 files with the real weights, then scoring."""
    if not os.path.exists(Config().weights):
        pytest.skip("run onset-convert-weights first")
    from tornado_onset.train import run

    cfg = load_config(
        None,
        max_files=24,
        epochs=1,
        batch_size=4,
        workers=0,
        bootstrap=5,
        exp_dir=str(tmp_path),
        exp_name="t",
        val_fraction=0.3,
    )
    # a tiny data root: 24 training and 12 test files from 2015, symlinked from the real one
    cat = load_catalog(tornet_root)
    test_rows = select_files(cat, "test", [2015]).head(12)
    train_rows = select_files(cat, "train", [2015]).head(24)
    small = tmp_path / "data"
    small.mkdir()
    import pandas as pd

    pd.concat([train_rows, test_rows]).drop(columns=["year", "day"]).to_csv(
        small / "catalog.csv", index=False
    )
    for sub in ("train/2015", "test/2015"):
        (small / sub).mkdir(parents=True)
    for f in pd.concat([train_rows, test_rows])["filename"]:
        os.symlink(os.path.join(tornet_root, f), small / f)
    results = run(cfg, str(small))
    assert results["epochs_run"] == 1
    assert os.path.exists(results["best_checkpoint"])
    scores = os.path.join(results["expdir"], "scores")
    assert {
        f"predictions_{w}_h{h}.csv"
        for w in ("head", "persistence", "forecast", "truth")
        for h in (0, 1)
    } <= set(os.listdir(scores))
    feats = np.load(os.path.join(scores, "features.npz"))
    assert feats["features_max"].shape == (12, 2 * 384) and feats["head_logits"].shape == (12, 2)
    with open(os.path.join(scores, "summary.json")) as f:
        s = json.load(f)
    assert len(s["horizons"]) == 2 and "mae_by_product" in s["horizons"][0]
    ds = OnsetDataset(str(small), test_rows["filename"].tolist()[:2])
    assert collate([ds[0], ds[1]])["x"].shape[1] == 2
