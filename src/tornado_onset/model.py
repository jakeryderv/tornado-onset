"""The onset model: the published TorNet encoder, fine-tuned, on the observed scans, with the
authors' head on top.

    observed scans (n_in x 14 channels, polar)
            |
    TorNet encoder (fine-tuned), run per scan, features concatenated across scans
            |
    the authors' head: 1x1 convolutions to a per-cell logit map, max over cells
            -> P(tornado at +5), P(tornado at +10)

One loss: weighted cross-entropy on the head's per-horizon logits against the labels of the
future scans. The frozen published detector (the judge) is not part of training; it scores the
floor (judge on the newest observed scan) and the ceiling (judge on the real future scan) at
evaluation. A U-Net decoder drawing the future radar as an auxiliary task was tried and dropped:
on 2015 it cost accuracy, time and half the parameters (see docs/project.md).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from tornado_onset.config import Config
from tornado_onset.tornet import TorNetEncoder, TorNetHead


class OnsetModel(nn.Module):
    def __init__(self, n_in: int = 2, horizons: int = 2, start_filters: int = 48):
        super().__init__()
        self.n_in, self.horizons = n_in, horizons
        self.encoder = TorNetEncoder(start_filters=start_filters)
        self.feature_channels = self.encoder.out_channels[-1] * n_in
        self.head = TorNetHead(self.feature_channels, out_channels=horizons)

    def features(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """(B, n_in, 14, A, R) -> bottleneck features with the scans side by side in channels."""
        b, n = x.shape[:2]
        f = self.encoder(x.flatten(0, 1), coords.repeat_interleave(n, dim=0))[-1]
        return f.reshape(b, n * f.shape[1], *f.shape[2:])

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.features(x, coords)
        cls_map = self.head.heat(feats)  # (B, horizons, 8, 15), the likelihood map
        return {
            "cls_map": cls_map,
            "logits": cls_map.amax(dim=(2, 3)),
            "features": feats,  # (B, n_in * 384, 8, 15), for the per-sample table
        }


def build_model(cfg: Config) -> OnsetModel:
    """The model with its encoder initialised from the converted published weights."""
    model = OnsetModel(cfg.n_in, cfg.horizons, cfg.start_filters)
    sd = torch.load(cfg.weights, map_location="cpu")
    enc = {k.removeprefix("encoder."): v for k, v in sd.items() if k.startswith("encoder.")}
    model.encoder.load_state_dict(enc, strict=True)
    if not cfg.encoder_trainable:
        model.encoder.requires_grad_(False)
    return model


class Losses:
    """Callable: (model output, batch) -> (total, {term: value})."""

    def __init__(self, cfg: Config, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def __call__(
        self, out: dict[str, torch.Tensor], batch: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        labels = batch["labels"].to(self.device)
        weight = batch["weight"].to(self.device)[:, None]
        cls = (self.bce(out["logits"], labels) * weight).mean()
        return self.cfg.cls_weight * cls, {"cls": cls.item()}
