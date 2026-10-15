"""The onset model: the published TorNet encoder, fine-tuned, on the observed scans, with the
authors' head on top.

    observed scans (n_in x 14 channels, polar)
            |
    TorNet encoder (fine-tuned), run per scan -> one 384 x 8 x 15 feature map per scan
            |
    combiner
      concat       maps side by side in channels (768); one head with `horizons` output maps
      extrapolate  f(+h) = f_new + (lead_h / dt) * (f_new - f_old) per horizon, using the real
                   scan timing; one shared 384-channel head applied to each f(+h)
            |
    the authors' head: 1x1 convolutions to a per-cell logit map, max over cells
            -> P(tornado at +5), P(tornado at +10)

Losses: weighted cross-entropy on the head's per-horizon logits against the labels of the
future scans; with `consistency_weight`, an L1 pulling each extrapolated map f(+h) toward the
encoder's own features of the real scan at +h (targets detached, so the encoder cannot cheat by
flattening its features). The frozen published detector (the judge) is not part of training; it scores the
floor (judge on the newest observed scan) and the ceiling (judge on the real future scan) at
evaluation. A U-Net decoder drawing the future radar as an auxiliary task was tried and dropped:
on 2015 it cost accuracy, time and half the parameters (see docs/project.md).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from tornado_onset.config import Config, resolve_weights
from tornado_onset.tornet import TorNetEncoder, TorNetHead


class OnsetModel(nn.Module):
    def __init__(
        self, n_in: int = 2, horizons: int = 2, start_filters: int = 48, combiner: str = "concat"
    ):
        super().__init__()
        self.n_in, self.horizons, self.combiner = n_in, horizons, combiner
        self.encoder = TorNetEncoder(start_filters=start_filters)
        c = self.encoder.out_channels[-1]
        if combiner == "concat":
            self.feature_channels = c * n_in
            self.head = TorNetHead(self.feature_channels, out_channels=horizons)
        elif combiner == "extrapolate":
            self.feature_channels = c
            self.head = TorNetHead(c, out_channels=1)
        else:
            raise ValueError(f"unknown combiner {combiner!r}")

    def scan_features(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """(B, T, 14, A, R) -> (B, T, C, h, w): the encoder's bottleneck map per scan."""
        b, n = x.shape[:2]
        f = self.encoder(x.flatten(0, 1), coords.repeat_interleave(n, dim=0))[-1]
        return f.reshape(b, n, *f.shape[1:])

    def extrapolate(
        self, per_scan: torch.Tensor, minutes: torch.Tensor, dt_in: torch.Tensor
    ) -> torch.Tensor:
        """(B, n_in, C, h, w) -> (B, horizons, C, h, w): the newest map plus the per-minute
        change between the last two scans, scaled by each horizon's real lead."""
        new, old = per_scan[:, -1], per_scan[:, -2]
        rate = (new - old) / dt_in.clamp_min(1.0).view(-1, 1, 1, 1)
        return new[:, None] + minutes.view(*minutes.shape, 1, 1, 1) * rate[:, None]

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        minutes: torch.Tensor | None = None,
        dt_in: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        per_scan = self.scan_features(x, coords)
        if self.combiner == "concat":
            feats = per_scan.flatten(1, 2)  # (B, n_in * C, h, w)
            cls_map = self.head.heat(feats)  # (B, horizons, h, w)
            return {"cls_map": cls_map, "logits": cls_map.amax(dim=(2, 3)), "features": feats}
        assert minutes is not None and dt_in is not None, "extrapolate needs the scan timing"
        future = self.extrapolate(per_scan, minutes, dt_in)  # (B, horizons, C, h, w)
        b, h = future.shape[:2]
        cls_map = self.head.heat(future.flatten(0, 1)).reshape(b, h, *future.shape[3:])
        return {
            "cls_map": cls_map,
            "logits": cls_map.amax(dim=(2, 3)),
            "features": torch.cat([per_scan[:, -1], per_scan[:, -1] - per_scan[:, -2]], dim=1),
            "future_features": future,
        }


def build_model(cfg: Config) -> OnsetModel:
    """The model with its encoder (and optionally its head) initialised from the converted
    published weights."""
    model = OnsetModel(cfg.n_in, cfg.horizons, cfg.start_filters, cfg.combiner)
    sd = torch.load(resolve_weights(cfg.weights), map_location="cpu")
    enc = {k.removeprefix("encoder."): v for k, v in sd.items() if k.startswith("encoder.")}
    model.encoder.load_state_dict(enc, strict=True)
    if cfg.head_init == "pretrained":
        head = {k.removeprefix("head."): v for k, v in sd.items() if k.startswith("head.")}
        model.head.load_state_dict(head, strict=True)
    if not cfg.encoder_trainable:
        model.encoder.requires_grad_(False)
    return model


class Losses:
    """Callable: (model output, batch) -> (total, {term: value}). With `consistency_weight` the
    model is needed too, to encode the real future scans as (detached) targets."""

    def __init__(self, cfg: Config, device: torch.device, model: OnsetModel | None = None):
        self.cfg = cfg
        self.device = device
        self.model = model
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        if cfg.consistency_weight and model is None:
            raise ValueError("consistency_weight needs the model")

    def __call__(
        self, out: dict[str, torch.Tensor], batch: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        labels = batch["labels"].to(self.device)
        weight = batch["weight"].to(self.device)[:, None]
        cls = (self.bce(out["logits"], labels) * weight).mean()
        total = self.cfg.cls_weight * cls
        parts = {"cls": cls.item()}
        if self.cfg.consistency_weight:
            assert self.model is not None and "future_features" in out
            with torch.no_grad():
                target = self.model.scan_features(
                    batch["future"].to(self.device), batch["coords"].to(self.device)
                )
            cons = (out["future_features"] - target.float()).abs().mean()
            total = total + self.cfg.consistency_weight * cons
            parts["consistency"] = cons.item()
        return total, parts
