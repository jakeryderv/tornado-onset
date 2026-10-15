"""The onset model: the published TorNet encoder, fine-tuned, on the observed scans, with the
authors' head on top.

    observed scans (input_scans x 14 channels, polar)
            |
    TorNet encoder, run per scan -> one 384 x 8 x 15 feature map per scan
            |
    combiner
      concat       maps side by side in channels (768); one head with `horizons` output maps
      extrapolate  f(+h) = f_new + (lead_h / dt) * (f_new - f_old) per horizon, using the real
                   scan timing; one shared 384-channel head applied to each f(+h)
            |
    the authors' head: 1x1 convolutions to a per-cell logit map, max over cells
            -> P(tornado at +5), P(tornado at +10)

One loss: weighted cross-entropy on the per-horizon logits against the labels of the future
scans. The frozen published detector (the judge) is not part of training; it scores the floor
(judge on the newest observed scan) and the ceiling (judge on the real future scan).
"""

from __future__ import annotations

import torch
from torch import nn

from tornado_onset.config import Config
from tornado_onset.tornet import TorNetEncoder, TorNetHead, load_state_dict


class OnsetModel(nn.Module):
    def __init__(
        self, n_in: int = 2, horizons: int = 2, start_filters: int = 48, combiner: str = "concat"
    ):
        super().__init__()
        self.n_in, self.horizons, self.combiner = n_in, horizons, combiner
        self.encoder = TorNetEncoder(start_filters=start_filters)
        c = self.encoder.out_channels
        if combiner == "concat":
            self.head = TorNetHead(c * n_in, out_channels=horizons)
        else:
            self.head = TorNetHead(c, out_channels=1)

    def scan_features(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """(B, T, 14, A, R) -> (B, T, C, h, w): the encoder's bottleneck map per scan."""
        b, n = x.shape[:2]
        f = self.encoder(x.flatten(0, 1), coords.repeat_interleave(n, dim=0))
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
        self, x: torch.Tensor, coords: torch.Tensor, minutes: torch.Tensor, dt_in: torch.Tensor
    ) -> torch.Tensor:
        """Per-horizon logits, (B, horizons)."""
        per_scan = self.scan_features(x, coords)
        if self.combiner == "concat":
            return self.head(per_scan.flatten(1, 2))
        future = self.extrapolate(per_scan, minutes, dt_in)  # (B, horizons, C, h, w)
        return self.head(future.flatten(0, 1)).reshape(x.shape[0], self.horizons)


def build_model(cfg: Config) -> OnsetModel:
    """The model with its encoder (and optionally its head) initialised from the published
    detector's weights."""
    sd = load_state_dict(cfg.weights)
    start_filters = int(sd["encoder.blocks.0.0.conv.weight"].shape[0])
    model = OnsetModel(cfg.input_scans, cfg.horizons, start_filters, cfg.combiner)
    enc = {k.removeprefix("encoder."): v for k, v in sd.items() if k.startswith("encoder.")}
    model.encoder.load_state_dict(enc, strict=True)
    if cfg.head_init == "pretrained":
        head = {k.removeprefix("head."): v for k, v in sd.items() if k.startswith("head.")}
        model.head.load_state_dict(head, strict=True)
    if not cfg.encoder_trainable:
        model.encoder.requires_grad_(False)
    return model


def onset_loss(logits: torch.Tensor, labels: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Sample-weighted binary cross-entropy, mean over samples and horizons."""
    bce = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return (bce * weight[:, None]).mean()
