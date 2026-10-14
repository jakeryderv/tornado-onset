"""The onset model and its losses.

    observed scans (n_in x 14 channels, polar)
            |
    TorNet encoder (fine-tuned), run per scan, features concatenated across scans at every scale
            |
    U-Net decoder (segmentation_models_pytorch) -> drawn future VEL + DBZ at each horizon
    bottleneck features -> the authors' head: per-cell logits, max -> P(tornado) per horizon

Two outputs, two routes to "tornado at +h". The head is the product. With `decoder: false`
the model is head-only: encoder and classification head, trained by the head's cross-entropy
alone, the ablation that says whether the drawing earns its place. The drawing is the auxiliary
signal: its loss terms reach the shared encoder and that is what lifts the head (fold B in the
fork: 0.741 -> 0.757 WC AUC at +10 min). The frozen published detector (the judge) sits inside
the loss, reading the drawn chip; its weights never move.

Losses, one total, each term updating only the parts that produced its output:
    field    L1 on VEL + DBZ in the detector's normalised space, times (1 + heatmap_weight *
             the judge's min-max-normalised heat map on the real target scan)   -> decoder, encoder
    detector the judge's weighted cross-entropy on the drawn chip (drawn products placed into
             the last observed scan; the other products and the range-folded channels persist)
                                                                                 -> decoder, encoder
    cls      weighted cross-entropy on the head's per-horizon logits             -> head, encoder
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from tornado_onset.config import Config
from tornado_onset.data import channel_indices
from tornado_onset.tornet import TorNetDetector, TorNetEncoder, TorNetHead, load_detector

TARGET_PRODUCTS = ["VEL", "DBZ"]
TARGET_CHANNELS = channel_indices(TARGET_PRODUCTS)  # [2, 3, 0, 1]


class OnsetModel(nn.Module):
    def __init__(
        self,
        n_in: int = 2,
        horizons: int = 2,
        start_filters: int = 48,
        decoder_channels: list[int] | None = None,
        decoder: bool = True,
    ):
        super().__init__()
        self.n_in, self.horizons = n_in, horizons
        self.encoder = TorNetEncoder(start_filters=start_filters)
        enc_channels = [c * n_in for c in self.encoder.out_channels]
        self.feature_channels = enc_channels[-1]
        self.decoder: nn.Module | None = None
        self.regression: nn.Module | None = None
        if decoder:
            from segmentation_models_pytorch.base import SegmentationHead
            from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder

            dec = list(decoder_channels or [256, 128, 64, 32])
            self.decoder = UnetDecoder(enc_channels, dec, n_blocks=len(dec), use_norm="batchnorm")
            self.regression = SegmentationHead(
                dec[-1], horizons * len(TARGET_CHANNELS), kernel_size=3
            )
        self.classification = TorNetHead(enc_channels[-1], out_channels=horizons)

    @property
    def draws(self) -> bool:
        return self.decoder is not None

    def encode(self, x: torch.Tensor, coords: torch.Tensor) -> list[torch.Tensor]:
        """(B, n_in, 14, A, R) -> feature pyramid with the scans side by side in channels."""
        b, n = x.shape[:2]
        feats = self.encoder(x.flatten(0, 1), coords.repeat_interleave(n, dim=0))
        return [f.reshape(b, n * f.shape[1], *f.shape[2:]) for f in feats]

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.encode(x, coords)
        cls_map = self.classification.heat(feats[-1])  # (B, horizons, 8, 15)
        out = {
            "cls_map": cls_map,
            "logits": cls_map.amax(dim=(2, 3)),
            "features": feats[-1],  # (B, n_in * 384, 8, 15), for the per-sample table
        }
        if self.decoder is not None and self.regression is not None:
            fields = self.regression(self.decoder(feats))
            b, _, a, r = fields.shape
            out["fields"] = fields.reshape(b, self.horizons, len(TARGET_CHANNELS), a, r)
        return out


def build_model(cfg: Config) -> OnsetModel:
    """The model with its encoder initialised from the converted published weights."""
    model = OnsetModel(
        cfg.n_in, cfg.horizons, cfg.start_filters, cfg.decoder_channels, decoder=cfg.decoder
    )
    sd = torch.load(cfg.weights, map_location="cpu")
    enc = {k.removeprefix("encoder."): v for k, v in sd.items() if k.startswith("encoder.")}
    model.encoder.load_state_dict(enc, strict=True)
    if not cfg.encoder_trainable:
        model.encoder.requires_grad_(False)
    return model


def full_chip(forecast: torch.Tensor, last_x: torch.Tensor) -> torch.Tensor:
    """The drawn VEL + DBZ placed into the last observed scan: (B, 14, A, R) for the judge."""
    chip = last_x.clone()
    chip[:, TARGET_CHANNELS] = forecast
    return chip


class Losses:
    """Callable: (model output, batch) -> (total, {term: value}). Holds the frozen judge."""

    def __init__(self, cfg: Config, device: torch.device, judge: TorNetDetector | None = None):
        self.cfg = cfg
        self.device = device
        self.judge = judge
        if judge is None and (cfg.heatmap_weight > 0 or cfg.detector_weight > 0):
            self.judge = load_detector(cfg.weights, device)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def spatial_weight(self, future: torch.Tensor, coords: torch.Tensor) -> torch.Tensor | None:
        """1 + heatmap_weight * the judge's heat map on the real target scan, min-max normalised
        per scan and upsampled to the chip: (B, horizons, 1, A, R)."""
        if self.cfg.heatmap_weight <= 0:
            return None
        assert self.judge is not None
        b, h, _, a, r = future.shape
        with torch.no_grad():
            heat = self.judge.heat(future.flatten(0, 1), coords.repeat_interleave(h, dim=0))
            heat = nn.functional.interpolate(heat, size=(a, r), mode="bilinear")
            lo = heat.amin(dim=(2, 3), keepdim=True)
            hi = heat.amax(dim=(2, 3), keepdim=True)
            heat = (heat - lo) / (hi - lo + 1e-6)
        return 1 + self.cfg.heatmap_weight * heat.reshape(b, h, 1, a, r)

    def __call__(
        self, out: dict[str, torch.Tensor], batch: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        cfg, dev = self.cfg, self.device
        future = batch["future"].to(dev)
        coords = batch["coords"].to(dev)
        labels = batch["labels"].to(dev)
        weight = batch["weight"].to(dev)[:, None]
        cls = (self.bce(out["logits"], labels) * weight).mean()
        total = cfg.cls_weight * cls
        parts = {"cls": float(cls)}
        if "fields" not in out:
            return total, parts
        err = (out["fields"] - future[:, :, TARGET_CHANNELS]).abs()
        sw = self.spatial_weight(future, coords)
        if sw is not None:
            err = err * sw
        field = err.mean()
        total = total + cfg.field_weight * field
        parts["field"] = float(field)
        if cfg.detector_weight > 0:
            assert self.judge is not None
            last = batch["x"][:, -1].to(dev)
            judged = torch.cat(
                [
                    self.judge(full_chip(out["fields"][:, h], last), coords)
                    for h in range(out["fields"].shape[1])
                ],
                dim=1,
            )
            det = (self.bce(judged, labels) * weight).mean()
            total = total + cfg.detector_weight * det
            parts["detector"] = float(det)
        return total, parts
