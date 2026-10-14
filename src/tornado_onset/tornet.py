"""The published TorNet detector as PyTorch, and its weights read straight from the `.keras` file.

`TorNetEncoder` is the backbone: four blocks of CoordConv layers (a 3x3 convolution over the
features concatenated with the two coordinate channels), each block followed by 2x2 max pooling
of features and coordinates and a dropout. It returns the feature pyramid in the convention
segmentation_models_pytorch uses, one tensor per stride (1, 2, 4, 8, 16), so it drops into an
SMP U-Net decoder. `TorNetHead` is the authors' max-pool head: 1x1 convolutions to a per-cell
logit map, max over cells. `TorNetDetector` is both, the published model.

A Keras 3 `.keras` file is a zip holding `model.weights.h5`; `state_dict_from_keras` reads that
with h5py and maps layer by layer, so Keras is not a dependency. `onset-convert-weights` writes
the state dict once and `--verify` checks the logits against the reference predictions saved
from the Keras model.
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from typing import cast

import h5py
import numpy as np
import torch
from torch import nn

from tornado_onset.data import N_INPUT_CHANNELS

HF_REPO = "tornet-ml/tornado_detector_baseline_v1"
HF_FILE = "tornado_detector_baseline.keras"
BLOCK_CONVS = (2, 2, 3, 3)
COORD_CHANNELS = 2


class CoordConv(nn.Module):
    """3x3 convolution over [features, coordinates], ReLU; returns features only."""

    def __init__(self, in_channels: int, out_channels: int, coord_channels: int = COORD_CHANNELS):
        super().__init__()
        self.conv = nn.Conv2d(in_channels + coord_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.conv(torch.cat([x, coords], dim=1)))


class TorNetEncoder(nn.Module):
    """`out_channels` lists the channels of each returned feature, strides 1, 2, 4, 8, 16."""

    def __init__(
        self,
        in_channels: int = N_INPUT_CHANNELS,
        start_filters: int = 48,
        block_convs: tuple[int, ...] = BLOCK_CONVS,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        channels = in_channels
        self.out_channels: list[int] = []
        for b, n_convs in enumerate(block_convs):
            filters = start_filters * 2**b
            layers = nn.ModuleList()
            for _ in range(n_convs):
                layers.append(CoordConv(channels, filters))
                channels = filters
            self.blocks.append(layers)
            self.out_channels.append(filters)
        self.out_channels.append(channels)  # the bottleneck after the last pool
        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # Keras padding="same"
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> list[torch.Tensor]:
        features = []
        c = coords
        for layers in self.blocks:
            for layer in cast(nn.ModuleList, layers):
                x = layer(x, c)
            features.append(x)
            x = self.dropout(self.pool(x))
            c = self.pool(c)
        features.append(x)
        return features

    def conv_layers(self) -> list[nn.Conv2d]:
        return [
            cast(CoordConv, layer).conv
            for layers in self.blocks
            for layer in cast(nn.ModuleList, layers)
        ]


class TorNetHead(nn.Module):
    """Two 1x1 convolutions with ReLU, a 1x1 map (one per output), max over cells. The map
    before the max is the paper's spatial likelihood map."""

    def __init__(
        self, in_channels: int, hidden: tuple[int, ...] = (512, 256), out_channels: int = 1
    ):
        super().__init__()
        layers: list[nn.Module] = []
        channels = in_channels
        for units in hidden:
            layers += [nn.Conv2d(channels, units, 1), nn.ReLU()]
            channels = units
        self.hidden = nn.Sequential(*layers)
        self.heatmap = nn.Conv2d(channels, out_channels, 1)

    def heat(self, features: torch.Tensor) -> torch.Tensor:
        """Per-cell logits, (batch, out_channels, H, W)."""
        return self.heatmap(self.hidden(features))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.heat(features).amax(dim=(2, 3))  # (batch, out_channels)

    def conv_layers(self) -> list[nn.Conv2d]:
        return [m for m in self.hidden if isinstance(m, nn.Conv2d)] + [self.heatmap]


class TorNetDetector(nn.Module):
    """Backbone plus head: the published tornado detector."""

    def __init__(self, start_filters: int = 48, in_channels: int = N_INPUT_CHANNELS):
        super().__init__()
        self.encoder = TorNetEncoder(in_channels=in_channels, start_filters=start_filters)
        self.head = TorNetHead(self.encoder.out_channels[-1])

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x, coords)[-1])

    def heat(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        return self.head.heat(self.encoder(x, coords)[-1])


# --- weights ---------------------------------------------------------------------------------


def download_keras() -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=HF_REPO, filename=HF_FILE)


def _ordered(names: list[str], prefix: str) -> list[str]:
    """Keras names `prefix`, `prefix_1`, `prefix_2`, ... in numeric order."""
    pat = re.compile(rf"^{re.escape(prefix)}(?:_(\d+))?$")
    found = [(int(m.group(1) or 0), n) for n in names if (m := pat.match(n))]
    return [n for _, n in sorted(found)]


def _kernel(k: np.ndarray) -> torch.Tensor:
    """Keras (kh, kw, in, out) -> torch (out, in, kh, kw)."""
    return torch.as_tensor(np.ascontiguousarray(np.transpose(k, (3, 2, 0, 1))))


def state_dict_from_keras(path: str | Path) -> dict[str, torch.Tensor]:
    """Read `model.weights.h5` out of the `.keras` zip and name every tensor as
    `TorNetDetector.state_dict()` names it."""
    with zipfile.ZipFile(path) as z:
        raw = z.read("model.weights.h5")
    with h5py.File(io.BytesIO(raw)) as f:
        layers = f["layers"]
        assert isinstance(layers, h5py.Group)
        names = list(layers.keys())
        coord = _ordered(names, "coord_conv2d")
        head = _ordered(names, "conv2d")
        kernels = [np.asarray(layers[f"{n}/conv/vars/0"]) for n in coord]
        biases = [np.asarray(layers[f"{n}/conv/vars/1"]) for n in coord]
        head_kernels = [np.asarray(layers[f"{n}/vars/0"]) for n in head]
        head_biases = [np.asarray(layers[f"{n}/vars/1"]) for n in head]
    start_filters = int(kernels[0].shape[-1])
    model = TorNetDetector(start_filters=start_filters)
    enc, hd = model.encoder.conv_layers(), model.head.conv_layers()
    if len(enc) != len(kernels) or len(hd) != len(head_kernels):
        raise ValueError(
            f"{len(kernels)} CoordConv and {len(head_kernels)} head layers in the file, "
            f"{len(enc)} and {len(hd)} in the module"
        )
    with torch.no_grad():
        for conv, k, b in zip(enc + hd, kernels + head_kernels, biases + head_biases, strict=True):
            w = _kernel(k)
            if tuple(w.shape) != tuple(conv.weight.shape):
                raise ValueError(f"kernel {tuple(w.shape)} != {tuple(conv.weight.shape)}")
            conv.weight.copy_(w)
            assert conv.bias is not None
            conv.bias.copy_(torch.as_tensor(np.ascontiguousarray(b)))
    return {k: v.clone() for k, v in model.state_dict().items()}


def convert(keras_path: str | Path | None, out_path: str | Path) -> dict[str, torch.Tensor]:
    sd = state_dict_from_keras(keras_path or download_keras())
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, out)
    return sd


def load_detector(path: str | Path, device: str | torch.device = "cpu") -> TorNetDetector:
    """The published detector, frozen, from a converted state dict."""
    sd = torch.load(path, map_location="cpu")
    start_filters = int(sd["encoder.blocks.0.0.conv.weight"].shape[0])
    model = TorNetDetector(start_filters=start_filters)
    model.load_state_dict(sd)
    return model.eval().requires_grad_(False).to(device)
