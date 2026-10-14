import io
import zipfile

import h5py
import numpy as np
import torch

from tornado_onset.data import N_INPUT_CHANNELS
from tornado_onset.tornet import (
    BLOCK_CONVS,
    TorNetDetector,
    TorNetEncoder,
    convert,
    load_detector,
    state_dict_from_keras,
)


def test_encoder_pyramid_shapes() -> None:
    enc = TorNetEncoder(start_filters=48)
    x = torch.zeros(1, N_INPUT_CHANNELS, 120, 240)
    c = torch.zeros(1, 2, 120, 240)
    feats = enc(x, c)
    assert [tuple(f.shape[1:]) for f in feats] == [
        (48, 120, 240),
        (96, 60, 120),
        (192, 30, 60),
        (384, 15, 30),
        (384, 8, 15),  # ceil-mode pool, as Keras padding="same"
    ]
    assert enc.out_channels == [48, 96, 192, 384, 384]
    assert len(enc.conv_layers()) == sum(BLOCK_CONVS)


def test_detector_outputs() -> None:
    det = TorNetDetector(start_filters=8).eval()  # dropout off, so two passes agree
    x = torch.randn(2, N_INPUT_CHANNELS, 120, 240)
    c = torch.randn(2, 2, 120, 240)
    assert det(x, c).shape == (2, 1)
    assert det.heat(x, c).shape == (2, 1, 8, 15)
    assert torch.equal(det(x, c), det.heat(x, c).amax(dim=(2, 3)))


def _fake_keras(path, start_filters: int = 8) -> TorNetDetector:
    """A .keras zip with the published file's HDF5 layout and random weights; returns a
    detector holding the same weights for comparison."""
    det = TorNetDetector(start_filters=start_filters)
    buf = io.BytesIO()
    rng = np.random.default_rng(0)
    with h5py.File(buf, "w") as f:
        enc = det.encoder.conv_layers()
        for i, conv in enumerate(enc):
            name = "coord_conv2d" if i == 0 else f"coord_conv2d_{i}"
            k = rng.standard_normal((3, 3, conv.in_channels, conv.out_channels)).astype(np.float32)
            b = rng.standard_normal(conv.out_channels).astype(np.float32)
            f.create_dataset(f"layers/{name}/conv/vars/0", data=k)
            f.create_dataset(f"layers/{name}/conv/vars/1", data=b)
            with torch.no_grad():
                conv.weight.copy_(torch.as_tensor(k.transpose(3, 2, 0, 1)))
                assert conv.bias is not None
                conv.bias.copy_(torch.as_tensor(b))
        for i, conv in enumerate(det.head.conv_layers()):
            name = "conv2d" if i == 0 else f"conv2d_{i}"
            k = rng.standard_normal((1, 1, conv.in_channels, conv.out_channels)).astype(np.float32)
            b = rng.standard_normal(conv.out_channels).astype(np.float32)
            f.create_dataset(f"layers/{name}/vars/0", data=k)
            f.create_dataset(f"layers/{name}/vars/1", data=b)
            with torch.no_grad():
                conv.weight.copy_(torch.as_tensor(k.transpose(3, 2, 0, 1)))
                assert conv.bias is not None
                conv.bias.copy_(torch.as_tensor(b))
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("config.json", "{}")
        z.writestr("model.weights.h5", buf.getvalue())
    return det


def test_state_dict_from_keras_matches_layer_order(tmp_path) -> None:
    path = tmp_path / "fake.keras"
    expected = _fake_keras(path)
    sd = state_dict_from_keras(path)
    assert sd.keys() == expected.state_dict().keys()
    for k, v in expected.state_dict().items():
        assert torch.equal(sd[k], v), k


def test_convert_and_load_round_trip(tmp_path) -> None:
    path = tmp_path / "fake.keras"
    expected = _fake_keras(path)
    out = tmp_path / "w" / "det.pt"
    convert(path, out)
    det = load_detector(out)
    assert not any(p.requires_grad for p in det.parameters())
    x = torch.randn(1, N_INPUT_CHANNELS, 120, 240)
    c = torch.randn(1, 2, 120, 240)
    expected.eval()
    assert torch.allclose(det(x, c), expected(x, c))
