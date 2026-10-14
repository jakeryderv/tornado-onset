"""Run configuration: a dataclass with the agreed defaults, overridden by a JSON file.

The loss design is fixed to what worked on fold B in the TorNet fork (heat-map-weighted L1 on
velocity and reflectivity, the frozen detector's cross-entropy on the drawn chip, cross-entropy on
the classification head). The weights of those three terms stay configurable so a control
(`heatmap_weight` 0, `detector_weight` 0) or a `cls_weight` 0 ablation is one JSON line, but
there are no channel-scope, anchor or bottleneck options.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_WEIGHTS = "weights/tornet_detector_v1.pt"


@dataclass
class Config:
    # data: the authors' partitions; validation is held out from the training partition by day
    train_years: list[int] = field(default_factory=lambda: [2015])
    test_years: list[int] = field(default_factory=lambda: [2015])
    val_fraction: float = 0.15
    n_in: int = 2
    horizons: int = 2
    max_files: int | None = None  # cap on training files, for smoke runs
    # model
    weights: str = DEFAULT_WEIGHTS  # the converted published detector: encoder init + judge
    encoder_trainable: bool = True
    start_filters: int = 48
    decoder_channels: list[int] = field(default_factory=lambda: [256, 128, 64, 32])
    # losses
    field_weight: float = 1.0
    heatmap_weight: float = 2.0  # spatial weight = 1 + heatmap_weight * judge heat map on truth
    cls_weight: float = 1.0
    detector_weight: float = 0.5  # the judge's cross-entropy on the drawn chip
    sample_weights: dict[str, float] = field(
        default_factory=lambda: {"wN": 1.0, "w0": 1.0, "w1": 1.0, "w2": 2.0, "wW": 0.5}
    )
    # training
    epochs: int = 6
    batch_size: int = 16
    learning_rate: float = 1e-4
    encoder_lr_scale: float = 0.1
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    amp: bool = True
    workers: int = 8
    seed: int = 1234
    # evaluation
    bootstrap: int = 1000
    # bookkeeping
    exp_name: str = "onset"
    exp_dir: str = field(default_factory=lambda: os.environ.get("EXP_DIR", "experiments"))

    def __post_init__(self) -> None:
        if self.n_in + self.horizons > 4:
            raise ValueError("a TorNet sample has four scans: n_in + horizons must be <= 4")
        if not 0 <= self.val_fraction < 1:
            raise ValueError("val_fraction must be in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    """Defaults, then the JSON file, then keyword overrides. Unknown keys are an error."""
    values: dict[str, Any] = {}
    if path is not None:
        with open(path) as f:
            values.update(json.load(f))
    values.update(overrides)
    known = {f.name for f in dataclasses.fields(Config)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown config keys {sorted(unknown)}")
    if "sample_weights" in values:
        values["sample_weights"] = {**Config().sample_weights, **values["sample_weights"]}
    return Config(**values)
