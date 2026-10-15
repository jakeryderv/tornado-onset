"""Run configuration: a dataclass with the agreed defaults, overridden by a JSON file.

The model is the fine-tuned TorNet encoder with the authors' head; the loss is the head's
weighted cross-entropy. Training stops early on the validation head AUC.
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
    # data: the Hub dataset's per-year configs (the authors' partitions), or explicit patterns;
    # validation is held out from the training partition by day
    dataset: str = "jakeryderv/tornet"
    train_years: list[int] = field(default_factory=lambda: [2015])
    test_years: list[int] = field(default_factory=lambda: [2015])
    data_files: dict[str, Any] | None = (
        None  # {"train": [patterns], "test": [patterns]} overrides years
    )
    val_fraction: float = 0.15
    n_in: int = 2
    horizons: int = 2
    max_files: int | None = None  # cap on training files, for smoke runs
    # model
    weights: str = DEFAULT_WEIGHTS  # the converted published detector: encoder init + judge
    encoder_trainable: bool = True
    start_filters: int = 48
    # loss
    cls_weight: float = 1.0
    sample_weights: dict[str, float] = field(
        default_factory=lambda: {"wN": 1.0, "w0": 1.0, "w1": 1.0, "w2": 2.0, "wW": 0.5}
    )
    # training
    epochs: int = 8  # cap; early stopping usually ends sooner
    patience: int = 2  # stop after this many epochs without a validation head-AUC gain (0: off)
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
    """Defaults, then the JSON file, then keyword overrides. Unknown keys are an error; keys
    starting with `_` (notes) are ignored."""
    values: dict[str, Any] = {}
    if path is not None:
        with open(path) as f:
            values.update({k: v for k, v in json.load(f).items() if not k.startswith("_")})
    values.update(overrides)
    known = {f.name for f in dataclasses.fields(Config)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown config keys {sorted(unknown)}")
    if "sample_weights" in values:
        values["sample_weights"] = {**Config().sample_weights, **values["sample_weights"]}
    return Config(**values)
