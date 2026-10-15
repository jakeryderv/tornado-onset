"""Run configuration: a dataclass with the defaults, and the four experiments as overrides.

The model is the fine-tuned TorNet encoder with the authors' head; the loss is the head's
weighted cross-entropy; training stops early on the validation head AUC. `onset-train <name>`
runs one of `EXPERIMENTS` (or `all`), with `--set key=value` for ad hoc changes.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from typing import Any

WEIGHTS_REPO = "jakeryderv/tornet-detector-torch"  # the published detector as a torch state dict
WEIGHTS_FILE = "tornet_detector_v1.pt"
DEFAULT_WEIGHTS = f"weights/{WEIGHTS_FILE}"  # a local conversion, if one exists, wins


def resolve_weights(path: str | None = None) -> str:
    """A local state-dict path if it exists, else the copy on the Hub (downloaded once)."""
    import os

    path = path or DEFAULT_WEIGHTS
    if os.path.exists(path):
        return path
    from huggingface_hub import hf_hub_download

    return hf_hub_download(WEIGHTS_REPO, WEIGHTS_FILE)


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
    n_in: int = 2  # observed scans per sample (the oldest n_in); the rest are targets
    input_scans: int = 2  # how many of the observed scans the model sees (the newest ones)
    horizons: int = 2
    max_files: int | None = None  # cap on training files, for smoke runs
    # model
    weights: str = (
        DEFAULT_WEIGHTS  # encoder init + judge; fetched from the Hub when missing locally
    )
    encoder_trainable: bool = True
    start_filters: int = 48
    # how the per-scan feature maps combine before the head:
    #   concat       scans side by side in channels, one head with `horizons` outputs
    #   extrapolate  f(+h) = f_new + (lead_h / dt) * (f_new - f_old), one shared 384-channel head
    combiner: str = "concat"
    head_init: str = "fresh"  # "pretrained": the published head's weights (extrapolate only)
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
    workers: int = 4  # each loader worker holds one decoded shard, about 1 GB
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
        if self.combiner not in ("concat", "extrapolate"):
            raise ValueError("combiner must be 'concat' or 'extrapolate'")
        if self.head_init not in ("fresh", "pretrained"):
            raise ValueError("head_init must be 'fresh' or 'pretrained'")
        if not 1 <= self.input_scans <= self.n_in:
            raise ValueError("input_scans must be between 1 and n_in")
        if self.combiner == "concat" and self.head_init == "pretrained":
            raise ValueError("head_init 'pretrained' needs combiner 'extrapolate'")
        if self.combiner == "extrapolate" and self.input_scans < 2:
            raise ValueError("combiner 'extrapolate' needs two input scans")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


EXPERIMENTS: dict[str, dict[str, Any]] = {
    # the model: fine-tuned encoder on the two observed scans, the authors' head, early stopping
    "baseline": {},
    # does fine-tuning on the onset target matter? the published encoder frozen, only the head trains
    "frozen_encoder": {"encoder_trainable": False},
    # does the second scan buy anything? the model sees only the newest observed scan; same
    # targets and the same pre-onset test set
    "one_scan": {"input_scans": 1},
    # a fixed time rule instead of a learned one: f(+h) = f_new + lead/dt * (f_new - f_old) with
    # the real scan timing, one shared head started from the published head
    "extrapolate": {"combiner": "extrapolate", "head_init": "pretrained"},
}


def experiment(name: str, **overrides: Any) -> Config:
    """The named experiment's config (its `exp_name` is the name), with keyword overrides."""
    if name not in EXPERIMENTS:
        raise ValueError(f"unknown experiment {name!r}; one of {sorted(EXPERIMENTS)}")
    return config_from_dict({**EXPERIMENTS[name], "exp_name": name, **overrides})


def config_from_dict(values: dict[str, Any]) -> Config:
    """A Config from a dict (a run's params.json, or overrides). Unknown keys are an error."""
    values = {k: v for k, v in values.items() if not k.startswith("_")}
    known = {f.name for f in dataclasses.fields(Config)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown config keys {sorted(unknown)}")
    if "sample_weights" in values:
        values["sample_weights"] = {**Config().sample_weights, **values["sample_weights"]}
    return Config(**values)
