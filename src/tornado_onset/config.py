"""Run configuration: the defaults every run shares. The four experiments in `run.py` are a few
overrides each."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    exp_name: str = "onset"
    exp_dir: str = "experiments"
    # data: one Hub config per year (the authors' train / test partitions); validation is held
    # out from the training partition by day
    train_years: list[int] = field(default_factory=lambda: [2015])
    test_years: list[int] = field(default_factory=lambda: [2015])
    val_fraction: float = 0.15
    n_in: int = 2  # observed scans per sample (the oldest n_in); the rest are targets
    input_scans: int = 2  # how many of the observed scans the model sees (the newest ones)
    horizons: int = 2
    max_files: int | None = None  # cap on training files, for smoke runs
    # model
    weights: str | None = None  # detector state dict; default: local copy, else the Hub
    encoder_trainable: bool = True
    # how the per-scan feature maps combine before the head:
    #   concat       scans side by side in channels, one head with `horizons` outputs
    #   extrapolate  f(+h) = f_new + (lead_h / dt) * (f_new - f_old), one shared head
    combiner: str = "concat"
    head_init: str = "fresh"  # "pretrained": the published head's weights (extrapolate only)
    # loss: weighted cross-entropy per horizon, the authors' sample weights
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

    def __post_init__(self) -> None:
        if self.n_in + self.horizons > 4:
            raise ValueError("a TorNet sample has four scans: n_in + horizons must be <= 4")
        if not 1 <= self.input_scans <= self.n_in:
            raise ValueError("input_scans must be between 1 and n_in")
        if self.combiner not in ("concat", "extrapolate"):
            raise ValueError("combiner must be 'concat' or 'extrapolate'")
        if self.head_init == "pretrained" and self.combiner != "extrapolate":
            raise ValueError("head_init 'pretrained' needs combiner 'extrapolate'")
        if self.combiner == "extrapolate" and self.input_scans < 2:
            raise ValueError("combiner 'extrapolate' needs two input scans")
