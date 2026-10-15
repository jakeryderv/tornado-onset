"""TorNet samples from the Hugging Face Hub, turned into the detector's tensors.

The dataset `jakeryderv/tornet` holds every TorNet v1.1 sample as Parquet: four radar scans
about five minutes apart, each (azimuth=120, range=240, tilt=2) for six products in physical
units (NaN where empty), the range-folded mask, a label per scan, scan times and event metadata.
One config per year (`"2015"` has `train` and `test` splits), or explicit `data_files` patterns
for any subset. Shards are cached and memory-mapped by `datasets`, so a year downloads once.

The published detector consumes, per scan, the six products normalised to roughly [-1, 1] with
empty gates set to ``BACKGROUND``, the two range-folded channels appended (14 channels), and two
CoordConv coordinate channels (range and inverse range). Everything here is channels-first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

HF_DATASET = "jakeryderv/tornet"
ALL_VARIABLES = ["DBZ", "VEL", "KDP", "RHOHV", "ZDR", "WIDTH"]
CHANNEL_MIN_MAX = {
    "DBZ": (-20.0, 60.0),
    "VEL": (-60.0, 60.0),
    "KDP": (-2.0, 5.0),
    "RHOHV": (0.2, 1.04),
    "ZDR": (-1.0, 8.0),
    "WIDTH": (0.0, 9.0),
}
BACKGROUND = -3.0  # the detector's fill value for empty gates, in normalised units
N_SWEEPS = 2
N_RADAR_CHANNELS = N_SWEEPS * len(ALL_VARIABLES)  # 12
N_INPUT_CHANNELS = N_RADAR_CHANNELS + N_SWEEPS  # + 2 range-folded channels = 14
CHANNEL_INDEX = {v: [N_SWEEPS * i, N_SWEEPS * i + 1] for i, v in enumerate(ALL_VARIABLES)}


def channel_indices(products: list[str]) -> list[int]:
    return [i for p in products for i in CHANNEL_INDEX[p]]


def channel_scale(name: str) -> tuple[float, float]:
    """(offset, scale) such that normalised = (raw - offset) / scale, as the detector does."""
    lo, hi = CHANNEL_MIN_MAX[name]
    return (lo + hi) / 2, (hi - lo) / 2


def normalize_fields(frames: dict[str, np.ndarray], background: float = BACKGROUND) -> np.ndarray:
    """{product: (..., A, R, tilt)} -> (..., A, R, 12), normalised and NaN-filled, in the
    detector's channel order (products in ALL_VARIABLES order, both tilts of each)."""
    out = []
    for v in ALL_VARIABLES:
        offset, scale = channel_scale(v)
        x = (frames[v].astype(np.float32) - offset) / scale
        out.append(np.where(np.isnan(x), background, x))
    return np.concatenate(out, axis=-1)


def denormalize_fields(fields: np.ndarray) -> dict[str, np.ndarray]:
    """Inverse of `normalize_fields` (channels last) for viewing in physical units; background
    gates stay at the fill value's physical equivalent."""
    out = {}
    for i, v in enumerate(ALL_VARIABLES):
        offset, scale = channel_scale(v)
        out[v] = fields[..., N_SWEEPS * i : N_SWEEPS * (i + 1)] * scale + offset
    return out


def coordinates(
    az_limits: tuple[float, float],
    rng_limits: tuple[float, float],
    shape: tuple[int, int] = (120, 240),
    min_range_m: float = 2125.0,
) -> np.ndarray:
    """The authors' CoordConv coordinates, (2, A, R): scaled range and its inverse."""
    scale = 1e-5
    rng_lower = (rng_limits[0] + 250) * scale
    rng_upper = (rng_limits[1] - 250) * scale
    az_lower = (90 - az_limits[0]) * np.pi / 180
    az_upper = (90 - az_limits[1]) * np.pi / 180
    az = np.linspace(az_lower, az_upper, shape[0])
    rg = np.linspace(rng_lower, rng_upper, shape[1])
    r, _ = np.meshgrid(rg, az, indexing="xy")  # (A, R)
    r = np.where(r >= min_range_m * scale, r, min_range_m * scale)
    return np.stack((r, 1 / r), axis=0).astype(np.float32)


@dataclass
class Sample:
    """One TorNet sample, every scan: fields (T, 14, A, R) in the detector's representation,
    coords (2, A, R), one label per scan, times in epoch seconds, and event metadata."""

    filename: str
    fields: np.ndarray
    coords: np.ndarray
    labels: np.ndarray
    times: np.ndarray
    category: str
    ef_number: int
    event_id: int
    tornado_start: int
    tornado_end: int
    rng_center_km: float


def sample_from_row(row: dict[str, Any]) -> Sample:
    """A row of the Hub dataset (numpy format) -> the detector's tensors."""
    frames = {v: np.asarray(row[v], dtype=np.float32) for v in ALL_VARIABLES}  # (T, A, R, tilt)
    rfm = np.asarray(row["range_folded_mask"], dtype=np.float32)
    fields = np.concatenate([normalize_fields(frames), rfm], axis=-1)  # (T, A, R, 14)
    ef = int(row["ef_number"])
    labels = np.asarray(row["frame_labels"], dtype=np.float32).copy()
    labels[-1] = float(ef >= 0)  # the authors' v1.0 consistency fix
    az = (float(row["az_lower"]), float(row["az_upper"]))
    rng = (float(row["rng_lower"]), float(row["rng_upper"]))
    return Sample(
        filename=str(row["filename"]),
        fields=np.ascontiguousarray(fields.transpose(0, 3, 1, 2)),
        coords=coordinates(az, rng, fields.shape[1:3]),
        labels=labels,
        times=np.asarray(row["time"], dtype=np.int64),
        category=str(row["category"]),
        ef_number=ef,
        event_id=int(row["event_id"]),
        tornado_start=int(row["tornado_start_time"]),
        tornado_end=int(row["tornado_end_time"]),
        rng_center_km=(rng[0] + rng[1]) / 2e3,
    )


def sample_weight(category: str, ef_number: int, w: dict[str, float]) -> float:
    """The authors' rule: warnings wW, nulls wN, EF0 w0, EF1 w1, EF2+ w2."""
    if category == "WRN":
        return float(w.get("wW", 0.5))
    if ef_number < 0:
        return float(w.get("wN", 1.0))
    if ef_number == 0:
        return float(w.get("w0", 1.0))
    if ef_number == 1:
        return float(w.get("w1", 1.0))
    return float(w.get("w2", 2.0))


def load_split(
    partition: str,
    years: list[int],
    dataset: str = HF_DATASET,
    data_files: dict[str, Any] | None = None,
):
    """A `datasets.Dataset` for one partition: the per-year configs of `years`, or, when
    `data_files` is given, those explicit patterns (a fold, a shard for a test)."""
    from datasets import concatenate_datasets, load_dataset

    if data_files:
        return load_dataset(dataset, data_files=data_files, split=partition)
    parts = [load_dataset(dataset, str(y), split=partition) for y in years]
    return parts[0] if len(parts) == 1 else concatenate_datasets(parts)


def days_of(ds, indices: np.ndarray | None = None) -> np.ndarray:
    """The UTC date of each sample's first scan, as 'YYYY-MM-DD' strings."""
    times = np.asarray([t[0] for t in ds["time"]], dtype="int64")
    if indices is not None:
        times = times[indices]
    return np.datetime_as_string(times.astype("datetime64[s]"), unit="D")


def split_by_day(
    ds, indices: np.ndarray, val_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out whole days for validation so consecutive scans of one storm never straddle the
    split. Returns (train_indices, val_indices) into `ds`."""
    days = days_of(ds, indices)
    unique = np.array(sorted(set(days)))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    val_days = set(unique[: round(len(unique) * val_fraction)])
    is_val = np.array([d in val_days for d in days])
    return indices[~is_val], indices[is_val]


class OnsetDataset(Dataset):
    """The oldest `n_in` scans in, the next `horizons` scans out, over a Hub dataset.

    Per item:
        x            (n_in, 14, A, R)      input scans
        coords       (2, A, R)
        future       (horizons, 14, A, R)  target scans
        labels       (horizons,)           tornado label of each target scan
        frame_labels (n_in + horizons,)    every scan's label, for scoping the evaluation
        minutes      (horizons,)           lead from the last input scan, real timestamps
        dt_in        ()                    minutes between the last two input scans
        weight       ()                    the authors' sample weight
        meta         filename, category, ef_number, event_id, time (last input), times
    """

    def __init__(
        self,
        ds,
        n_in: int = 2,
        horizons: int = 2,
        weights: dict[str, float] | None = None,
        indices: np.ndarray | list[int] | None = None,
    ):
        if n_in + horizons > 4:
            raise ValueError("a TorNet sample has four scans")
        self.ds = ds.with_format("numpy")
        self.indices = np.arange(len(ds)) if indices is None else np.asarray(indices)
        self.n_in, self.horizons = n_in, horizons
        self.weights = weights or {}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        n, h = self.n_in, self.horizons
        s = sample_from_row(self.ds[int(self.indices[idx])])
        t = s.times
        return {
            "x": torch.from_numpy(s.fields[:n]),
            "coords": torch.from_numpy(s.coords),
            "future": torch.from_numpy(s.fields[n : n + h]),
            "labels": torch.from_numpy(s.labels[n : n + h]),
            "frame_labels": torch.from_numpy(s.labels[: n + h]),
            "minutes": torch.from_numpy(((t[n : n + h] - t[n - 1]) / 60).astype(np.float32)),
            "dt_in": torch.tensor((t[n - 1] - t[max(n - 2, 0)]) / 60, dtype=torch.float32),
            "weight": torch.tensor(sample_weight(s.category, s.ef_number, self.weights)),
            "meta": {
                "filename": s.filename,
                "category": s.category,
                "ef_number": s.ef_number,
                "event_id": s.event_id,
                "time": int(t[n - 1]),
                "times": t[n : n + h].tolist(),
                "rng_center_km": s.rng_center_km,
            },
        }


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {k: torch.stack([b[k] for b in batch]) for k in batch[0] if k != "meta"}
    out["meta"] = [b["meta"] for b in batch]
    return out
