"""Reading TorNet samples and turning them into the detector's tensors.

A TorNet sample is a netCDF file with four radar scans about five minutes apart, each
(azimuth=120, range=240, tilt=2) for six products, plus a range-folded mask for the two Doppler
products, a tornado label per scan and event metadata. The published detector consumes, per scan,
the six products normalised to roughly [-1, 1] with empty gates set to ``BACKGROUND``, the two
range-folded channels appended (14 channels), and two CoordConv coordinate channels (range and
inverse range). Everything here is channels-first.
"""

from __future__ import annotations

import datetime
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset

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
CATEGORIES = {0: "TOR", 1: "NUL", 2: "WRN"}
CHANNEL_INDEX = {v: [N_SWEEPS * i, N_SWEEPS * i + 1] for i, v in enumerate(ALL_VARIABLES)}
_EPOCH = datetime.datetime(1970, 1, 1)


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
    """One TorNet file, every scan: fields (T, 14, A, R) in the detector's representation,
    coords (2, A, R), one label per scan, times in epoch seconds, and event metadata."""

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


def _to_epoch(s: str) -> int:
    return int((datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S") - _EPOCH).total_seconds())


def read_sample(path: str) -> Sample:
    with xr.open_dataset(path) as ds:
        frames = {v: ds[v].values for v in ALL_VARIABLES}  # (T, A, R, tilt)
        rfm = ds["range_folded_mask"].values.astype(np.float32)
        labels = ds["frame_labels"].values.astype(np.float32)
        ef = int(ds.attrs["ef_number"])
        labels[-1] = float(ef >= 0)  # the authors' v1.0 consistency fix
        times = (ds["time"].values.astype(np.int64) / 1e9).astype(np.int64)
        az = ds["azimuth_limits"].values
        rng = ds["range_limits"].values
        category = ds.attrs["category"]
        event_id = int(ds.attrs["event_id"])
        if ef >= 0 and "tornado_start_time" in ds.attrs:
            start, end = (
                _to_epoch(ds.attrs["tornado_start_time"]),
                _to_epoch(ds.attrs["tornado_end_time"]),
            )
        else:
            start = end = 0
    fields = np.concatenate([normalize_fields(frames), rfm], axis=-1)  # (T, A, R, 14)
    return Sample(
        fields=np.ascontiguousarray(fields.transpose(0, 3, 1, 2)),
        coords=coordinates(
            (float(az[0]), float(az[1])), (float(rng[0]), float(rng[1])), fields.shape[1:3]
        ),
        labels=labels,
        times=times,
        category=category,
        ef_number=ef,
        event_id=event_id,
        tornado_start=start,
        tornado_end=end,
        rng_center_km=float((rng[0] + rng[1]) / 2e3),
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


def data_root() -> str:
    try:
        return os.environ["TORNET_ROOT"]
    except KeyError as e:
        raise RuntimeError("set TORNET_ROOT to the directory holding catalog.csv") from e


def load_catalog(root: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(root, "catalog.csv"), parse_dates=["start_time", "end_time"])
    df["year"] = df["start_time"].dt.year
    df["day"] = df["start_time"].dt.strftime("%Y-%m-%d")
    return df


def select_files(catalog: pd.DataFrame, partition: str, years: list[int]) -> pd.DataFrame:
    """Rows of one partition ("train" or "test") and set of years, catalog order."""
    return catalog[(catalog["type"] == partition) & catalog["year"].isin(years)].reset_index(
        drop=True
    )


def split_by_day(
    rows: pd.DataFrame, val_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out whole days for validation so consecutive scans of one storm never straddle the
    split. Returns (train_rows, val_rows)."""
    days = np.array(sorted(rows["day"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(days)
    n_val = round(len(days) * val_fraction)
    val_days = set(days[:n_val])
    is_val = rows["day"].isin(val_days)
    return rows[~is_val].reset_index(drop=True), rows[is_val].reset_index(drop=True)


class OnsetDataset(Dataset):
    """The oldest `n_in` scans in, the next `horizons` scans out.

    Per item:
        x            (n_in, 14, A, R)      input scans
        coords       (2, A, R)
        future       (horizons, 14, A, R)  target scans (regression targets are a subset)
        labels       (horizons,)           tornado label of each target scan
        frame_labels (n_in + horizons,)    every scan's label, for scoping the evaluation
        minutes      (horizons,)           lead from the last input scan, real timestamps
        weight       ()                    the authors' sample weight
        meta         filename, category, ef_number, event_id, time (last input), times
    """

    def __init__(
        self,
        root: str,
        files: list[str],
        n_in: int = 2,
        horizons: int = 2,
        weights: dict[str, float] | None = None,
    ):
        if n_in + horizons > 4:
            raise ValueError("a TorNet sample has four scans")
        self.root, self.files = root, list(files)
        self.n_in, self.horizons = n_in, horizons
        self.weights = weights or {}

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        n, h = self.n_in, self.horizons
        s = read_sample(os.path.join(self.root, self.files[idx]))
        t = s.times
        return {
            "x": torch.from_numpy(s.fields[:n]),
            "coords": torch.from_numpy(s.coords),
            "future": torch.from_numpy(s.fields[n : n + h]),
            "labels": torch.from_numpy(s.labels[n : n + h]),
            "frame_labels": torch.from_numpy(s.labels[: n + h]),
            "minutes": torch.from_numpy(((t[n : n + h] - t[n - 1]) / 60).astype(np.float32)),
            "weight": torch.tensor(sample_weight(s.category, s.ef_number, self.weights)),
            "meta": {
                "filename": self.files[idx],
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
