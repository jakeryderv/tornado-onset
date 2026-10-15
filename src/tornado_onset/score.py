"""Evaluation on the test partition, per horizon, and the metrics behind it.

Three probabilities per sample: the head, and the frozen judge reading persistence (the newest
observed scan: the floor) and the real future scan (the ceiling). Each is written as a
predictions CSV with the per-scan label pattern (e.g. `0011`) and `pre_onset` / `ongoing` flags.
`summary.json` has AUC / AUCPR per view (the paper's NC, RC, WC) for all cases and for the
pre-onset subset (no input scan tornadic, last target scan tornadic; nulls kept), with bootstrap
intervals and a paired bootstrap of head and ceiling against persistence.
"""

from __future__ import annotations

import json
import logging
import math
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from tornado_onset.config import Config
from tornado_onset.data import OnsetDataset, collate, shard_paths
from tornado_onset.model import OnsetModel
from tornado_onset.tornet import TorNetDetector

SOURCES = ("head", "persistence", "truth")
# which null categories are kept alongside confirmed tornadoes
VIEWS = {
    "NC": ("all nulls vs confirmed", {"TOR", "NUL", "WRN"}),
    "RC": ("random nulls vs confirmed", {"TOR", "NUL"}),
    "WC": ("warnings vs confirmed", {"TOR", "WRN"}),
}


# --- metrics ---------------------------------------------------------------------------------


def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    """Mann-Whitney AUC with average ranks for ties; nan if only one class is present."""
    y = np.asarray(y, bool)
    n_pos, n_neg = y.sum(), (~y).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(p).rank(method="average").to_numpy()
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, bool)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-p, kind="stable")
    hits = y[order]
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float(precision[hits].mean())


def _ci(values: list[float]) -> list[float]:
    return [float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))]


def views(df: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """AUC and AUCPR with bootstrap intervals, per view that has both classes."""
    rng = np.random.default_rng(seed)
    out = {}
    for key, (_desc, cats) in VIEWS.items():
        sub = df[df["category"].isin(cats)]
        y, p = sub["label"].to_numpy(bool), sub["prob"].to_numpy(float)
        if not 0 < y.sum() < len(y):
            continue
        entry: dict = {
            "n": len(sub),
            "n_confirmed": int(y.sum()),
            "auc": roc_auc(y, p),
            "aucpr": average_precision(y, p),
        }
        if n_boot:
            idx = rng.integers(0, len(sub), (n_boot, len(sub)))
            entry["auc_ci95"] = _ci([roc_auc(y[i], p[i]) for i in idx])
            entry["aucpr_ci95"] = _ci([average_precision(y[i], p[i]) for i in idx])
        out[key] = entry
    return out


def paired_difference(a: pd.DataFrame, b: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """Bootstrap the AUC and AUCPR difference (b - a) over the same files, per view."""
    m = a.merge(b, on="filename", suffixes=("_a", "_b"))
    rng = np.random.default_rng(seed)
    out: dict = {"n_common": len(m)}
    for key, (_desc, cats) in VIEWS.items():
        sub = m[m["category_a"].isin(cats)]
        y = sub["label_a"].to_numpy(bool)
        pa, pb = sub["prob_a"].to_numpy(float), sub["prob_b"].to_numpy(float)
        idx = rng.integers(0, len(sub), (n_boot, len(sub)))
        out[key] = {
            "delta_auc": roc_auc(y, pb) - roc_auc(y, pa),
            "delta_auc_ci95": _ci([roc_auc(y[i], pb[i]) - roc_auc(y[i], pa[i]) for i in idx]),
            "delta_aucpr": average_precision(y, pb) - average_precision(y, pa),
            "delta_aucpr_ci95": _ci(
                [average_precision(y[i], pb[i]) - average_precision(y[i], pa[i]) for i in idx]
            ),
        }
    return out


# --- evaluation ------------------------------------------------------------------------------


def pattern_flags(frame_labels: np.ndarray, n_in: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """pre_onset: no input scan tornadic and the last scan tornadic; ongoing: an input scan
    tornadic; and the label pattern as a string."""
    inputs = frame_labels[:, :n_in].max(axis=1) > 0
    last = frame_labels[:, -1] > 0
    patterns = ["".join(str(int(v)) for v in row) for row in frame_labels]
    return (~inputs) & last, inputs, patterns


def test_loader(cfg: Config) -> DataLoader:
    paths = shard_paths("test", cfg.test_years)
    return DataLoader(
        OnsetDataset(
            paths, cfg.n_in, cfg.horizons, cfg.sample_weights, input_scans=cfg.input_scans
        ),
        batch_size=cfg.batch_size,
        num_workers=cfg.workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def evaluate(
    cfg: Config, model: OnsetModel, judge: TorNetDetector, device: torch.device, out_dir: str
) -> dict:
    model.eval()
    H = cfg.horizons
    rows: dict[str, list[list[dict]]] = {w: [[] for _ in range(H)] for w in SOURCES}
    for batch in test_loader(cfg):
        x, coords = batch["x"].to(device), batch["coords"].to(device)
        with torch.autocast(device.type, enabled=cfg.amp and device.type == "cuda"):
            head_logits = model(x, coords, batch["minutes"].to(device), batch["dt_in"].to(device))
        head_logits = head_logits.float().cpu().numpy()
        future = batch["future"].to(device)
        pre_onset, ongoing, patterns = pattern_flags(batch["frame_labels"].numpy(), cfg.n_in)
        for h in range(H):
            logits = {
                "head": head_logits[:, h],
                "persistence": judge(x[:, -1], coords).squeeze(1).cpu().numpy(),
                "truth": judge(future[:, h], coords).squeeze(1).cpu().numpy(),
            }
            for j, meta in enumerate(batch["meta"]):
                base = {
                    "filename": meta["filename"],
                    "label": int(batch["labels"][j, h]),
                    "category": meta["category"],
                    "ef_number": meta["ef_number"],
                    "event_id": meta["event_id"],
                    "time": int(meta["times"][h]),
                    "minutes": float(batch["minutes"][j, h]),
                    "pattern": patterns[j],
                    "pre_onset": bool(pre_onset[j]),
                    "ongoing": bool(ongoing[j]),
                }
                for w in SOURCES:
                    lg = float(logits[w][j])
                    rows[w][h].append({**base, "logit": lg, "prob": 1 / (1 + math.exp(-lg))})
    os.makedirs(out_dir, exist_ok=True)
    summary: dict = {"test_years": cfg.test_years, "n_in": cfg.n_in, "horizons": []}
    for h in range(H):
        preds = {w: pd.DataFrame(rows[w][h]) for w in SOURCES}
        for w, df in preds.items():
            df.to_csv(os.path.join(out_dir, f"predictions_{w}_h{h}.csv"), index=False)
        scoped = {w: df[df["pre_onset"] | (df["category"] != "TOR")] for w, df in preds.items()}
        entry = {
            "horizon": h,
            "lead_minutes_median": float(preds["head"]["minutes"].median()),
            "n": len(preds["head"]),
            "positives": int(preds["head"]["label"].sum()),
            "pre_onset_cases": int(preds["head"]["pre_onset"].sum()),
            "ongoing_cases": int(preds["head"]["ongoing"].sum()),
            "all": {w: views(df, cfg.bootstrap, cfg.seed) for w, df in preds.items()},
            "pre_onset": {w: views(df, cfg.bootstrap, cfg.seed) for w, df in scoped.items()},
            "pre_onset_paired_vs_persistence": {
                w: paired_difference(scoped["persistence"], scoped[w], cfg.bootstrap, cfg.seed)
                for w in ("head", "truth")
            },
        }
        summary["horizons"].append(entry)
        line = " ".join(
            f"{w}={entry['pre_onset'].get(w, {}).get('WC', {}).get('auc', float('nan')):.4f}"
            for w in ("persistence", "truth", "head")
        )
        logging.info(
            "h%d lead %.0f min, pre-onset WC AUC: %s", h, entry["lead_minutes_median"], line
        )
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def headline(summary: dict) -> dict:
    """Pre-onset WC AUC per source and horizon, for results.json."""
    return {
        "pre_onset_WC_AUC": {
            f"h{e['horizon']}": {
                w: e["pre_onset"].get(w, {}).get("WC", {}).get("auc")
                for w in ("persistence", "truth", "head")
            }
            for e in summary["horizons"]
        }
    }
