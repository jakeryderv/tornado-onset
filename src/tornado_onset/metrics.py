"""Scoring helpers: the paper's three test views, AUC / AUCPR with bootstrap intervals, the
best-CSI threshold, and paired bootstrap differences over the same files."""

from __future__ import annotations

import numpy as np
import pandas as pd

# which null categories are kept alongside confirmed tornadoes
VIEWS = {
    "NC": ("all nulls vs confirmed", {"TOR", "NUL", "WRN"}),
    "RC": ("random nulls vs confirmed", {"TOR", "NUL"}),
    "WC": ("warnings vs confirmed", {"TOR", "WRN"}),
}


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


def best_threshold(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Threshold maximising CSI = TP / (TP + FP + FN), with the metrics at that threshold."""
    y = np.asarray(y, bool)
    best = {"threshold": 0.5, "csi": -1.0, "precision": 0.0, "recall": 0.0}
    for t in np.unique(np.round(p, 4)):
        pred = p >= t
        tp = int((pred & y).sum())
        fp = int((pred & ~y).sum())
        fn = int((~pred & y).sum())
        csi = tp / (tp + fp + fn) if tp + fp + fn else 0.0
        if csi > best["csi"]:
            best = {
                "threshold": float(t),
                "csi": float(csi),
                "precision": tp / (tp + fp) if tp + fp else 0.0,
                "recall": tp / (tp + fn) if tp + fn else 0.0,
            }
    return best


def view_metrics(df: pd.DataFrame, n_boot: int, rng: np.random.Generator) -> dict:
    y, p = df["label"].to_numpy(bool), df["prob"].to_numpy(float)
    out: dict = {
        "n": len(df),
        "n_confirmed": int(y.sum()),
        "auc": roc_auc(y, p),
        "aucpr": average_precision(y, p),
        **{f"best_{k}": v for k, v in best_threshold(y, p).items()},
    }
    if n_boot:
        aucs, aps = [], []
        for _ in range(n_boot):
            idx = rng.integers(0, len(df), len(df))
            aucs.append(roc_auc(y[idx], p[idx]))
            aps.append(average_precision(y[idx], p[idx]))
        out["auc_ci95"] = [float(np.nanpercentile(aucs, 2.5)), float(np.nanpercentile(aucs, 97.5))]
        out["aucpr_ci95"] = [float(np.nanpercentile(aps, 2.5)), float(np.nanpercentile(aps, 97.5))]
    return out


def views(df: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """`view_metrics` per view that has both classes."""
    rng = np.random.default_rng(seed)
    out = {}
    for key, (_desc, cats) in VIEWS.items():
        sub = df[df["category"].isin(cats)]
        if 0 < sub["label"].sum() < len(sub):
            out[key] = view_metrics(sub, n_boot, rng)
    return out


def paired_difference(a: pd.DataFrame, b: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """Bootstrap the AUC and AUCPR difference (b - a) over the same files, per view."""
    m = a.merge(b, on="filename", suffixes=("_a", "_b"))
    if m.empty:
        return {"error": "no filenames in common"}
    rng = np.random.default_rng(seed)
    out: dict = {"n_common": len(m)}
    for key, (_desc, cats) in VIEWS.items():
        sub = m[m["category_a"].isin(cats)]
        y = sub["label_a"].to_numpy(bool)
        pa, pb = sub["prob_a"].to_numpy(float), sub["prob_b"].to_numpy(float)
        boots_auc, boots_ap = [], []
        for _ in range(n_boot):
            idx = rng.integers(0, len(sub), len(sub))
            boots_auc.append(roc_auc(y[idx], pb[idx]) - roc_auc(y[idx], pa[idx]))
            boots_ap.append(average_precision(y[idx], pb[idx]) - average_precision(y[idx], pa[idx]))
        out[key] = {
            "delta_auc": roc_auc(y, pb) - roc_auc(y, pa),
            "delta_auc_ci95": _ci(boots_auc) if n_boot else None,
            "delta_aucpr": average_precision(y, pb) - average_precision(y, pa),
            "delta_aucpr_ci95": _ci(boots_ap) if n_boot else None,
        }
    return out


def _ci(values: list[float]) -> list[float]:
    return [float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))]
