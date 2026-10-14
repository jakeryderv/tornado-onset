"""Evaluation on a test partition, per horizon.

Four probabilities per sample: the head (direct), and the frozen judge reading persistence
(the newest observed scan: the floor), the drawn forecast (health check) and the real future scan
(the ceiling). Each is written as a predictions CSV with the per-scan label pattern (e.g. `0011`)
and `pre_onset` / `ongoing` flags; `summary.json` has AUC / AUCPR per view for all cases and for
the pre-onset subset (no input scan tornadic, last target scan tornadic; nulls kept), pixel MAE of
the drawing against the truth per product, and paired bootstraps against persistence.
`maps_h<k>.npz` holds every 8x15 likelihood map; `features.npz` holds the head's logits and the
max- and mean-pooled bottleneck features per sample, for a later meta model.
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
from tornado_onset.data import OnsetDataset, collate, load_catalog, select_files
from tornado_onset.metrics import paired_difference, views
from tornado_onset.model import TARGET_CHANNELS, TARGET_PRODUCTS, OnsetModel, full_chip
from tornado_onset.tornet import TorNetDetector, load_detector

VIEWS_OF = ("head", "persistence", "forecast", "truth")


def pattern_flags(frame_labels: np.ndarray, n_in: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """pre_onset: no input scan tornadic and the last scan tornadic; ongoing: an input scan
    tornadic; and the label pattern as a string."""
    inputs = frame_labels[:, :n_in].max(axis=1) > 0
    last = frame_labels[:, -1] > 0
    patterns = ["".join(str(int(v)) for v in row) for row in frame_labels]
    return (~inputs) & last, inputs, patterns


def test_loader(cfg: Config, root: str) -> DataLoader:
    files = select_files(load_catalog(root), "test", cfg.test_years)["filename"].tolist()
    ds = OnsetDataset(root, files, cfg.n_in, cfg.horizons, cfg.sample_weights)
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def evaluate(
    cfg: Config,
    model: OnsetModel,
    device: torch.device,
    out_dir: str,
    root: str,
    judge: TorNetDetector | None = None,
) -> dict:
    model.eval()
    judge = judge or load_detector(cfg.weights, device)
    loader = test_loader(cfg, root)
    H = cfg.horizons
    rows: dict[str, list[list[dict]]] = {w: [[] for _ in range(H)] for w in VIEWS_OF}
    maps: dict[str, list[list[np.ndarray]]] = {w: [[] for _ in range(H)] for w in VIEWS_OF}
    mae: dict[str, list[list[np.ndarray]]] = {
        w: [[] for _ in range(H)] for w in ("forecast", "persistence")
    }
    files: list[str] = []
    feat_max, feat_mean, head_logits_all = [], [], []
    for batch in loader:
        x, coords = batch["x"].to(device), batch["coords"].to(device)
        with torch.autocast(device.type, enabled=cfg.amp and device.type == "cuda"):
            out = model(x, coords)
        fields = out["fields"].float()
        head_logits = out["logits"].float().cpu().numpy()
        head_map = out["cls_map"].float().cpu().numpy()
        feats = out["features"].float()
        feat_max.append(feats.amax(dim=(2, 3)).cpu().numpy())
        feat_mean.append(feats.mean(dim=(2, 3)).cpu().numpy())
        head_logits_all.append(head_logits)
        files += [m["filename"] for m in batch["meta"]]
        future = batch["future"].to(device)
        last = x[:, -1]
        pre_onset, ongoing, patterns = pattern_flags(batch["frame_labels"].numpy(), cfg.n_in)
        for h in range(H):
            chips = {
                "persistence": last,
                "forecast": full_chip(fields[:, h], last),
                "truth": future[:, h],
            }
            heats = {w: judge.heat(c, coords) for w, c in chips.items()}  # (B, 1, 8, 15)
            logits = {w: hm.amax(dim=(2, 3)).squeeze(1).cpu().numpy() for w, hm in heats.items()}
            logits["head"] = head_logits[:, h]
            for w, hm in heats.items():
                maps[w][h].append(hm.squeeze(1).cpu().numpy().astype(np.float16))
            maps["head"][h].append(head_map[:, h].astype(np.float16))
            target = future[:, h, TARGET_CHANNELS]
            mae["forecast"][h].append((fields[:, h] - target).abs().mean(dim=(2, 3)).cpu().numpy())
            mae["persistence"][h].append(
                (last[:, TARGET_CHANNELS] - target).abs().mean(dim=(2, 3)).cpu().numpy()
            )
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
                for w in VIEWS_OF:
                    lg = float(logits[w][j])
                    rows[w][h].append({**base, "logit": lg, "prob": 1 / (1 + math.exp(-lg))})
    os.makedirs(out_dir, exist_ok=True)
    summary: dict = {
        "test_years": cfg.test_years,
        "n_in": cfg.n_in,
        "target_products": TARGET_PRODUCTS,
        "horizons": [],
    }
    for h in range(H):
        preds = {}
        for w in VIEWS_OF:
            df = pd.DataFrame(rows[w][h])
            df["year"] = pd.to_datetime(df["time"], unit="s").dt.year
            df.to_csv(os.path.join(out_dir, f"predictions_{w}_h{h}.csv"), index=False)
            preds[w] = df
        scoped = {w: df[df["pre_onset"] | (df["category"] != "TOR")] for w, df in preds.items()}
        err = {w: np.concatenate(v[h]).mean(axis=0) for w, v in mae.items()}  # per channel
        by_product = {
            w: {p: float(e[2 * i : 2 * i + 2].mean()) for i, p in enumerate(TARGET_PRODUCTS)}
            for w, e in err.items()
        }
        entry = {
            "horizon": h,
            "lead_minutes_median": float(preds["head"]["minutes"].median()),
            "n": len(preds["head"]),
            "positives": int(preds["head"]["label"].sum()),
            "pre_onset_cases": int(preds["head"]["pre_onset"].sum()),
            "ongoing_cases": int(preds["head"]["ongoing"].sum()),
            "mae_forecast": float(err["forecast"].mean()),
            "mae_persistence": float(err["persistence"].mean()),
            "mae_by_product": by_product,
            "all": {w: views(df, cfg.bootstrap, cfg.seed) for w, df in preds.items()},
            "pre_onset": {w: views(df, cfg.bootstrap, cfg.seed) for w, df in scoped.items()},
            "pre_onset_paired_vs_persistence": {
                w: paired_difference(scoped["persistence"], scoped[w], cfg.bootstrap, cfg.seed)
                for w in ("head", "forecast", "truth")
            },
        }
        summary["horizons"].append(entry)
        line = " ".join(
            f"{w}={entry['pre_onset'].get(w, {}).get('WC', {}).get('auc', float('nan')):.4f}"
            for w in ("persistence", "forecast", "truth", "head")
        )
        logging.info(
            "h%d lead %.0f min, pre-onset WC AUC: %s; MAE forecast %.4f persistence %.4f",
            h,
            entry["lead_minutes_median"],
            line,
            entry["mae_forecast"],
            entry["mae_persistence"],
        )
        np.savez_compressed(
            os.path.join(out_dir, f"maps_h{h}.npz"),
            filenames=np.array(files),
            **{w: np.concatenate(v[h]) for w, v in maps.items()},  # ty: ignore[invalid-argument-type]
        )
    np.savez_compressed(
        os.path.join(out_dir, "features.npz"),
        filenames=np.array(files),
        head_logits=np.concatenate(head_logits_all),
        features_max=np.concatenate(feat_max).astype(np.float16),
        features_mean=np.concatenate(feat_mean).astype(np.float16),
    )
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def headline(summary: dict) -> dict:
    """Pre-onset WC AUC per view and horizon, and the drawing's MAE, for results.json."""
    return {
        "pre_onset_WC_AUC": {
            f"h{e['horizon']}": {
                w: e["pre_onset"].get(w, {}).get("WC", {}).get("auc")
                for w in ("persistence", "forecast", "truth", "head")
            }
            for e in summary["horizons"]
        },
        "mae": {
            f"h{e['horizon']}": {"forecast": e["mae_forecast"], "persistence": e["mae_persistence"]}
            for e in summary["horizons"]
        },
    }
