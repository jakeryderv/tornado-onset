"""Training: AdamW with a smaller learning rate on the fine-tuned encoder, cosine schedule,
mixed precision, gradient clipping; the checkpoint with the best mean validation head AUC is
kept (the head is the product) and scored."""

from __future__ import annotations

import datetime
import json
import logging
import os
import subprocess
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from tornado_onset.config import Config
from tornado_onset.data import OnsetDataset, collate, load_catalog, select_files, split_by_day
from tornado_onset.metrics import roc_auc
from tornado_onset.model import Losses, OnsetModel, build_model
from tornado_onset.score import evaluate, headline
from tornado_onset.tornet import load_detector


def make_loaders(cfg: Config, root: str) -> tuple[DataLoader, DataLoader | None, dict[str, int]]:
    rows = select_files(load_catalog(root), "train", cfg.train_years)
    if cfg.max_files:
        rows = (
            rows.sample(frac=1, random_state=cfg.seed).iloc[: cfg.max_files].reset_index(drop=True)
        )
    train_rows, val_rows = split_by_day(rows, cfg.val_fraction, cfg.seed)

    def loader(r: pd.DataFrame, shuffle: bool) -> DataLoader:
        ds = OnsetDataset(root, r["filename"].tolist(), cfg.n_in, cfg.horizons, cfg.sample_weights)
        return DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            num_workers=cfg.workers,
            collate_fn=collate,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=cfg.workers > 0,
        )

    counts = {
        "train_files": len(train_rows),
        "train_days": int(train_rows["day"].nunique()),
        "val_files": len(val_rows),
        "val_days": int(val_rows["day"].nunique()),
    }
    return loader(train_rows, True), loader(val_rows, False) if len(val_rows) else None, counts


def run_epoch(
    model: OnsetModel,
    loader: DataLoader,
    losses: Losses,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    cfg = losses.cfg
    model.train(training)
    sums: dict[str, float] = {}
    n = 0
    logits_all, labels_all = [], []
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        coords = batch["coords"].to(device, non_blocking=True)
        with (
            torch.set_grad_enabled(training),
            torch.autocast(device.type, enabled=cfg.amp and device.type == "cuda"),
        ):
            out = model(x, coords)
            out = {k: v.float() for k, v in out.items()}
            total, parts = losses(out, batch)
        if training:
            assert optimizer is not None and scaler is not None
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            if cfg.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        bs = x.shape[0]
        for k, v in {"loss": float(total), **parts}.items():
            sums[k] = sums.get(k, 0.0) + v * bs
        n += bs
        logits_all.append(out["logits"].detach().cpu().numpy())
        labels_all.append(batch["labels"].numpy())
    metrics = {k: v / max(n, 1) for k, v in sums.items()}
    logits, labels = np.concatenate(logits_all), np.concatenate(labels_all)
    for h in range(labels.shape[1]):
        metrics[f"AUC_h{h}"] = roc_auc(labels[:, h].astype(bool), logits[:, h])
    return metrics


def fit(
    cfg: Config, model: OnsetModel, losses: Losses, device: torch.device, expdir: str, root: str
) -> tuple[list[dict[str, Any]], str, str]:
    """Train; returns (history, best checkpoint path, last checkpoint path)."""
    train_loader, val_loader, counts = make_loaders(cfg, root)
    logging.info("files: %s", counts)
    enc = [p for p in model.encoder.parameters() if p.requires_grad]
    enc_ids = {id(p) for p in model.encoder.parameters()}
    rest = [p for p in model.parameters() if id(p) not in enc_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": enc, "lr": cfg.learning_rate * cfg.encoder_lr_scale},
            {"params": rest, "lr": cfg.learning_rate},
        ],
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(cfg.epochs, 1))
    scaler = torch.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")
    ckpt_dir = os.path.join(expdir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path, last_path = os.path.join(ckpt_dir, "best.pt"), os.path.join(ckpt_dir, "last.pt")
    history: list[dict[str, Any]] = []
    best = -np.inf
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        row: dict[str, Any] = {"epoch": epoch}
        row.update(
            {
                f"train_{k}": v
                for k, v in run_epoch(
                    model, train_loader, losses, device, optimizer, scaler
                ).items()
            }
        )
        if val_loader is not None:
            va = run_epoch(model, val_loader, losses, device)
            row.update({f"val_{k}": v for k, v in va.items()})
            score = float(np.nanmean([v for k, v in va.items() if k.startswith("AUC_h")]))
            if score > best:
                best = score
                row["best"] = True
                torch.save(model.state_dict(), best_path)
        scheduler.step()
        row["seconds"] = time.time() - t0
        history.append(row)
        pd.DataFrame(history).to_csv(os.path.join(expdir, "history.csv"), index=False)
        logging.info(
            " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items())
        )
    torch.save(model.state_dict(), last_path)
    if not os.path.exists(best_path):
        best_path = last_path
    return history, best_path, last_path


def git_revision() -> tuple[str | None, bool | None]:
    here = os.path.dirname(os.path.abspath(__file__))

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", here, *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return git("rev-parse", "--short", "HEAD"), bool(
            git("status", "--porcelain", "--untracked-files=no")
        )
    except (subprocess.CalledProcessError, OSError):
        return None, None


def make_expdir(cfg: Config) -> str:
    stamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    path = os.path.join(cfg.exp_dir, f"{cfg.exp_name}_{stamp}")
    os.makedirs(path, exist_ok=False)
    return path


def run(cfg: Config, root: str) -> dict[str, Any]:
    """Train, keep the best checkpoint, score it; everything under a fresh run directory."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    expdir = make_expdir(cfg)
    logging.info("device %s; run dir %s; config %s", device, expdir, cfg.to_dict())
    model = build_model(cfg).to(device)
    judge = load_detector(cfg.weights, device)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("model: %s params, %s trainable", f"{n_params:,}", f"{n_train:,}")
    with open(os.path.join(expdir, "params.json"), "w") as f:
        json.dump({"config": cfg.to_dict(), "params": n_params, "trainable": n_train}, f, indent=2)
    history, best_path, last_path = fit(
        cfg, model, Losses(cfg, device, judge), device, expdir, root
    )
    model.load_state_dict(torch.load(best_path, map_location=device))
    summary = evaluate(cfg, model, device, os.path.join(expdir, "scores"), root, judge)
    commit, dirty = git_revision()
    results: dict[str, Any] = {
        "exp_name": cfg.exp_name,
        "expdir": expdir,
        "params": n_params,
        "trainable": n_train,
        "epochs_run": len(history),
        "best_epoch": next((r["epoch"] for r in reversed(history) if r.get("best")), None),
        "best_checkpoint": best_path,
        "last_checkpoint": last_path,
        "history_last": history[-1] if history else None,
        "git_commit": commit,
        "git_dirty": dirty,
        **headline(summary),
    }
    with open(os.path.join(expdir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    logging.info("results: %s", results)
    return results
