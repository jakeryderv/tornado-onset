"""Tornado onset: the baseline and three experiments on TorNet 2015.

    uv run run.py all                 # the four runs back to back, then the table
    uv run run.py baseline            # one of them
    uv run run.py baseline --smoke    # a few thousand files, one epoch: does the path run?

Each run trains on the 2015 training partition (validation held out by day), keeps the
checkpoint with the best validation head AUC, scores it on the 2015 test partition against the
persistence floor and the ceiling, and lands under experiments/<name>_<timestamp>/ with
params.json, history.csv, checkpoints/, scores/ and results.json. Model weights and data come
from the Hugging Face Hub and are cached on first use.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any

from tornado_onset.config import Config
from tornado_onset.train import run

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
SMOKE: dict[str, Any] = {"max_files": 3000, "epochs": 1, "bootstrap": 100, "exp_dir": "smoke"}


def summary_table(root: str) -> str:
    """One line per finished run and horizon: pre-onset WC AUC of the floor, ceiling and head,
    the head's paired gain over the floor with its bootstrap interval, and the best epoch."""
    lines = [
        "run                             h lead cases  floor  ceil   head   head-floor [95%]      best ep"
    ]
    for d in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        path = os.path.join(root, d)
        try:
            with open(os.path.join(path, "results.json")) as f:
                res = json.load(f)
            with open(os.path.join(path, "scores", "summary.json")) as f:
                summary = json.load(f)
        except FileNotFoundError:
            continue
        for e in summary["horizons"]:
            auc = {w: e["pre_onset"][w]["WC"]["auc"] for w in ("persistence", "truth", "head")}
            d_ = e["pre_onset_paired_vs_persistence"]["head"]["WC"]
            lo, hi = d_["delta_auc_ci95"]
            lines.append(
                f"{d:<30} {e['horizon']}  {e['lead_minutes_median']:3.0f} {e['pre_onset_cases']:5d}  "
                f"{auc['persistence']:.3f}  {auc['truth']:.3f}  {auc['head']:.3f}  "
                f"{d_['delta_auc']:+.3f} [{lo:+.3f}, {hi:+.3f}]   {res.get('best_epoch')}"
            )
    return "\n".join(lines) if len(lines) > 1 else "no finished runs under " + root


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("experiment", choices=[*EXPERIMENTS, "all"])
    p.add_argument("--smoke", action="store_true", help="a short run to check the path")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    names = list(EXPERIMENTS) if a.experiment == "all" else [a.experiment]
    cfg = Config()
    for name in names:
        cfg = Config(exp_name=name, **EXPERIMENTS[name], **(SMOKE if a.smoke else {}))
        run(cfg)
    print(summary_table(cfg.exp_dir))


if __name__ == "__main__":
    main()
