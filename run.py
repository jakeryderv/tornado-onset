"""Tornado onset: the published detector and three runs that build on it, on TorNet 2015.

    uv run run.py all          # every step in order, then the table
    uv run run.py finetuned    # one step (step zero is made first if missing)

    step  run               what changes                          question
    0     detector          nothing trained: the published         what the detector already gives
                            detector as designed, one scan,        on the onset question. Newest
                            scored on the 2015 test set            observed scan: the floor. Real
                                                                   future scan: the ceiling.
    1     frozen_one_scan   a new head trained on the onset        the scope change alone: can the
                            target; same single scan, encoder      detector's features be re-read
                            frozen                                 for "will it appear"
    2     frozen_two_scans  the second observed scan added,        does time help when the
                            features concatenated                  features are fixed
    3     finetuned         encoder unfrozen at a tenth of the     does letting the features adapt
                            head's learning rate                   to the target buy more

Each trained run uses the 2015 training partition (validation held out by day), keeps the
checkpoint with the best validation head AUC and scores it against step zero's floor and
ceiling, under experiments/<name>_<timestamp>/ (params.json, history.csv, checkpoints/, scores/,
results.json); step zero lives at experiments/detector/. Model weights and data come from the
Hugging Face Hub and are cached on first use.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any

from tornado_onset.config import Config
from tornado_onset.train import run, run_detector

RUNS: dict[str, dict[str, Any]] = {
    "frozen_one_scan": {"encoder_trainable": False, "input_scans": 1},
    "frozen_two_scans": {"encoder_trainable": False, "input_scans": 2},
    "finetuned": {"encoder_trainable": True, "input_scans": 2},
}


def summary_table(root: str) -> str:
    """Step zero's floor and ceiling, then one line per trained run and horizon: the head's
    pre-onset WC AUC, its paired gain over the floor with the bootstrap interval, best epoch."""

    def load(d: str) -> tuple[dict, dict] | None:
        try:
            with open(os.path.join(root, d, "scores", "summary.json")) as f:
                summary = json.load(f)
            res: dict = {}
            res_path = os.path.join(root, d, "results.json")
            if os.path.exists(res_path):
                with open(res_path) as f:
                    res = json.load(f)
            return summary, res
        except FileNotFoundError:
            return None

    def wc(entry: dict, name: str) -> float:
        return entry["pre_onset"].get(name, {}).get("WC", {}).get("auc", float("nan"))

    det = load("detector") if os.path.isdir(root) else None
    if det is None:
        return "no detector step under " + root
    lines = ["step 0, the published detector on one scan, pre-onset WC AUC:"]
    for e in det[0]["horizons"]:
        lines.append(
            f"  h{e['horizon']} (+{e['lead_minutes_median']:.0f} min, {e['pre_onset_cases']} cases):"
            f"  floor {wc(e, 'persistence'):.3f}   ceiling {wc(e, 'truth'):.3f}"
        )
    lines.append("")
    lines.append(
        "run                              h  lead   head   head-floor [95%]        best ep"
    )
    for d in sorted(os.listdir(root)):
        if d == "detector" or (loaded := load(d)) is None:
            continue
        summary, res = loaded
        for e in summary["horizons"]:
            d_ = e["pre_onset_paired_vs_persistence"]["head"]["WC"]
            lo, hi = d_["delta_auc_ci95"]
            lines.append(
                f"{d:<32} {e['horizon']}  {e['lead_minutes_median']:3.0f}   {wc(e, 'head'):.3f}  "
                f"{d_['delta_auc']:+.3f} [{lo:+.3f}, {hi:+.3f}]   {res.get('best_epoch')}"
            )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("run", choices=["detector", *RUNS, "all"])
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    names = list(RUNS) if a.run == "all" else [] if a.run == "detector" else [a.run]
    run_detector(Config(exp_name="detector"))
    for name in names:
        run(Config(exp_name=name, **RUNS[name]))
    print(summary_table(Config().exp_dir))


if __name__ == "__main__":
    main()
