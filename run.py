"""Tornado onset: the published detector and three runs that build on it, on TorNet 2015.

    uv run run.py all                          # every step in order, into results/<stamp>/
    uv run run.py finetuned results/X          # one step into an existing directory
    uv run run.py summary results/X            # rebuild summary.csv/.txt and the figures

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

Each invocation writes one self-contained directory, results/<stamp>/: detector/ (floor and
ceiling predictions), one directory per trained run (params.json, history.csv, checkpoints/,
scores/ with the head predictions and summary.json, results.json), summary.csv, summary.txt
and figures/ (ladder, paired gain, ROC and PR curves, training curves). Each trained run uses
the 2015 training partition (validation held out by day), keeps the checkpoint with the best
validation head AUC and scores it against step zero. Model weights and data come from the
Hugging Face Hub and are cached on first use.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
from typing import Any

from tornado_onset.config import Config
from tornado_onset.figures import summarize_root
from tornado_onset.train import run, run_detector

RUNS: dict[str, dict[str, Any]] = {
    "frozen_one_scan": {"encoder_trainable": False, "input_scans": 1},
    "frozen_two_scans": {"encoder_trainable": False, "input_scans": 2},
    "finetuned": {"encoder_trainable": True, "input_scans": 2},
}


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("run", choices=["all", *RUNS, "detector", "summary"])
    p.add_argument(
        "dir",
        nargs="?",
        help="results directory to write into (default results/<timestamp>) or, for summary, "
        "the finished one to rebuild; a single run into an existing directory reuses its "
        "detector step",
    )
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    out = a.dir or os.path.join("results", datetime.datetime.now().strftime("%y%m%d_%H%M"))
    if a.run != "summary":
        names = list(RUNS) if a.run == "all" else [] if a.run == "detector" else [a.run]
        run_detector(Config(exp_name="detector", out_dir=out))
        for name in names:
            run(Config(exp_name=name, **RUNS[name], out_dir=out))
    print(summarize_root(out))


if __name__ == "__main__":
    main()
