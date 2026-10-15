"""Tables over finished runs: the head against the floor and ceiling per config and horizon,
mean and spread across seeds, and paired comparisons between two configs on the same files.

    onset-summary [ROOT ...]                 default root: experiments/
    onset-summary --compare EXP_A EXP_B      paired head-vs-head delta per seed, pre-onset WC
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import pandas as pd

from tornado_onset.metrics import paired_difference


def find_runs(roots: list[str]) -> list[str]:
    runs = []
    for root in roots:
        for d, _dirs, files in os.walk(root):
            if "results.json" in files and os.path.exists(
                os.path.join(d, "scores", "summary.json")
            ):
                runs.append(d)
    return sorted(runs)


def run_rows(run: str) -> list[dict]:
    with open(os.path.join(run, "params.json")) as f:
        cfg = json.load(f)["config"]
    with open(os.path.join(run, "results.json")) as f:
        res = json.load(f)
    with open(os.path.join(run, "scores", "summary.json")) as f:
        summary = json.load(f)
    rows = []
    nan = float("nan")

    def auc(view: dict, w: str) -> float:
        return view.get(w, {}).get("WC", {}).get("auc", nan)

    for e in summary["horizons"]:
        pre, paired = e["pre_onset"], e["pre_onset_paired_vs_persistence"]
        d = paired.get("head", {}).get("WC", {})
        ci = d.get("delta_auc_ci95") or [nan, nan]
        rows.append(
            {
                "exp_name": cfg["exp_name"],
                "seed": cfg["seed"],
                "run": run,
                "horizon": e["horizon"],
                "lead": e["lead_minutes_median"],
                "pre_onset_cases": e["pre_onset_cases"],
                "floor": auc(pre, "persistence"),
                "ceiling": auc(pre, "truth"),
                "head": auc(pre, "head"),
                "delta": d.get("delta_auc", nan),
                "delta_lo": ci[0],
                "delta_hi": ci[1],
                "best_epoch": res.get("best_epoch") or nan,
                "epochs_run": res.get("epochs_run"),
            }
        )
    return rows


def table(rows: list[dict]) -> pd.DataFrame:
    """One line per config and horizon: seeds, head mean ± sd, floor, ceiling, mean delta."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    g = df.groupby(["exp_name", "horizon"])
    out = g.agg(
        seeds=("seed", "nunique"),
        lead=("lead", "median"),
        cases=("pre_onset_cases", "first"),
        floor=("floor", "mean"),
        ceiling=("ceiling", "mean"),
        head=("head", "mean"),
        head_sd=("head", "std"),
        delta=("delta", "mean"),
        delta_lo=("delta_lo", "mean"),
        delta_hi=("delta_hi", "mean"),
        best_epoch=("best_epoch", "mean"),
    ).reset_index()
    out["head_sd"] = out["head_sd"].fillna(0.0)
    return out


def format_table(t: pd.DataFrame) -> str:
    if t.empty:
        return "no runs found"
    lines = [
        "exp_name                        h  lead seeds cases  floor  ceil   head (±sd)      head-floor [95%]        best ep"
    ]
    for r in t.to_dict("records"):  # not itertuples/iterrows: "head" collides with pandas
        lines.append(
            f"{r['exp_name']:<30} {int(r['horizon'])}  {r['lead']:4.0f} {int(r['seeds']):5d} "
            f"{int(r['cases']):5d}  {r['floor']:.3f}  {r['ceiling']:.3f}  "
            f"{r['head']:.3f} (±{r['head_sd']:.3f})  "
            f"{r['delta']:+.3f} [{r['delta_lo']:+.3f}, {r['delta_hi']:+.3f}]   {r['best_epoch']:.1f}"
        )
    return "\n".join(lines)


def compare(rows: list[dict], a: str, b: str, n_boot: int, seed: int = 0) -> list[dict]:
    """Paired head-vs-head AUC difference (b - a) on the common pre-onset-scoped files, per
    horizon and per seed pair (seeds matched by rank)."""
    runs: dict[str, dict[int, str]] = defaultdict(dict)
    for r in rows:
        runs[r["exp_name"]][r["seed"]] = r["run"]
    if a not in runs or b not in runs:
        raise SystemExit(f"need runs for both {a!r} and {b!r}; have {sorted(runs)}")
    horizons = sorted({r["horizon"] for r in rows})
    out = []
    for ra, rb in zip(sorted(runs[a].items()), sorted(runs[b].items()), strict=False):
        for h in horizons:
            pa = pd.read_csv(os.path.join(ra[1], "scores", f"predictions_head_h{h}.csv"))
            pb = pd.read_csv(os.path.join(rb[1], "scores", f"predictions_head_h{h}.csv"))
            scope = lambda d: d[d["pre_onset"] | (d["category"] != "TOR")]  # noqa: E731
            d = paired_difference(scope(pa), scope(pb), n_boot, seed)
            out.append(
                {
                    "horizon": h,
                    "seed_a": ra[0],
                    "seed_b": rb[0],
                    **{v: d[v] for v in ("NC", "RC", "WC")},
                }
            )
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("roots", nargs="*", default=["experiments"])
    p.add_argument("--compare", nargs=2, metavar=("EXP_A", "EXP_B"))
    p.add_argument("--bootstrap", type=int, default=1000)
    a = p.parse_args(argv)
    rows = [r for run in find_runs([r for r in a.roots if os.path.isdir(r)]) for r in run_rows(run)]
    print(format_table(table(rows)))
    if a.compare:
        print(f"\npaired, pre-onset scoped: {a.compare[1]} minus {a.compare[0]} (head vs head)")
        for c in compare(rows, a.compare[0], a.compare[1], a.bootstrap):
            w = c["WC"]
            print(
                f"  h{c['horizon']} seeds {c['seed_a']}/{c['seed_b']}: WC {w['delta_auc']:+.3f} "
                f"[{w['delta_auc_ci95'][0]:+.3f}, {w['delta_auc_ci95'][1]:+.3f}]  "
                f"NC {c['NC']['delta_auc']:+.3f}  RC {c['RC']['delta_auc']:+.3f}"
            )
    if not rows:
        print("(no runs with results.json and scores/summary.json under", ", ".join(a.roots) + ")")
    return None


if __name__ == "__main__":
    main()
