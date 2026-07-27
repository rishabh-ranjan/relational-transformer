#!/usr/bin/env python
"""Stage 4: per-task JSONs -> the two result tables.

Reads ``<out-dir>/<method>/<db>__<table>.json`` (written by ``eval.py``) and prints,
per task type, one row per method and one column pair per context point: the metric
and the mean in-context label count.

    method | 256 | lbl | 512 | lbl | ... | 8192 | lbl

The metric is an **unweighted mean over tasks** -- AUROC for classification (higher
better), NMAE for regression (lower better). ``lbl`` is the same mean over the same
tasks of each task's mean in-context labels.

``lbl`` is the fairness check, not decoration. It is a pure function of the sampler
-- context config, seeds, ``items_per_task`` -- and has nothing to do with which
model consumed the context. So every method must report the *same* ``lbl`` at a
given ctx; if they do not, the methods were not evaluated on the same x-axis and the
metric columns are not comparable. This script diffs them and exits non-zero when
they disagree by more than a rounding hair.

    python expts/dbinfer/reduce.py --out-dir /dfs/user/$USER/dbinfer-scaling
    python expts/dbinfer/reduce.py --out-dir ... --tsv results.tsv --per-task
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CTX_SIZES = [256, 512, 1024, 2048, 4096, 8192]
# Row order in the printed tables: the baseline first, RT-J last as the headline.
METHOD_ORDER = ["rdblearn_tabicl", "rt"]
DISPLAY = {"rdblearn_tabicl": "rdblearn_tabicl", "rt": "**rt-j**"}
# `lbl` values are means of floats over the same tasks, so they should agree to the
# last bit; allow a hair for accumulation order across separate runs.
LBL_TOL = 1e-6


def _nanmean(xs) -> float:
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else float("nan")


def load(out_dir: Path) -> dict:
    """``{method: [record, ...]}`` for every method directory present."""
    per_method = {}
    for mdir in sorted(p for p in out_dir.iterdir() if p.is_dir()):
        recs = [json.loads(p.read_text()) for p in sorted(mdir.glob("*.json"))]
        if recs:
            per_method[mdir.name] = recs
    return per_method


def aggregate(recs: list, task_type: str, ctxs: list) -> dict:
    """Mean metric and mean ``mean_labels`` per ctx over one task type's tasks."""
    sel = [r for r in recs if r["task_type"] == task_type]
    out = {
        "tasks": sorted(r["task"] for r in sel),
        "n": len(sel),
        "metric": {},
        "lbl": {},
    }
    for ctx in ctxs:
        vals, lbls = [], []
        for r in sel:
            entry = r["per_ctx"].get(str(ctx))
            if entry is None:
                continue
            vals.append(entry["metric_value"])
            lbls.append(entry["mean_labels"])
        out["metric"][ctx] = _nanmean(vals)
        out["lbl"][ctx] = _nanmean(lbls)
    return out


def _fmt(v, nd=4) -> str:
    return "--" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def table(
    per_method: dict, task_type: str, ctxs: list, metric_name: str, arrow: str
) -> str:
    aggs = {m: aggregate(recs, task_type, ctxs) for m, recs in per_method.items()}
    aggs = {m: a for m, a in aggs.items() if a["n"]}
    if not aggs:
        return f"(no {task_type} tasks)\n"
    n = max(a["n"] for a in aggs.values())
    head = f"mean {metric_name} over {n} {task_type} task(s) ({arrow})"
    lines = [head, ""]
    lines.append("| method | " + " | ".join(f"{c} | lbl" for c in ctxs) + " |")
    lines.append("|---" * (1 + 2 * len(ctxs)) + "|")
    order = [m for m in METHOD_ORDER if m in aggs] + [
        m for m in aggs if m not in METHOD_ORDER
    ]
    for m in order:
        a = aggs[m]
        cells = []
        for c in ctxs:
            cells.append(_fmt(a["metric"][c]))
            cells.append(_fmt(a["lbl"][c], 2))
        lines.append(f"| {DISPLAY.get(m, m)} | " + " | ".join(cells) + " |")
    lines.append("")
    for m in order:
        lines.append(f"  {m}: n={aggs[m]['n']}  {', '.join(aggs[m]['tasks'])}")
    return "\n".join(lines) + "\n"


def check_labels(per_method: dict, ctxs: list) -> list:
    """Every method must report the same ``lbl`` per (task_type, ctx). Return the diffs."""
    problems = []
    for task_type in ("clf", "reg"):
        aggs = {m: aggregate(recs, task_type, ctxs) for m, recs in per_method.items()}
        aggs = {m: a for m, a in aggs.items() if a["n"]}
        if len(aggs) < 2:
            continue
        # Comparing means is only meaningful over the same task set.
        task_sets = {m: tuple(a["tasks"]) for m, a in aggs.items()}
        if len(set(task_sets.values())) > 1:
            problems.append(
                f"{task_type}: methods cover different task sets, means are not comparable: "
                + "; ".join(f"{m}={list(t)}" for m, t in task_sets.items())
            )
        for ctx in ctxs:
            vals = {
                m: a["lbl"][ctx] for m, a in aggs.items() if np.isfinite(a["lbl"][ctx])
            }
            if len(vals) < 2:
                continue
            spread = max(vals.values()) - min(vals.values())
            if spread > LBL_TOL:
                problems.append(
                    f"{task_type} ctx={ctx}: mean_labels differ by {spread:.6g} "
                    + "("
                    + ", ".join(f"{m}={v:.6f}" for m, v in vals.items())
                    + ") "
                    "-- the methods did not see the same contexts"
                )
    return problems


def per_task_table(per_method: dict, ctxs: list) -> str:
    lines = [
        "per task",
        "",
        "| method | task | type | n | "
        + " | ".join(str(c) for c in ctxs)
        + " | lbl@256 |",
        "|---" * (5 + len(ctxs)) + "|",
    ]
    for m, recs in per_method.items():
        for r in sorted(recs, key=lambda r: (r["task_type"], r["task"])):
            cells = []
            for c in ctxs:
                e = r["per_ctx"].get(str(c))
                cells.append("--" if e is None else _fmt(e["metric_value"]))
            e256 = r["per_ctx"].get(str(ctxs[0]))
            n = e256["n"] if e256 else "--"
            lbl = _fmt(e256["mean_labels"], 2) if e256 else "--"
            lines.append(
                f"| {m} | {r['task']} | {r['task_type']} | {n} | "
                + " | ".join(cells)
                + f" | {lbl} |"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ctx-sizes", nargs="+", type=int, default=CTX_SIZES)
    ap.add_argument(
        "--per-task", action="store_true", help="also print the per-task appendix"
    )
    ap.add_argument("--tsv", default=None, help="write the aggregate rows here as TSV")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    per_method = load(out_dir)
    if not per_method:
        return print(f"no results under {out_dir}") or 1
    ctxs = sorted(args.ctx_sizes)

    print("## Table 1 -- classification\n")
    print(table(per_method, "clf", ctxs, "AUROC", "higher is better"))
    print("## Table 2 -- regression\n")
    print(table(per_method, "reg", ctxs, "NMAE", "lower is better"))

    if args.per_task:
        print(per_task_table(per_method, ctxs))

    if args.tsv:
        rows = ["method\ttask_type\tn\tctx\tmetric\tmean_labels"]
        for m, recs in per_method.items():
            for tt in ("clf", "reg"):
                a = aggregate(recs, tt, ctxs)
                if not a["n"]:
                    continue
                for c in ctxs:
                    rows.append(
                        f"{m}\t{tt}\t{a['n']}\t{c}\t{a['metric'][c]:.6f}\t{a['lbl'][c]:.6f}"
                    )
        Path(args.tsv).expanduser().write_text("\n".join(rows) + "\n")
        print(f"wrote {args.tsv}")

    problems = check_labels(per_method, ctxs)
    if problems:
        print("\nFAIRNESS CHECK FAILED")
        for p in problems:
            print(f"  - {p}")
        return 1
    if len(per_method) > 1:
        print("\nfairness check: mean_labels agree across all methods at every ctx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
