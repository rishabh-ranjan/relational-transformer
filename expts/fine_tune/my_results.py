"""Build my_results.csv from the ensembling runs, and my_results.md from that
plus results.csv -- the published comparison with our row in it.

    pixi run python expts/fine_tune/my_results.py

`my_results.csv` holds our numbers alone, in results.csv's columns so the two
concatenate. `my_results.md` is results.md's four tables with `Ours` added,
same ordering, bolding and units -- the rendering is imported from
`make_results.py` rather than repeated.

Our row is one fine-tuned checkpoint per task (`submit.py`), its context
configuration chosen on validation over `submit_hpo_ens.py`'s grid, scored on
test as the average over 4 context seeds. `val_score` is the winning
configuration's validation score, which is a selection criterion and so
optimistically biased, exactly as the published `(H)` rows are.

A cell marked `^` is a test set the run subsampled (`submit_ens_only.items_for`
caps the largest ones), so it is not comparable to a published number over the
whole split. Rerunning the capped tasks uncapped is what removes the marker.
"""

import json
from pathlib import Path

import pandas as pd
import wandb
from make_results import NOTES, SHORT, legend, sections, stds
from submit_ens_only import TASKS, items_for, ntest

ENTITY = "rtv2"
PROJECT = "2026-08-10-fine_tune_hpo_ens"
OUT_ROOT = Path("/dfs/user/ranjanr/ckpts")

# The row label our numbers get in the published comparison.
OURS = "Ours"

# What a task is scored by, and which relbench metric name that is.
METRICS = {"auroc": "roc_auc", "nmae": "mae"}


def task_types() -> dict[str, str]:
    """`{db}/{task}` -> results.csv's `task_type`, so our rows land in the same
    table as everyone else's."""
    d = pd.read_csv(Path(__file__).parent / "results.csv")
    return dict(zip(d.dataset + "/" + d.task, d.task_type))


def tuned() -> dict[str, dict]:
    """`{db}/{task}` -> the `tuning.json` its run wrote: winning config and the
    validation score that chose it."""
    out = {}
    for path in (OUT_ROOT / ENTITY / PROJECT).glob("*/tuning.json"):
        out.update(json.loads(path.read_text()))
    return out


def tested() -> dict[str, tuple[str, float, int]]:
    """`{db}/{task}` -> (metric, test value in percent, ensemble size).

    The largest ensemble size the run reached, so a table can be built while
    the sweep is still going; the size comes back with it to say so.
    """
    out: dict[str, tuple[str, float, int]] = {}
    for run in wandb.Api().runs(f"{ENTITY}/{PROJECT}"):
        name = run.config["run_name"]
        keys = [m for m in METRICS if f"{m}/test/{name}" in run.summary]
        if not keys:
            continue
        (metric,) = keys
        key = f"{metric}/test/{name}"
        rows = [
            (int(r["ens_size"]), float(r[key]))
            for r in run.history(keys=["ens_size", key], pandas=False)
            if r.get("ens_size") is not None and r.get(key) is not None
        ]
        if not rows:
            continue
        size, value = max(rows)
        if name not in out or size > out[name][2]:
            out[name] = (metric, value, size)
    return out


def our_rows() -> pd.DataFrame:
    """Our results in results.csv's schema, one row per task.

    Scores go in raw, the units results.csv keeps: a probability for AUROC and
    an unnormalized MAE for regression, which is what `make_results.table`
    expects to normalize itself.
    """
    types, tune, test = task_types(), tuned(), tested()
    rows = []
    for db, task in TASKS:
        pair = f"{db}/{task}"
        if pair not in test or pair not in tune:
            continue
        metric, value, size = test[pair]
        cfg = tune[pair]["best_cfg"]
        # `metric_for` scores regression on the normalized scale, so its value
        # is already nMAE; results.csv wants the raw MAE behind it.
        scale = 1.0 if metric == "auroc" else stds[pair]
        rows.append(
            {
                "model": OURS,
                "dataset": db,
                "task": task,
                "task_type": types[pair],
                "seed": 0,
                "n_trials": len(tune[pair]["val_scores"]),
                "metric": METRICS[metric],
                "device": "cuda",
                "selected": True,
                "config": json.dumps(
                    dict(
                        zip(
                            (
                                "ctx_size",
                                "local_ctx_size",
                                "bfs_width",
                                "prefer_latest",
                            ),
                            cfg,
                        )
                    )
                ),
                "config_tag": "hpo",
                "val_score": tune[pair]["best_value"] * scale,
                "test_score": value / 100 * scale,
                "ensemble_size": size,
                "test_items": min(items_for(db, task), int(ntest()[pair])),
                "test_rows": int(ntest()[pair]),
            }
        )
    return pd.DataFrame(rows)


def marks(ours: pd.DataFrame) -> dict[tuple[str, str], str]:
    """`^` on every cell of ours whose test set was subsampled."""
    return {
        (OURS, f"{r.dataset}/{r.task}"): "^"
        for r in ours.itertuples()
        if r.test_items < r.test_rows
    }


def main() -> None:
    ours = our_rows()
    ours.to_csv(Path(__file__).parent / "my_results.csv", index=False)

    published = pd.read_csv(Path(__file__).parent / "results.csv")
    published["pair"] = published.dataset + "/" + published.task
    dflt = published[published.config_tag == "default"].assign(run="D")
    hpo = published[published.selected].assign(run="H")
    a = pd.concat([dflt, hpo])
    a["row"] = a.model + " (" + a.run + ")"

    mine = ours.copy()
    mine["pair"] = mine.dataset + "/" + mine.task
    mine["row"] = OURS

    frame = pd.concat([a, mine], ignore_index=True)
    partial = sorted({p for (_, p) in marks(ours)})
    out = (
        "# Results, ours included\n\n"
        + "\n\n".join(sections(frame, marks(ours)))
        + "\n\n# Legend\n\n"
        + legend(list(SHORT))
        + "\n\n"
        + NOTES
        + f"- `{OURS}` = a Relational Transformer fine-tuned per task, its context configuration"
        " chosen on validation and its test prediction averaged over context seeds.\n"
        "- `^` = the test split was subsampled for that cell, so it is not"
        " comparable to a number over the whole split.\n"
        + (f"- Subsampled tasks: {', '.join(partial)}.\n" if partial else "")
    )
    (Path(__file__).parent / "my_results.md").write_text(out)
    print(f"wrote my_results.csv ({len(ours)} tasks) and my_results.md")


if __name__ == "__main__":
    main()
