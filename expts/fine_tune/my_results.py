import json
from pathlib import Path

import pandas as pd
import wandb
from make_results import NOTES, SHORT, legend, sections, stds
from submit_ens_only import TASKS, items_for, ntest

ENTITY = "rtv2"
PROJECT = "2026-08-10-fine_tune_hpo_ens"
OUT_ROOT = Path("~/scratch/ckpts").expanduser()
OURS = "Ours"

METRICS = {"auroc": "roc_auc", "nmae": "mae"}


def task_types() -> dict[str, str]:
    d = pd.read_csv(Path(__file__).parent / "results.csv")
    return dict(zip(d.dataset + "/" + d.task, d.task_type))


def tuned() -> dict[str, dict]:
    out = {}
    root = OUT_ROOT / ENTITY / PROJECT
    for path in sorted(root.glob("*/tuning.json"), key=lambda q: q.parent.name):
        out.update(json.loads(path.read_text()))
    return out


def tested() -> dict[str, tuple[str, float, int]]:
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
    types, tune, test = task_types(), tuned(), tested()
    rows = []
    for db, task in TASKS:
        pair = f"{db}/{task}"
        if pair not in test or pair not in tune:
            continue
        metric, value, size = test[pair]
        cfg = tune[pair]["best_cfg"]
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
