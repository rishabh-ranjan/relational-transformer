"""Aggregate one arm's per-task JSONs into the wandb run its figure reads.

Reads every ``<arm_dir>/<db>__<table>.json`` written by ``run.py``, refuses to
reduce an incomplete arm (the figure would silently average over fewer tasks),
and logs ONE wandb run with one history row per ctx size:

    ctx_size
    ctx_scaling/steps=0/test/avg_mae               mean NMAE over reg tasks
    ctx_scaling/steps=0/test/avg_auc               mean AUROC over clf tasks
    ctx_scaling/steps=0/test/avg_mean_labels_reg   mean in-context labels, reg
    ctx_scaling/steps=0/test/avg_mean_labels_clf   mean in-context labels, clf
    per_task/ctx_scaling/steps=0/relbench/<db>/<table>/test/{mae,auc,mean_labels}

Edit the loop at the bottom and run directly; each (project, run_name,
arm_dir, expected task count) line is one figure series.
"""

import json
from pathlib import Path

import numpy as np
import wandb

ENTITY = "rtv2"
KEY = "ctx_scaling/steps=0/test"


def reduce_arm(*, project: str, run_name: str, arm_dir: str, n_tasks: int) -> None:
    paths = sorted(Path(arm_dir).expanduser().glob("*.json"))
    assert len(paths) == n_tasks, (
        f"{arm_dir}: {len(paths)} task JSONs, expected {n_tasks}; "
        f"reduce only a complete arm"
    )
    recs = [json.loads(p.read_text()) for p in paths]

    ctxs = sorted({int(c) for r in recs for c in r["per_ctx"]})
    run = wandb.init(
        entity=ENTITY,
        project=project,
        name=run_name,
        config={"arm_dir": str(arm_dir), "ctx_sizes": ctxs, "n_tasks": n_tasks},
        reinit="finish_previous",
    )
    for ctx in ctxs:
        row = {"ctx_size": ctx}
        by_type = {
            "clf": {"metric": [], "labels": []},
            "reg": {"metric": [], "labels": []},
        }
        for r in recs:
            entry = r["per_ctx"].get(str(ctx))
            if entry is None:
                continue
            mkey = "auc" if r["task_type"] == "clf" else "mae"
            base = f"per_task/{KEY.removesuffix('/test')}/relbench/{r['db']}/{r['table']}/test"
            row[f"{base}/{mkey}"] = entry["metric_value"]
            row[f"{base}/mean_labels"] = entry["mean_labels"]
            if entry["metric_value"] is not None:
                by_type[r["task_type"]]["metric"].append(entry["metric_value"])
            by_type[r["task_type"]]["labels"].append(entry["mean_labels"])
        row[f"{KEY}/avg_auc"] = float(np.mean(by_type["clf"]["metric"]))
        row[f"{KEY}/avg_mae"] = float(np.mean(by_type["reg"]["metric"]))
        row[f"{KEY}/avg_mean_labels_clf"] = float(np.mean(by_type["clf"]["labels"]))
        row[f"{KEY}/avg_mean_labels_reg"] = float(np.mean(by_type["reg"]["labels"]))
        wandb.log(row)
        print(
            f"{run_name} ctx={ctx}: auc={row[f'{KEY}/avg_auc']:.4f} "
            f"mae={row[f'{KEY}/avg_mae']:.4f} "
            f"(clf n={len(by_type['clf']['metric'])}, reg n={len(by_type['reg']['metric'])})",
            flush=True,
        )
    run.finish()


ROOT = "/dfs/user/ranjanr/ckpts/rtv2/repaper-scaling"

if __name__ == "__main__":
    # (project, run name the gen script fetches, arm dir, expected #tasks).
    # One line per finished arm; a line for an unfinished arm fails its count
    # assert, so enable them as they complete.
    for project, run_name, arm, n_tasks in [
        ("2026-08-19-repaper-abl", "abl/rw", f"{ROOT}/subsampled/rt", 21),
        ("2026-08-19-repaper-abl", "abl/sem", f"{ROOT}/subsampled/rt", 21),
        ("2026-08-19-repaper-abl", "abl/nosem", f"{ROOT}/abl/nosem", 21),
        ("2026-08-19-repaper-abl", "abl/rand", f"{ROOT}/abl/rand", 21),
        ("2026-08-19-repaper-abl", "abl/bfs32", f"{ROOT}/abl/bfs32", 21),
        ("2026-08-19-repaper-abl", "abl/bfs256", f"{ROOT}/abl/bfs256", 21),
        ("2026-08-19-repaper-abl", "abl/vdb_rdblearn", f"{ROOT}/abl/vdb_rdblearn", 21),
        ("2026-08-19-repaper-abl", "abl/vdb_rt", f"{ROOT}/abl/vdb_rt", 21),
        ("2026-08-19-repaper-subsampled", "rt-j", f"{ROOT}/subsampled/rt", 21),
        (
            "2026-08-19-repaper-subsampled",
            "precomputed_rdblearn_tabicl_batched",
            f"{ROOT}/subsampled/rdblearn_tabicl",
            21,
        ),
        (
            "2026-08-19-repaper-subsampled",
            "precomputed_sql_tabicl_batched",
            f"{ROOT}/subsampled/sql_tabicl",
            21,
        ),
        (
            "2026-08-19-repaper-subsampled",
            "precomputed_sql_lightgbm",
            f"{ROOT}/subsampled/sql_lgbm",
            21,
        ),
        # ("2026-08-19-repaper-subsampled", "precomputed_rdblearn_lightgbm", f"{ROOT}/subsampled/rdblearn_lgbm", 21),
        # ("2026-08-19-repaper-fulltest", "rt-j", f"{ROOT}/fulltest/rt", 21),
        # ("2026-08-19-repaper-fulltest", "precomputed_rdblearn_tabicl_batched", f"{ROOT}/fulltest/rdblearn_tabicl", 21),
        # ("2026-08-19-repaper-fulltest", "precomputed_sql_tabicl_batched", f"{ROOT}/fulltest/sql_tabicl", 21),
        # ("2026-08-19-repaper-fulltest", "precomputed_rdblearn_lightgbm", f"{ROOT}/fulltest/rdblearn_lgbm", 21),
        # ("2026-08-19-repaper-fulltest", "precomputed_sql_lightgbm", f"{ROOT}/fulltest/sql_lgbm", 21),
    ]:
        reduce_arm(project=project, run_name=run_name, arm_dir=arm, n_tasks=n_tasks)
