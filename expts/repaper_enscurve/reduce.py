"""Aggregate the per-task ensemble curves into the two wandb runs the
ensembling figure reads (runs ``default`` and ``tuned``; keys ``ens_size``,
``test/avg_mae``, ``test/avg_auc``, and
``per_task/relbench/<db>/<table>/test/{mae,auc}``).

    pixi run python -m expts.repaper_enscurve.reduce
"""

import json
from pathlib import Path

import numpy as np
import wandb

ENTITY = "rtv2"
PROJECT = "2026-08-19-repaper-enscurve"
OUT_ROOT = "/dfs/user/ranjanr/ckpts/rtv2/repaper-enscurve"
N_TASKS = 21
N_SEEDS = 16


def reduce_variant(variant: str) -> None:
    paths = sorted((Path(OUT_ROOT) / variant).glob("*.json"))
    assert len(paths) == N_TASKS, (
        f"{variant}: {len(paths)} task curves, expected {N_TASKS}"
    )
    recs = [json.loads(p.read_text()) for p in paths]
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name=variant,
        config={"n_tasks": N_TASKS, "n_seeds": N_SEEDS},
        reinit="finish_previous",
    )
    for k in range(1, N_SEEDS + 1):
        row = {"ens_size": k}
        by_type = {"clf": [], "reg": []}
        for r in recs:
            v = r["curve"][str(k)]
            by_type[r["task_type"]].append(v)
            mkey = "auc" if r["task_type"] == "clf" else "mae"
            row[f"per_task/relbench/{r['db']}/{r['table']}/test/{mkey}"] = v
        row["test/avg_auc"] = float(np.mean(by_type["clf"]))
        row["test/avg_mae"] = float(np.mean(by_type["reg"]))
        wandb.log(row)
        print(
            f"{variant} ens={k}: auc={row['test/avg_auc']:.4f} "
            f"mae={row['test/avg_mae']:.4f}",
            flush=True,
        )
    run.finish()


if __name__ == "__main__":
    reduce_variant("default")
    # reduce_variant("tuned")
