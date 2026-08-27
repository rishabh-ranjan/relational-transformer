import json
from pathlib import Path

import numpy as np
import wandb

from expts.repaper.config import OUT_ROOT, project

N_TASKS = 21
N_SEEDS = 16


def reduce_variant(variant: str) -> None:
    root = Path(OUT_ROOT).expanduser() / "repaper-enscurve" / variant
    marker = root / f"wandb-{project('enscurve')}.json"
    if marker.exists():
        print(f"{variant}: logged already, {json.loads(marker.read_text())['url']}")
        return
    paths = sorted(root.glob("*__*.json"))
    if len(paths) != N_TASKS:
        print(f"{variant}: {len(paths)}/{N_TASKS} task curves, not reduced", flush=True)
        return
    recs = [json.loads(p.read_text()) for p in paths]
    run = wandb.init(
        entity="rtv2",
        project=project("enscurve"),
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
    marker.write_text(json.dumps({"id": run.id, "url": run.url}) + "\n")


if __name__ == "__main__":
    reduce_variant("default")
    reduce_variant("tuned")
