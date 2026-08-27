import json
from pathlib import Path

import numpy as np
import wandb

from expts.repaper.config import OUT_ROOT, project

KEY = "ctx_scaling/steps=0/test"


def reduce_arm(
    *,
    project: str,
    run_name: str,
    arm_dir: str,
    n_tasks: int,
    ext_dir: str | None = None,
    n_ext_tasks: int = 0,
) -> bool:
    arm = Path(arm_dir).expanduser()
    marker = arm / (
        f"wandb-{project}-{run_name.replace('/', '-')}{'-ext' if ext_dir else ''}.json"
    )
    if marker.exists():
        print(f"{run_name}: logged already, {json.loads(marker.read_text())['url']}")
        return True
    paths = sorted(arm.glob("*__*.json"))
    if len(paths) != n_tasks:
        print(f"{run_name}: {len(paths)}/{n_tasks} task JSONs, not reduced", flush=True)
        return False
    recs = [json.loads(p.read_text()) for p in paths]
    if ext_dir:
        ext_paths = sorted(Path(ext_dir).expanduser().glob("*__*.json"))
        if len(ext_paths) != n_ext_tasks:
            print(
                f"{run_name}: {len(ext_paths)}/{n_ext_tasks} extension JSONs, "
                f"not reduced",
                flush=True,
            )
            return False
        ext = {
            json.loads(p.read_text())["task"]: json.loads(p.read_text())
            for p in ext_paths
        }
        for r in recs:
            if r["task"] in ext:
                assert not set(r["per_ctx"]) & set(ext[r["task"]]["per_ctx"])
                r["per_ctx"].update(ext[r["task"]]["per_ctx"])

    ctxs = sorted({int(c) for r in recs for c in r["per_ctx"]})
    run = wandb.init(
        entity="rtv2",
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
            if str(ctx) not in r["per_ctx"]:
                continue
            entry = r["per_ctx"][str(ctx)]
            mkey = "auc" if r["task_type"] == "clf" else "mae"
            base = f"per_task/{KEY.removesuffix('/test')}/relbench/{r['db']}/{r['table']}/test"
            row[f"{base}/{mkey}"] = entry["metric_value"]
            row[f"{base}/mean_labels"] = entry["mean_labels"]
            by_type[r["task_type"]]["metric"].append(entry["metric_value"])
            by_type[r["task_type"]]["labels"].append(entry["mean_labels"])
        for tt, mkey in (("clf", "auc"), ("reg", "mae")):
            n_type = sum(r["task_type"] == tt for r in recs)
            if len(by_type[tt]["metric"]) == n_type:
                row[f"{KEY}/avg_{mkey}"] = float(np.mean(by_type[tt]["metric"]))
                row[f"{KEY}/avg_mean_labels_{tt}"] = float(
                    np.mean(by_type[tt]["labels"])
                )
            else:
                assert not by_type[tt]["metric"], (
                    f"{run_name} ctx={ctx}: {len(by_type[tt]['metric'])}/{n_type} "
                    f"{tt} tasks have this context size"
                )
        wandb.log(row)
        print(
            f"{run_name} ctx={ctx}: auc={row.get(f'{KEY}/avg_auc', float('nan')):.4f} "
            f"mae={row.get(f'{KEY}/avg_mae', float('nan')):.4f} "
            f"(clf n={len(by_type['clf']['metric'])}, reg n={len(by_type['reg']['metric'])})",
            flush=True,
        )
    run.finish()
    marker.write_text(json.dumps({"id": run.id, "url": run.url}) + "\n")
    return True


ROOT = f"{OUT_ROOT}/repaper-scaling"

if __name__ == "__main__":
    done = []
    for project_name, run_name, arm, n_tasks, ext in [
        ("fulltest", "rt-j", f"{ROOT}/fulltest/rt", 21, None),
        (
            "fulltest",
            "precomputed_rdblearn_tabicl_batched",
            f"{ROOT}/fulltest/rdblearn_tabicl",
            21,
            f"{ROOT}/fulltest_ext/rdblearn_tabicl",
        ),
        (
            "fulltest",
            "precomputed_sql_tabicl_batched",
            f"{ROOT}/fulltest/sql_tabicl",
            21,
            f"{ROOT}/fulltest_ext/sql_tabicl",
        ),
        (
            "fulltest",
            "precomputed_rdblearn_lightgbm",
            f"{ROOT}/fulltest/rdblearn_lgbm",
            21,
            None,
        ),
        ("fulltest", "precomputed_sql_lightgbm", f"{ROOT}/fulltest/sql_lgbm", 21, None),
        ("subsampled", "rt-j", f"{ROOT}/subsampled/rt", 21, None),
        (
            "subsampled",
            "precomputed_rdblearn_tabicl_batched",
            f"{ROOT}/subsampled/rdblearn_tabicl",
            21,
            None,
        ),
        (
            "subsampled",
            "precomputed_sql_tabicl_batched",
            f"{ROOT}/subsampled/sql_tabicl",
            21,
            None,
        ),
        (
            "subsampled",
            "precomputed_rdblearn_lightgbm",
            f"{ROOT}/subsampled/rdblearn_lgbm",
            21,
            None,
        ),
        (
            "subsampled",
            "precomputed_sql_lightgbm",
            f"{ROOT}/subsampled/sql_lgbm",
            21,
            None,
        ),
        ("abl", "abl/rw", f"{ROOT}/subsampled/rt", 21, None),
        ("abl", "abl/sem", f"{ROOT}/subsampled/rt", 21, None),
        ("abl", "abl/nosem", f"{ROOT}/abl/nosem", 21, None),
        ("abl", "abl/rand", f"{ROOT}/abl/rand", 21, None),
        ("abl", "abl/bfs32", f"{ROOT}/abl/bfs32", 21, None),
        ("abl", "abl/bfs256", f"{ROOT}/abl/bfs256", 21, None),
        ("abl", "abl/vdb_rdblearn", f"{ROOT}/abl/vdb_rdblearn", 21, None),
        ("abl", "abl/vdb_rt", f"{ROOT}/abl/vdb_rt", 21, None),
    ]:
        done.append(
            reduce_arm(
                project=project(project_name),
                run_name=run_name,
                arm_dir=arm,
                n_tasks=n_tasks,
                ext_dir=ext,
                n_ext_tasks=9 if ext else 0,
            )
        )
    print(f"{sum(done)}/{len(done)} arms logged", flush=True)
