import json
import os
import subprocess
from pathlib import Path

from roach.slurm import Resources, submit
from roach.slurm.clusters.ilc import ILC

from expts.repaper.adapter.featurize_target_cells import EXCLUDE_DBS
from expts.repaper.config import CKPT, CLONE_ROOT, LOG_ROOT, SECRETS_DIR, SHARE

PRE_DIR = "~/scratch/hf/stanford-star/the-join-preprocessed"
REPO_ROOT = str(Path(__file__).resolve().parents[3])
ROWS_PLAN = f"{SHARE}/rows_plan_join_u12_prop.json"

N_B200, N_A100 = 4, 20
B200_SPEEDUP = 2.5
FIXED_ROWS_PER_DB = 8192


def shards() -> list[tuple[str, list[str], int]]:
    plan = json.loads(Path(ROWS_PLAN).expanduser().read_text())["rows"]
    by_db: dict[str, int] = {}
    for key, rows in plan.items():
        db = key.split("/", 1)[0]
        assert db not in EXCLUDE_DBS, f"{db} is excluded but is in the plan"
        by_db[db] = by_db.get(db, 0) + rows

    load = [(rows + FIXED_ROWS_PER_DB, db) for db, rows in by_db.items()]
    speed = [B200_SPEEDUP] * N_B200 + [1.0] * N_A100
    out: list[list] = [[0, []] for _ in speed]
    for rows, db in sorted(load, reverse=True):
        i = min(range(len(speed)), key=lambda j: out[j][0] / speed[j])
        out[i][0] += rows
        out[i][1].append(db)
    plan_out = []
    for i, (rows, dbs) in enumerate(out):
        card = "b200" if i < N_B200 else "a100"
        print(
            f"  shard {i:>2} {card}: {len(dbs):>3} dbs, {rows / 1e6:6.2f}M rows, "
            f"{rows / speed[i] / 1e6:6.2f}M speed-adjusted"
        )
        plan_out.append((card, dbs, rows - FIXED_ROWS_PER_DB * len(dbs)))
    return plan_out


def queued() -> set[str]:
    out = subprocess.run(
        ["squeue", "-h", "-u", os.environ["USER"], "-o", "%j"],
        capture_output=True,
        text=True,
        check=True,
    )
    return set(out.stdout.split())


busy = queued()

for i, (card, dbs, rows) in enumerate(shards()):
    name = f"adapter-featurize-join-prop-s{i:02d}"
    if name in busy:
        continue
    submit(
        "expts.repaper.adapter.featurize_target_cells:featurize_dbs",
        args=dict(
            dbs=dbs,
            db_task_list="expts/pretrain/all_5gb_cutoff.json",
            pre_dir=PRE_DIR,
            features_root=f"{SHARE}/features_join_u12_prop",
            ckpt=CKPT,
            local_ctx_size=256,
            bfs_width=32,
            shuffle_seed=0,
            context_seed=0,
            batch_size=1024,
            min_rows=128,
            max_rows=4096,
            rows_plan_path=ROWS_PLAN,
            expected_gib=rows * 1024 / 2**30 * 1.2,
        ),
        resources=Resources(
            partition="il",
            account="infolab",
            qos="il-lo",
            time="12:00:00",
            gpus=f"{card}:1",
            cpus_per_task=14 if card == "a100" else 32,
            ntasks=None,
            exclusive=False,
            mem="120G" if card == "a100" else "240G",
            mem_per_gpu=None,
            constraint="ampere" if card == "a100" else None,
            nodelist=None if card == "a100" else "blackwell1",
            reservation=None,
            dependency=None,
            exclude="ampere4,ampere6,ampere7,ampere9" if card == "a100" else None,
        ),
        name=name,
        repo_root=REPO_ROOT,
        cluster=ILC,
        job_env="expts/job_env.sh",
        log_root=f"{LOG_ROOT}/repaper/adapter/slurm-logs",
        clone_root=CLONE_ROOT,
        secrets_dir=SECRETS_DIR,
    )
