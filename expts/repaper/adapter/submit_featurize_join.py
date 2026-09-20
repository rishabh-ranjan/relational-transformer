import json
import os
import subprocess
from pathlib import Path

from roach.slurm import Resources, submit
from roach.slurm.clusters.ilc import ILC

from expts.repaper.config import CKPT, CLONE_ROOT, LOG_ROOT, SECRETS_DIR, SHARE

# Phase 1 of the adapter work: RT-J's representation of the masked target cell,
# at the end of relational unit 12, for the Join's pretraining mixture. The net
# is frozen, so an embedding is a fixed function of (row, context shape) and
# training the adapter never has to run RT-J again.
#
# The mixture is expts/pretrain/all_5gb_cutoff.json, the list this repo's own
# pretraining runs pass -- 13,243 pairs over 523 dbs, every one of them present
# and resolving to a task in this build of the-join-preprocessed. Not the Hub's
# db-task-lists/rt-j.json: 911 of its 10,815 pairs name 26 dbs the preprocessed
# repo does not ship.
#
# Shape and normalisation are what features_rt-j was built at: local_ctx_size
# 256, bfs_width 32, prefer_latest False, context_seed 0, and norm_out over the
# last block. So this is the same construction as rt_features and RelBench needs
# no dump of its own -- rt_features already holds every row of all 21 tasks,
# which is also the validation context, at whatever size we index out of it.
#
# 4096 rows per task, uniformly drawn rather than a prefix, since node order is
# time order. 36.6M rows, 34.9 GiB. The dumped row count is NOT the mixture
# weight: pretraining draws from a pool holding min(rows, 100_000) per task, so
# the manifest carries total_nodes and phase 2 weights by that.
PRE_DIR = "~/scratch/hf/stanford-star/the-join-preprocessed"
REPO_ROOT = str(Path(__file__).resolve().parents[3])

# 4 b200 and 20 a100. A b200 is ~2.5x an a100 per step (expts/fine_tune), so the
# b200 shards carry 2.5x the rows and every shard lands at about the same wall
# clock -- ~1.2M rows on an a100, ~3.1M on a b200, half an hour each at the
# 600-800 rows/s the relbench taps dump measured.
N_B200, N_A100 = 4, 20
B200_SPEEDUP = 2.5
# Each db costs one mmap populate of its ~0.5 GiB of rustler artifacts on top of
# its rows. Balancing on rows alone swept every tiny db into one shard -- 199 of
# them, whose startup would have dwarfed the work.
FIXED_ROWS_PER_DB = 8192


def shards() -> list[tuple[str, list[str], int]]:
    pre = Path(PRE_DIR).expanduser()
    by_db: dict[str, set[str]] = {}
    for db, name in json.loads(
        (Path(REPO_ROOT) / "expts/pretrain/all_5gb_cutoff.json").read_text()
    ):
        by_db.setdefault(db, set()).add(name)

    load = []
    for db, names in by_db.items():
        meta = json.loads((pre / db / "meta.json").read_text())
        info = json.loads((pre / db / "table_info.json").read_text())
        rows = 0
        for t in meta.get("tasks", []):
            if t["name"] not in names:
                continue
            table = t["entity_table"] if t.get("kind") == "autocomplete" else t["name"]
            key = f"{table}:Train" if f"{table}:Train" in info else f"{table}:Db"
            if key in info and info[key]["num_nodes"] >= 128:
                rows += min(info[key]["num_nodes"], 4096)
        load.append((rows + FIXED_ROWS_PER_DB, db))

    # Greedy longest-first against a per-shard speed, so a db with 276 tasks
    # does not hold the sweep open on its own.
    speed = [B200_SPEEDUP] * N_B200 + [1.0] * N_A100
    out = [[0, []] for _ in speed]
    for rows, db in sorted(load, reverse=True):
        i = min(range(len(speed)), key=lambda j: out[j][0] / speed[j])
        out[i][0] += rows
        out[i][1].append(db)
    plan = []
    for i, (rows, dbs) in enumerate(out):
        card = "b200" if i < N_B200 else "a100"
        print(f"  shard {i:>2} {card}: {len(dbs):>3} dbs, {rows / 1e6:.2f}M rows")
        plan.append((card, dbs, rows - FIXED_ROWS_PER_DB * len(dbs)))
    return plan


def queued() -> set[str]:
    out = subprocess.run(
        ["squeue", "-h", "-u", os.environ["USER"], "-o", "%j"],
        capture_output=True,
        text=True,
        check=True,
    )
    return set(out.stdout.split())


busy = queued()

# il and il-lo only; il-interactive is not ours to spend here. Read the split
# fresh every submission -- this one is not a default to inherit.
#
# The b200 cap of 2 is the *partition* QOS, so two b200 shards go to il, where
# priority 1000 outranks the il-lo jobs that hold blackwell1 and the cap is
# reachable; the other two go to il-lo, which is flagged OverPartQOS (b200=8).
# Those two spend 2 of il's ten gpus, leaving 8 for amperes, and il-lo takes the
# remaining twelve. The dump skips a task whose files exist, so a preempted
# shard resumes at the task it was in.
TIERS = ["il"] * 2 + ["il-lo"] * (N_B200 - 2) + ["il"] * 8 + ["il-lo"] * (N_A100 - 8)

for i, (card, dbs, rows) in enumerate(shards()):
    name = f"adapter-featurize-join-s{i:02d}"
    if name in busy:
        continue
    submit(
        "expts.repaper.adapter.featurize_target_cells:featurize_dbs",
        args=dict(
            dbs=dbs,
            db_task_list="expts/pretrain/all_5gb_cutoff.json",
            pre_dir=PRE_DIR,
            features_root=f"{SHARE}/features_join_u12",
            ckpt=CKPT,
            local_ctx_size=256,
            bfs_width=32,
            shuffle_seed=0,
            context_seed=0,
            batch_size=1024,
            min_rows=128,
            max_rows=4096,
            expected_gib=rows * 1024 / 2**30 * 1.2,
        ),
        resources=Resources(
            partition="il",
            account="infolab",
            qos=TIERS[i],
            time="12:00:00",
            gpus=f"{card}:1",
            # 14 is the per-a100 cap on the amperes (252154M with it);
            # blackwell1 is 288 cores over 8 cards.
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
