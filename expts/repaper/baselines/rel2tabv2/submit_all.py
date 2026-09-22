import json
import os
import subprocess
from pathlib import Path

from roach.slurm import Resources, submit
from roach.slurm.clusters.ilc import ILC

from expts.repaper.config import (
    CLONE_ROOT,
    LOG_ROOT,
    OUT_ROOT,
    PRE_DIR,
    SECRETS_DIR,
    SHARE,
)

# {rdblearn, rt-j, rt-plurel, gnn} x {exaone, tabpfn} on all 21 forecast tasks,
# one configuration, no grid: the query-independent retriever drawing exactly
# 1024 train rows with the uniform sampler at seed 0.
#
# Its own round directory rather than global_uniform_seed0_n1024, which submit.py
# is concurrently filling with a ctx grid: run.main returns early when its json
# exists, so two submitters sharing a round and disagreeing about ctx would each
# silently skip the other's work. The feature blobs are shared; the results are
# not.
ROUND = "n1024_4feat_2pred"
N_ROWS = [1024]
DB_TASK_LIST = f"{PRE_DIR}/db-task-lists/forecast.json"
TASKS = [tuple(p) for p in json.loads(Path(DB_TASK_LIST).expanduser().read_text())]

# arm -> (method, features_root, blob subdir, featurizer label, blob is
# bit-exactly regenerable)
GNN_ROOT = f"{SHARE}/features_gnn-c512-s0"
ARMS = {
    "rdblearn-exaone": (
        "rdblearn_exaone",
        f"{SHARE}/features",
        "rdblearn_features",
        "rdblearn",
        True,
    ),
    "rdblearn-tabpfn": (
        "rdblearn_tabpfn",
        f"{SHARE}/features",
        "rdblearn_features",
        "rdblearn",
        True,
    ),
    "rt-j-exaone": ("rt_exaone", f"{SHARE}/features_rt-j", "rt_features", "rt-j", True),
    "rt-j-tabpfn": ("rt_tabpfn", f"{SHARE}/features_rt-j", "rt_features", "rt-j", True),
    "rt-plurel-exaone": (
        "rt_exaone",
        f"{SHARE}/features_rt-plurel",
        "rt_features",
        "rt-plurel",
        True,
    ),
    "rt-plurel-tabpfn": (
        "rt_tabpfn",
        f"{SHARE}/features_rt-plurel",
        "rt_features",
        "rt-plurel",
        True,
    ),
    # 512 channels, seed 0, produced by submit_featurize_gnn_all.py. NOT
    # regenerable bit-exactly -- pyg-lib's per-thread sampler engines mean the
    # blob depends on thread scheduling -- so the blob is the artifact of record
    # and collate.py reports its sha256. The False here is what carries that into
    # every result's `tags`.
    "gnn-exaone": ("gnn_exaone", GNN_ROOT, "gnn_features", "gnn-c512-s0", False),
    "gnn-tabpfn": ("gnn_tabpfn", GNN_ROOT, "gnn_features", "gnn-c512-s0", False),
    # Zero hops: the entity table's own encoded row, plus the entity id and the
    # cutoff. The floor the other three featurizers are measured against, and
    # deterministic -- it reads the same materialization the gnn arm does and
    # gathers one row per task row.
    "entity-exaone": (
        "entity_exaone",
        f"{SHARE}/features_entity",
        "entity_features",
        "entity",
        True,
    ),
    "entity-tabpfn": (
        "entity_tabpfn",
        f"{SHARE}/features_entity",
        "entity_features",
        "entity",
        True,
    ),
}

# The run loads the whole blob into memory: rel-amazon at 512 wide is ~5.4 M rows
# x 512 x 4 B = 11 G, on top of the preprocessed mmap. Sized by db, not by arm.
BIG = {"rel-amazon", "rel-hm", "rel-stack"}

REPO_ROOT = str(Path(__file__).resolve().parents[4])


def blob_ready(features_root: str, subdir: str, db: str, table: str) -> bool:
    feat_dir = Path(features_root).expanduser() / db / subdir
    return (feat_dir / f"{table}_meta.json").is_file() and (
        feat_dir / f"{table}_vectors.bin"
    ).is_file()


def done(arm: str, db: str, table: str) -> bool:
    out = Path(OUT_ROOT).expanduser() / f"repaper-rel2tabv2/{ROUND}/{arm}"
    return (out / f"{db}__{table}.pkl").is_file()


def queued() -> set[str]:
    out = subprocess.run(
        ["squeue", "-h", "-u", os.environ["USER"], "-o", "%j"],
        capture_output=True,
        text=True,
        check=True,
    )
    return set(out.stdout.split())


busy = queued()
skipped_no_blob = []

# Backfill under a two-a100 budget. entity-exaone is the only arm with cells
# left, and EXAONE cannot go to the idle rtx8000s -- its attention calls
# F.scaled_dot_product_attention, which has no sm_75 kernel (the rel-event wave
# lost 64 jobs to exactly that). So the four run two at a time: 192955 holds one
# slot, the first submission below takes the other, and the rest wait on a
# predecessor so nothing ever makes a third.
#
# Set to None to submit normally again.
CHAIN_AFTER = {
    ("rel-hm", "item-sales"): None,
    ("rel-stack", "post-votes"): "192955",
    ("rel-trial", "site-success"): ("rel-hm", "item-sales"),
}
chained: dict[tuple[str, str], str] = {}

for db, table in TASKS:
    for arm, (
        method,
        features_root,
        subdir,
        featurizer,
        regenerable,
    ) in ARMS.items():
        name = f"rel2tabv2-{ROUND}-{arm}-{db}-{table}"
        if name in busy or done(arm, db, table):
            continue
        # Not an error: the gnn blobs arrive db by db. Rerun this file as they
        # land and only the newly possible runs are queued.
        if not blob_ready(features_root, subdir, db, table):
            skipped_no_blob.append(f"{arm}/{db}/{table}")
            continue
        after = CHAIN_AFTER.get((db, table), None) if CHAIN_AFTER else None
        if isinstance(after, tuple):
            after = chained[after]
        job = submit(
            "expts.repaper.baselines.rel2tabv2.run:main",
            args=dict(
                method=method,
                db=db,
                table=table,
                split="test",
                pre_dir=PRE_DIR,
                features_root=features_root,
                out_dir=f"{OUT_ROOT}/repaper-rel2tabv2/{ROUND}/{arm}",
                ctx_size_list=N_ROWS,
                # What the evaluator is told, not what the predictor gets: the
                # global retriever discards the evaluator's per-query context,
                # and the sampler pads every batch to this width. 64 matches
                # local_ctx_size, so nothing is padded and nothing is wasted.
                sampler_ctx_size=64,
                items_per_task=10_000_000,
                # the sampler only says which rows are queries and what their
                # labels are; its context is discarded, so num_walks=0 and a
                # small local_ctx_size keep it from doing work nobody reads
                local_ctx_size=64,
                bfs_width=32,
                prefer_latest=True,
                num_walks=0,
                walk_length=0,
                shuffle_seed=0,
                context_seed=0,
                # eval_bs = tokens_per_gpu // sampler_ctx_size, so this is
                # 256 query rows per batch regardless of the context size.
                tokens_per_gpu=64 * 256,
                num_workers=8,
                prefetch_factor=2,
                mmap_populate=True,
                db_cutoff=None,
                vector_db_path=None,
                tabicl_dir=f"{SHARE}/tabicl",
                tabicl_max_batch_size=1024,
                tabicl_min_bin_size=48,
                tabicl_softmax_temperature=0.9,
                lgbm_n_jobs=8,
                exaone_ensemble_count=8,
                tabfm_backend="pytorch",
                tabpfn_dir=f"{SHARE}/tabpfn",
                tabpfn_n_estimators="auto",
                tabpfn_fit_mode="fit_with_cache",
                retriever="global_train",
                context_sampler="uniform",
                context_split="train",
                tags=dict(
                    round=ROUND,
                    arm=arm,
                    featurizer=featurizer,
                    predictor=method.rsplit("_", 1)[1],
                    # False for gnn: collate.py turns this into the provenance
                    # note, and it is recorded per result rather than inferred
                    # from the arm name later.
                    features_regenerable=regenerable,
                    gnn_channels=512 if featurizer.startswith("gnn") else None,
                    hops=0 if featurizer == "entity" else None,
                ),
            ),
            resources=Resources(
                partition="il",
                account="infolab",
                qos="il",
                time="12:00:00" if db in BIG else "4:00:00",
                # a100, and never rtx8000 or 2080ti: EXAONE has no SDPA kernel
                # on sm_75. b200 is capped at 2 for the whole il partition by
                # QoS=il-part, so it is not worth routing a 168-run sweep to.
                gpus="a100:1",
                cpus_per_task=8,
                ntasks=None,
                exclusive=False,
                mem="120G" if db in BIG else "64G",
                mem_per_gpu=None,
                constraint="ampere",
                nodelist=None,
                reservation=None,
                dependency=None,
                exclude="ampere4,ampere6,ampere7,ampere9",
            ),
            name=name,
            repo_root=REPO_ROOT,
            cluster=ILC,
            job_env="expts/job_env.sh",
            log_root=f"{LOG_ROOT}/repaper/baselines/slurm-logs",
            clone_root=CLONE_ROOT,
            secrets_dir=SECRETS_DIR,
            after=after,
        )
        chained[(db, table)] = job.id

if skipped_no_blob:
    print(f"\n{len(skipped_no_blob)} runs waiting on a blob:")
    for s in skipped_no_blob:
        print(f"  {s}")
