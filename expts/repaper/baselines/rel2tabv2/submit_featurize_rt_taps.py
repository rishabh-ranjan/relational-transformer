from pathlib import Path

from roach.slurm import Resources, submit
from roach.slurm.clusters.ilc import ILC

from expts.repaper.config import (
    CKPT,
    CLONE_ROOT,
    LOG_ROOT,
    PRE_DIR,
    SECRETS_DIR,
    SHARE,
)

# The multi-tap, multi-cell rt-j dump. Per row of a task table it keeps the
# representation of every column in the seed row's union set -- the task row's
# own cells plus the cells of the entity row its f2p link reaches -- at the end
# of relational units 1, 4, 8 and 12. The target cell is included; its unit-12
# vector is what features_rt-j already holds, verified bit-for-bit by
# probe_rt_taps (jobs 190155-190157).
#
# Raw residual stream, no norm_out: a tap is then comparable to itself across
# depths, and the choice of normalisation stays with the consumer.
#
# Layout per task, so "the target column at unit 4" is one index:
#     <table>_taps.bin   bfloat16 (total_nodes, n_taps, n_slots, 512)
#     <table>_meta.json  taps, ordered slot names, target_slot, min_offset,
#                        per-slot fill counts, dead slots
TAP_UNITS = [1, 4, 8, 12]

# The 12 tasks under 200k rows. The other nine are rel-amazon, rel-hm and
# rel-stack, where this dump runs to 2.5 TB against 690 GB free -- they are out
# of scope until something narrows it. Row counts are total_nodes (all splits),
# which is what the blob is indexed by.
#
# rel-f1 was smoked first (job 190200): 0.92 GiB written against 0.92 GiB
# predicted, 2 dead slots per table, 0 duplicate writes. Its unit-12 target
# slot reproduces features_rt-j on all but 46 of 6.49M elements, and those
# differ by a bf16 rounding of the pre-scale value between cpu and gpu --
# accepted. featurize_db_taps skips a table whose blob and meta already exist,
# so rel-f1 is a no-op on this pass.
TASKS_BY_DB = {
    "rel-avito": ["ad-ctr", "user-clicks", "user-visits"],
    "rel-event": ["user-attendance", "user-ignore", "user-repeat"],
    "rel-f1": ["driver-dnf", "driver-position", "driver-top3"],
    "rel-trial": ["site-success", "study-adverse", "study-outcome"],
}
# total_nodes x 4 taps x n_slots x 512 x 2 bytes, summed over each db's
# tasks, computed against the real slot lists union_slots() produces (9-31
# slots; keys are kept as slots and come out all-NaN). 27.0 GiB in total
# against 690 GiB free. The job asserts on both the free space before a table
# and the running total after one, so an overrun stops the sweep rather than
# being found afterwards.
EXPECTED_GIB = {
    "rel-avito": 9.96,
    "rel-event": 1.95,
    "rel-f1": 0.92,
    "rel-trial": 14.20,
}

DB_TASK_LIST = f"{PRE_DIR}/db-task-lists/forecast.json"
REPO_ROOT = str(Path(__file__).resolve().parents[4])

for db, tables in TASKS_BY_DB.items():
    submit(
        "expts.repaper.baselines.featurize_rt_taps:featurize_db_taps",
        args=dict(
            db=db,
            tables=tables,
            db_task_list=DB_TASK_LIST,
            pre_dir=PRE_DIR,
            features_root=f"{SHARE}/features_rt-j",
            ckpt=CKPT,
            tap_units=TAP_UNITS,
            # Identical to the pass that built features_rt-j. Changing either
            # would make the unit-12 target slot stop matching that blob, which
            # is the only external check this dump has.
            local_ctx_size=256,
            bfs_width=32,
            shuffle_seed=0,
            context_seed=0,
            db_cutoff=None,
            batch_size=1024,
            expected_gib=EXPECTED_GIB[db],
        ),
        resources=Resources(
            partition="il",
            account="infolab",
            qos="il",
            time="8:00:00",
            gpus="a100:1",
            cpus_per_task=8,
            ntasks=None,
            exclusive=False,
            mem="64G",
            mem_per_gpu=None,
            constraint="ampere",
            nodelist=None,
            reservation=None,
            dependency=None,
            exclude="ampere4",
        ),
        name=f"rel2tabv2-feat-taps-{db}",
        repo_root=REPO_ROOT,
        cluster=ILC,
        job_env="expts/job_env.sh",
        log_root=f"{LOG_ROOT}/repaper/baselines/slurm-logs",
        clone_root=CLONE_ROOT,
        secrets_dir=SECRETS_DIR,
    )
