import csv
import json
import subprocess
from pathlib import Path

from roach.slurm.clusters.ilc import ILC

from roach.slurm import Resources, submit
from rt.data import get_tasks

HERE = Path(__file__).parent

PROJECT = "2026-08-25-icl"
ENTITY = "rtv2"
OUT_ROOT = "~/scratch/relational-transformer/icl"
PRE_DIR = "~/scratch/hf/stanford-star/relbench-preprocessed"

MODELS = (
    ("rt-plurel", "~/scratch/hf/stanford-star/rt-plurel"),
    ("rt-j", "~/scratch/hf/stanford-star/rt-j"),
)

TASKS = (
    ("rel-amazon", "user-churn"),
    ("rel-amazon", "user-ltv"),
    ("rel-amazon", "item-ltv"),
    ("rel-amazon", "item-churn"),
    ("rel-stack", "user-badge"),
    ("rel-stack", "post-votes"),
    ("rel-hm", "item-sales"),
    ("rel-stack", "user-engagement"),
    ("rel-hm", "user-churn"),
    ("rel-avito", "user-clicks"),
    ("rel-avito", "user-visits"),
    ("rel-trial", "site-success"),
    ("rel-trial", "study-adverse"),
    ("rel-event", "user-attendance"),
    ("rel-event", "user-ignore"),
    ("rel-avito", "ad-ctr"),
    ("rel-trial", "study-outcome"),
    ("rel-f1", "driver-position"),
    ("rel-f1", "driver-top3"),
    ("rel-f1", "driver-dnf"),
    ("rel-event", "user-repeat"),
)


def lcs_bw_pl_grid() -> list[tuple[int, int, bool]]:
    return [
        (lcs, bw, pl)
        for lcs in (256, 512, 1024, 2048, 4096, 8192)
        for bw in (8, 32, 128)
        for pl in (True, False)
    ]


def ctx_sizes() -> list[int]:
    return [512, 1024, 2048, 4096, 8192]


def reference() -> dict[str, dict[str, str]]:
    with open(HERE / "reference.csv", newline="") as f:
        return {row["task"]: row for row in csv.DictReader(f)}


def targets_for(db: str, task: str) -> dict[str, float]:
    row = reference()[f"{db}/{task}"]
    metric = {"roc_auc": "auroc", "nmae": "nmae"}[row["metric"]]
    return {
        f"{metric}/test/{db}/{task}": float(row["rt-j-icl"]),
        f"{metric}/test/{db}/{task}/fine-tuned": float(row["rt-plurel-ft"]),
    }


def checkpoint(ckpt_root: str, db: str, task: str) -> str:
    (t,) = get_tasks(PRE_DIR, [(db, task)], ("val",))
    return (
        f"{ckpt_root}/{ {'clf': 'classification', 'reg': 'regression'}[t.task_type] }"
    )


def stage_dir(run_id: str) -> Path:
    return Path(OUT_ROOT).expanduser() / ENTITY / PROJECT / run_id


def a100(qos: str, time: str) -> Resources:
    return Resources(
        partition="il",
        account="infolab",
        qos=qos,
        time=time,
        gpus="a100:1",
        cpus_per_task=14,
        ntasks=None,
        exclusive=False,
        mem=None,
        mem_per_gpu=None,
        constraint="ampere",
        nodelist=None,
        reservation=None,
        dependency=None,
        exclude="ampere4,ampere7",
    )


def b200(qos: str, time: str) -> Resources:
    return Resources(
        partition="il",
        account="infolab",
        qos=qos,
        time=time,
        gpus="b200:1",
        cpus_per_task=36,
        ntasks=None,
        exclusive=False,
        mem="375000M",
        mem_per_gpu=None,
        constraint=None,
        nodelist="blackwell1",
        reservation=None,
        dependency=None,
    )


# The rt-plurel plan is a day's log of moves against the live cluster
# (2026-08-25, git holds each): the sweep started on il-lo a100s with the
# fine_tune sweep holding every high tier; il-lo preemptions by other users'
# il bursts hit nine tuning jobs, each resuming from its per-entry state;
# the fix that held was to ask il-lo for 2-hour chunks, which backfill into
# planned-away cards where a 2-day request waits behind forty jobs, with
# roach requeueing at the wall clock; and as fine_tune wound down, its il /
# il-interactive slots and free b200s went to the jobs with the most left,
# biggest database first (a b200 is ~2.2x an a100 on this grid; a b200 under
# il still counts among il's ten). ampere4 (disk 99% full) and ampere7 (a100
# comes up with 16 MB free, CUDA OOM at the first forward) stay excluded.
# 2026-08-26 00:20, rt-j joins: fine_tune is starting its own rt-j arm on
# il-lo and asks that the b200 slots (il and il-interactive) go to it as
# mine finish, so rt-j's tuning runs on il-lo a100s in 2-hour chunks, and
# rt-plurel's remaining ensemble units keep the il a100 slots its tuning
# jobs vacate. 00:45: an il a100 freed with no rt-plurel unit waiting, so an
# rt-j tuning job (rel-hm/item-sales) took it off its chunks -- and gave it
# back two minutes later: fine_tune's longest rt-j refit needs that tenth il
# slot for a b200.
TUNE: dict[tuple[str, str, str], Resources] = {
    ("rt-plurel", "rel-amazon", "user-churn"): b200("il", "2-00:00:00"),
    ("rt-plurel", "rel-amazon", "user-ltv"): a100("il", "2-00:00:00"),
    ("rt-plurel", "rel-amazon", "item-ltv"): a100("il", "2-00:00:00"),
    ("rt-plurel", "rel-amazon", "item-churn"): a100("il", "2-00:00:00"),
    ("rt-plurel", "rel-stack", "user-badge"): b200("il-interactive", "12:00:00"),
    ("rt-plurel", "rel-stack", "post-votes"): a100("il", "2-00:00:00"),
    ("rt-plurel", "rel-hm", "item-sales"): a100("il", "2-00:00:00"),
    ("rt-plurel", "rel-stack", "user-engagement"): a100("il", "2-00:00:00"),
    ("rt-plurel", "rel-hm", "user-churn"): b200("il", "2-00:00:00"),
    ("rt-plurel", "rel-avito", "user-clicks"): a100("il", "2-00:00:00"),
    ("rt-plurel", "rel-avito", "user-visits"): b200("il-interactive", "12:00:00"),
    ("rt-plurel", "rel-trial", "site-success"): a100("il", "2-00:00:00"),
    ("rt-plurel", "rel-trial", "study-adverse"): b200("il", "12:00:00"),
    ("rt-plurel", "rel-event", "user-attendance"): a100("il-lo", "2:00:00"),
    ("rt-plurel", "rel-event", "user-ignore"): a100("il-lo", "2:00:00"),
    ("rt-plurel", "rel-avito", "ad-ctr"): a100("il-lo", "2-00:00:00"),
    ("rt-plurel", "rel-trial", "study-outcome"): a100("il-lo", "2:00:00"),
    ("rt-plurel", "rel-f1", "driver-position"): a100("il-lo", "2-00:00:00"),
    ("rt-plurel", "rel-f1", "driver-top3"): a100("il-lo", "2-00:00:00"),
    ("rt-plurel", "rel-f1", "driver-dnf"): a100("il-lo", "2-00:00:00"),
    ("rt-plurel", "rel-event", "user-repeat"): a100("il", "2-00:00:00"),
    ("rt-j", "rel-amazon", "user-churn"): a100("il-lo", "2:00:00"),
    ("rt-j", "rel-amazon", "user-ltv"): a100("il-lo", "2:00:00"),
    ("rt-j", "rel-amazon", "item-ltv"): a100("il-lo", "2:00:00"),
    ("rt-j", "rel-amazon", "item-churn"): a100("il-lo", "2:00:00"),
    ("rt-j", "rel-stack", "user-badge"): a100("il-lo", "2:00:00"),
    ("rt-j", "rel-stack", "post-votes"): a100("il-lo", "2:00:00"),
    ("rt-j", "rel-hm", "item-sales"): a100("il-lo", "2:00:00"),
    ("rt-j", "rel-stack", "user-engagement"): a100("il-lo", "2:00:00"),
    ("rt-j", "rel-hm", "user-churn"): a100("il-lo", "2:00:00"),
    ("rt-j", "rel-avito", "user-clicks"): a100("il-lo", "2:00:00"),
    ("rt-j", "rel-avito", "user-visits"): a100("il-lo", "2:00:00"),
    ("rt-j", "rel-trial", "site-success"): a100("il-lo", "2:00:00"),
    ("rt-j", "rel-trial", "study-adverse"): a100("il-lo", "2:00:00"),
    ("rt-j", "rel-event", "user-attendance"): a100("il-lo", "2:00:00"),
    ("rt-j", "rel-event", "user-ignore"): a100("il-lo", "2:00:00"),
    ("rt-j", "rel-avito", "ad-ctr"): a100("il-lo", "2:00:00"),
    ("rt-j", "rel-trial", "study-outcome"): a100("il-lo", "2:00:00"),
    ("rt-j", "rel-f1", "driver-position"): a100("il-lo", "2:00:00"),
    ("rt-j", "rel-f1", "driver-top3"): a100("il-lo", "2:00:00"),
    ("rt-j", "rel-f1", "driver-dnf"): a100("il-lo", "2:00:00"),
    ("rt-j", "rel-event", "user-repeat"): a100("il-lo", "2:00:00"),
}

# The ensemble units are submitted once tuned_configs.json holds a task, one
# unit per (task, config), placed per rank: the big tasks' units are hours
# (four full test passes; a pass over rel-amazon's 352k rows is ~3h on an
# a100 at ctx 8192, and a preemption loses the pass in flight), so they take
# il / il-interactive slots as tuning jobs vacate them; a task under 5k test
# rows finishes in minutes and asks il-lo for 2 hours so it backfills.
# 01:45: user-badge's il-lo a100 unit was preempted after 3h with one seed
# saved; it goes back at 6h (three passes left) so it can backfill. 01:55:
# site-success tuned and its il a100 freed; fine_tune holds all four b200
# slots and wants nothing more, so the slot goes to user-badge's waiting
# unit, and site-success' four units (23k rows, ~1h each) to il-lo at 3h.
# 02:15: rel-amazon/user-ltv and item-ltv tuned at ctx 512-1024, so their
# units are ~2-3h on an a100, not 13h: il-lo at 6h (fine_tune took the tenth
# il slot for its longest refit). 02:20: an il slot is free after all;
# hm/user-churn's cfg1, preempted at 01:43 and still queued, takes it. 02:25:
# item-churn tuned; its cfg0 takes the il slot its tuning job vacated, the
# rest go to il-lo at 6h. 02:30-02:40: user-engagement, item-sales,
# post-votes and user-clicks tuned -- rt-plurel's grid is complete -- same
# placement. 03:00: as il slots free, the longest queued unit moves up:
# item-churn cfg1 (ctx 8192, ~6h) first, then item-sales cfg3 (8192, ~4h),
# user-engagement cfg3 (8192, ~3h), item-sales cfg2 (4096, ~2h),
# user-engagement cfg2 (4096), item-churn cfg3 (2048, 167k rows) -- which
# went back to il-lo minutes later: fine_tune wanted that il slot for a b200.
ENS: dict[tuple[str, str, str], list[Resources]] = {
    ("rt-plurel", "rel-amazon", "user-churn"): [
        b200("il", "1-00:00:00"),
        a100("il-lo", "1-00:00:00"),
        a100("il-lo", "1-00:00:00"),
        a100("il-lo", "1-00:00:00"),
    ],
    ("rt-plurel", "rel-amazon", "user-ltv"): [a100("il-lo", "6:00:00")] * 4,
    ("rt-plurel", "rel-amazon", "item-ltv"): [a100("il-lo", "6:00:00")] * 4,
    ("rt-plurel", "rel-amazon", "item-churn"): [
        a100("il", "12:00:00"),
        a100("il", "12:00:00"),
        a100("il-lo", "6:00:00"),
        a100("il-lo", "6:00:00"),
    ],
    ("rt-plurel", "rel-stack", "user-badge"): [
        b200("il-interactive", "12:00:00"),
        b200("il-lo", "12:00:00"),
        b200("il-interactive", "12:00:00"),
        a100("il", "12:00:00"),
    ],
    ("rt-plurel", "rel-stack", "post-votes"): [
        a100("il", "12:00:00"),
        a100("il-lo", "6:00:00"),
        a100("il-lo", "6:00:00"),
        a100("il-lo", "6:00:00"),
    ],
    ("rt-plurel", "rel-hm", "item-sales"): [
        a100("il", "12:00:00"),
        a100("il-lo", "6:00:00"),
        a100("il", "12:00:00"),
        a100("il", "12:00:00"),
    ],
    ("rt-plurel", "rel-stack", "user-engagement"): [
        a100("il", "12:00:00"),
        a100("il-lo", "6:00:00"),
        a100("il", "12:00:00"),
        a100("il", "12:00:00"),
    ],
    ("rt-plurel", "rel-hm", "user-churn"): [
        b200("il", "12:00:00"),
        a100("il", "6:00:00"),
        a100("il-lo", "6:00:00"),
        a100("il-lo", "6:00:00"),
    ],
    ("rt-plurel", "rel-avito", "user-clicks"): [
        a100("il", "12:00:00"),
        a100("il-lo", "6:00:00"),
        a100("il-lo", "6:00:00"),
        a100("il-lo", "6:00:00"),
    ],
    ("rt-plurel", "rel-avito", "user-visits"): [a100("il-lo", "3:00:00")] * 4,
    ("rt-plurel", "rel-trial", "site-success"): [a100("il-lo", "3:00:00")] * 4,
    ("rt-plurel", "rel-trial", "study-adverse"): [a100("il-lo", "2:00:00")] * 4,
    ("rt-plurel", "rel-event", "user-attendance"): [a100("il-lo", "2:00:00")] * 4,
    ("rt-plurel", "rel-event", "user-ignore"): [a100("il-lo", "2:00:00")] * 4,
    ("rt-plurel", "rel-avito", "ad-ctr"): [a100("il-lo", "2:00:00")] * 4,
    ("rt-plurel", "rel-trial", "study-outcome"): [a100("il-lo", "2:00:00")] * 4,
    ("rt-plurel", "rel-f1", "driver-position"): [a100("il-lo", "2:00:00")] * 4,
    ("rt-plurel", "rel-f1", "driver-top3"): [a100("il-lo", "2:00:00")] * 4,
    ("rt-plurel", "rel-f1", "driver-dnf"): [a100("il-lo", "2:00:00")] * 4,
    ("rt-plurel", "rel-event", "user-repeat"): [a100("il-lo", "2:00:00")] * 4,
    ("rt-j", "rel-amazon", "user-churn"): [a100("il-lo", "1-00:00:00")] * 4,
    ("rt-j", "rel-amazon", "user-ltv"): [a100("il-lo", "1-00:00:00")] * 4,
    ("rt-j", "rel-amazon", "item-ltv"): [a100("il-lo", "1-00:00:00")] * 4,
    ("rt-j", "rel-amazon", "item-churn"): [a100("il-lo", "1-00:00:00")] * 4,
    ("rt-j", "rel-stack", "user-badge"): [a100("il-lo", "1-00:00:00")] * 4,
    ("rt-j", "rel-stack", "post-votes"): [a100("il-lo", "1-00:00:00")] * 4,
    ("rt-j", "rel-hm", "item-sales"): [a100("il-lo", "1-00:00:00")] * 4,
    ("rt-j", "rel-stack", "user-engagement"): [a100("il-lo", "1-00:00:00")] * 4,
    ("rt-j", "rel-hm", "user-churn"): [a100("il-lo", "1-00:00:00")] * 4,
    ("rt-j", "rel-avito", "user-clicks"): [a100("il-lo", "1-00:00:00")] * 4,
    ("rt-j", "rel-avito", "user-visits"): [a100("il-lo", "1-00:00:00")] * 4,
    ("rt-j", "rel-trial", "site-success"): [a100("il-lo", "1-00:00:00")] * 4,
    ("rt-j", "rel-trial", "study-adverse"): [a100("il-lo", "2:00:00")] * 4,
    ("rt-j", "rel-event", "user-attendance"): [a100("il-lo", "2:00:00")] * 4,
    ("rt-j", "rel-event", "user-ignore"): [a100("il-lo", "2:00:00")] * 4,
    ("rt-j", "rel-avito", "ad-ctr"): [a100("il-lo", "2:00:00")] * 4,
    ("rt-j", "rel-trial", "study-outcome"): [a100("il-lo", "2:00:00")] * 4,
    ("rt-j", "rel-f1", "driver-position"): [a100("il-lo", "2:00:00")] * 4,
    ("rt-j", "rel-f1", "driver-top3"): [a100("il-lo", "2:00:00")] * 4,
    ("rt-j", "rel-f1", "driver-dnf"): [a100("il-lo", "2:00:00")] * 4,
    ("rt-j", "rel-event", "user-repeat"): [a100("il-lo", "2:00:00")] * 4,
}


def queued() -> set[str]:
    out = subprocess.run(
        ["squeue", "-h", "-u", "ranjanr", "-o", "%j"],
        capture_output=True,
        text=True,
        check=True,
    )
    return set(out.stdout.split())


def main() -> None:
    busy = queued()
    tuned_path = HERE / "tuned_configs.json"
    tuned = json.loads(tuned_path.read_text()) if tuned_path.exists() else {}
    for model, ckpt_root in MODELS:
        for db, task in TASKS:
            run_id = f"tune-{model}-{db}-{task}"
            name = f"icl-{run_id}"
            if name in busy:
                print(f"  {name:48s} queued already")
                continue
            if (stage_dir(run_id) / "tuning.json").exists():
                print(f"  {name:48s} done already")
                continue
            resources = TUNE[model, db, task]
            print(f"  {name:48s} {resources.gpus} {resources.qos:15s} {resources.time}")
            submit(
                "rt.eval:main",
                args=dict(
                    load_ckpt_path=checkpoint(ckpt_root, db, task),
                    embedder="all-MiniLM-L12-v2",
                    d_text=384,
                    num_blocks=12,
                    d_model=512,
                    num_heads=8,
                    d_ff=2048,
                    splits=["val"],
                    db_task_list=[(db, task)],
                    pre_dir=PRE_DIR,
                    tokens_per_gpu=2**18,
                    num_workers=resources.cpus_per_task,
                    prefetch_factor=2,
                    num_walks=10_000,
                    walk_length=20,
                    val_items_per_task=4096,
                    test_items_per_task=None,
                    ctx_size_list=ctx_sizes(),
                    mmap_populate=True,
                    shuffle_seed=0,
                    context_seed=0,
                    vector_db_path=None,
                    db_cutoff=None,
                    lcs_bw_pl_grid=lcs_bw_pl_grid(),
                    val_ensemble_size=4,
                    test_ensemble_size=1,
                    run_name=f"{model}/{db}/{task}/tune",
                    targets=targets_for(db, task),
                    project=PROJECT,
                    entity=ENTITY,
                    out_root=OUT_ROOT,
                    wandb_disabled=False,
                ),
                resources=resources,
                name=name,
                run_id=run_id,
                repo_root=str(HERE.parents[1]),
                cluster=ILC,
                job_env="expts/job_env.sh",
                log_root=f"{OUT_ROOT}/slurm-logs",
                clone_root="~/roach_clones",
                secrets_dir="~/scratch/.secrets",
            )

        for db, task in TASKS:
            rec = tuned.get(model, {}).get(f"{db}/{task}")
            if rec is None:
                print(f"  {model}/{db}/{task}: not in tuned_configs.json yet")
                continue
            for rank, (ctx, lcs, bw, pl) in enumerate(rec["top_cfgs"]):
                run_id = f"ens-{model}-{db}-{task}-cfg{rank}"
                name = f"icl-{run_id}"
                if name in busy:
                    print(f"  {name:48s} queued already")
                    continue
                if (stage_dir(run_id) / "result.json").exists():
                    print(f"  {name:48s} done already")
                    continue
                resources = ENS[model, db, task][rank]
                print(
                    f"  {name:48s} {resources.gpus} {resources.qos:15s} {resources.time}"
                    f"  {(ctx, lcs, bw, pl)}"
                )
                submit(
                    "expts.icl.run:main",
                    args=dict(
                        db=db,
                        task=task,
                        load_ckpt_path=checkpoint(ckpt_root, db, task),
                        pre_dir=PRE_DIR,
                        ctx_size=int(ctx),
                        local_ctx_size=int(lcs),
                        bfs_width=int(bw),
                        prefer_latest=bool(pl),
                        n_seeds=4,
                        items_per_task=1_000_000_000,
                        num_walks=10_000,
                        walk_length=20,
                        shuffle_seed=0,
                        context_seed=0,
                        tokens_per_gpu=2**18,
                        num_workers=resources.cpus_per_task,
                        prefetch_factor=2,
                        mmap_populate=True,
                        db_cutoff=None,
                        run_name=f"{model}/{db}/{task}/cfg{rank}",
                        targets=targets_for(db, task),
                        project=PROJECT,
                        entity=ENTITY,
                        out_root=OUT_ROOT,
                        wandb_disabled=False,
                    ),
                    resources=resources,
                    name=name,
                    run_id=run_id,
                    repo_root=str(HERE.parents[1]),
                    cluster=ILC,
                    job_env="expts/job_env.sh",
                    log_root=f"{OUT_ROOT}/slurm-logs",
                    clone_root="~/roach_clones",
                    secrets_dir="~/scratch/.secrets",
                )


if __name__ == "__main__":
    main()
