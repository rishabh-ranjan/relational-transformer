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
CKPT_ROOT = "~/scratch/hf/stanford-star/rt-plurel"

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


def checkpoint(db: str, task: str) -> str:
    (t,) = get_tasks(PRE_DIR, [(db, task)], ("val",))
    return (
        f"{CKPT_ROOT}/{ {'clf': 'classification', 'reg': 'regression'}[t.task_type] }"
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


# 2026-08-25 11:50, read off the cluster: the fine_tune sweep holds
# il-interactive 2/2 (b200) and il 9/10 (8 a100 + 1 b200); blackwell1 is 8/8
# allocated (5 of them il-lo jobs of another user with up to 40h left), ~17
# a100 free across ampere2/3/5/6/8/9 with four 8-gpu il-lo jobs of other users
# queued for whole nodes. 12:05: the one free il slot is fine_tune's to keep
# (the human's instruction), so every tuning job runs on il-lo a100; as
# fine_tune jobs finish, freed il / il-interactive slots go to the rel-amazon
# jobs first (the biggest database, ~6h15 on a b200 for the same grid in the
# RT-J rerun), then down the list. ampere4 (disk 99% full) and ampere7 (a100
# comes up with 16 MB free, CUDA OOM at the first forward) stay excluded, as
# in fine_tune. 13:45: fine_tune's last 11 jobs are all running and two il
# a100 slots stand free, so the two tuning jobs still pending on il-lo take
# them (the human's call). 13:50: fine_tune's il b200 job finished and a b200
# stands free, so the biggest database goes there under il (b200 is ~2.2x an
# a100 on this grid). 14:55: fine_tune needs that il b200 slot back for a
# refit (il allows 2 b200 per user and fine_tune holds the other), so the job
# moves to an il a100 by its run_id, losing at most one grid entry. 15:05:
# rel-stack/user-engagement was preempted off ampere9 after 3h (6 grid
# entries saved) and would requeue at the back of the il-lo queue; il has a
# slot free, so it resumes there, ahead of every il-lo job for the next card.
# 15:10: user-repeat's il slot freed; rel-amazon/user-ltv, the il-lo tuning job
# with the most left (~9h), moves onto it -- one grid entry lost now against
# a preemption that would also requeue it behind a dozen il-lo jobs. 15:40:
# rel-amazon/item-ltv preempted off ampere5 after 3h37 (the second il-lo
# preemption today); it resumes on the il slot that is free. 15:45: the third,
# rel-hm/item-sales off ampere8 after 3h40; il is 10/10 and il-lo has 40
# single-gpu jobs of other users queued ahead, so it takes the il-interactive
# slot that stands free (12h, ~9h of grid left). 15:48: that slot blocked a
# fine_tune il-interactive job (QOSMaxGRESPerUser), and two il slots had just
# freed, so it moves again, to il. 15:55: the fine_tune session says nothing
# more of its will be promoted, so the il room is this sweep's; the il-lo
# tuning jobs with the most left move up as slots free, biggest database
# first, each losing at most one grid entry. 16:05: five il-lo tuning jobs
# preempted by another user's il burst sit at priority ~760 behind 40 il-lo
# jobs while 2-hour ensemble units sail past them: the free cards are planned
# for a big pending job and only a job that ends before it backfills in. So
# the requeued jobs ask for 2 hours; roach requeues them at the wall clock
# and they resume per grid entry, losing at most one entry per chunk. 16:55:
# an il slot freed; rel-avito/user-clicks, chunked with ~7h left, takes it.
# 17:35: four more preempted off ampere5/6 (user-attendance and user-ignore
# with ~1h left, user-badge and post-votes with ~6h); il is full, so they
# go into 2-hour chunks too. 18:20: an il slot freed while two chunks sat
# in the queue after their wall clock; rel-hm/user-churn takes it. 18:30: a
# fine_tune il-interactive job ended with nothing of theirs waiting, and a
# b200 is free: rel-avito/user-visits' chunk goes there (12h, ~3h of grid on
# a b200); it vacates on request. 18:40: fine_tune's last il-interactive job
# ended and another b200 is free; rel-stack/user-badge (2h chunks, ~6h of grid
# left on an a100, the heaviest ensemble stage after rel-amazon) takes it.
# 18:55: a third b200 is free and il's b200 sub-cap has room, so
# rel-amazon/user-churn (the most left of the il a100 jobs, ~9h) swaps its
# a100 for it -- the same tier, twice the speed, one grid entry lost (and no
# il slot freed: a b200 under il is still one of the ten -- post-votes, sent
# to il on that misreading, sat on QOSMaxGRESPerUser and went back to chunks).
# 20:00: fine_tune's last il job ended; post-votes takes the slot for real.
# 20:08: another b200 free with il's b200 sub-cap at 1/2: rel-hm/user-churn
# (the il a100 job with the most left, ~9h) swaps onto it. 20:20: il is
# 9/10 (all mine); rel-trial/site-success, the last 2-day il-lo job (~2h
# left, and a preemption would strand it behind the il-lo queue), moves up.
TUNE: dict[tuple[str, str], Resources] = {
    ("rel-amazon", "user-churn"): b200("il", "2-00:00:00"),
    ("rel-amazon", "user-ltv"): a100("il", "2-00:00:00"),
    ("rel-amazon", "item-ltv"): a100("il", "2-00:00:00"),
    ("rel-amazon", "item-churn"): a100("il", "2-00:00:00"),
    ("rel-stack", "user-badge"): b200("il-interactive", "12:00:00"),
    ("rel-stack", "post-votes"): a100("il", "2-00:00:00"),
    ("rel-hm", "item-sales"): a100("il", "2-00:00:00"),
    ("rel-stack", "user-engagement"): a100("il", "2-00:00:00"),
    ("rel-hm", "user-churn"): b200("il", "2-00:00:00"),
    ("rel-avito", "user-clicks"): a100("il", "2-00:00:00"),
    ("rel-avito", "user-visits"): b200("il-interactive", "12:00:00"),
    ("rel-trial", "site-success"): a100("il", "2-00:00:00"),
    ("rel-trial", "study-adverse"): a100("il-lo", "2:00:00"),
    ("rel-event", "user-attendance"): a100("il-lo", "2:00:00"),
    ("rel-event", "user-ignore"): a100("il-lo", "2:00:00"),
    ("rel-avito", "ad-ctr"): a100("il-lo", "2-00:00:00"),
    ("rel-trial", "study-outcome"): a100("il-lo", "2:00:00"),
    ("rel-f1", "driver-position"): a100("il-lo", "2-00:00:00"),
    ("rel-f1", "driver-top3"): a100("il-lo", "2-00:00:00"),
    ("rel-f1", "driver-dnf"): a100("il-lo", "2-00:00:00"),
    ("rel-event", "user-repeat"): a100("il", "2-00:00:00"),
}

# The ensemble units are submitted once tuned_configs.json holds a task; this
# plan is rewritten against the cluster at that moment. TASKS is in test-row
# order, which is the cost order of this stage (four full test passes per unit):
# the two rel-amazon user tasks are ~13h per unit on an a100 at ctx 8192, the
# rel-f1 and rel-event tasks minutes. 14:45: an il a100 request is planned
# 8h out (Priority, behind an 8-gpu il job) although ampere8 has 4 free cards,
# and would preempt an il-lo job -- likely one of my own tuning jobs -- to
# start sooner; a 2-day il-lo request cannot backfill into those cards either.
# So the units of the tasks that finish in minutes (< 5k test rows) ask il-lo
# for 2 hours, which backfills into any gap; the big tasks keep 2 days.
# 22:35 (user-badge tuned, the first big task): a unit is placed on its own
# -- one per rank -- because the safe tiers are scarce: user-badge's four
# passes over 255k rows are ~9h on an a100 and ~4.5h on a b200, so cfg0
# takes the il-interactive slot its tuning job vacated (b200), cfg1/cfg2 the
# two b200s standing free under il-lo, cfg3 an il-lo a100 at 12h (a limit
# that fits the work and can still backfill); il-lo units move up to il /
# il-interactive as tuning jobs vacate those.
ENS: dict[tuple[str, str], list[Resources]] = {
    ("rel-amazon", "user-churn"): [a100("il-lo", "2-00:00:00")] * 4,
    ("rel-amazon", "user-ltv"): [a100("il-lo", "2-00:00:00")] * 4,
    ("rel-amazon", "item-ltv"): [a100("il-lo", "2-00:00:00")] * 4,
    ("rel-amazon", "item-churn"): [a100("il-lo", "2-00:00:00")] * 4,
    ("rel-stack", "user-badge"): [
        b200("il-interactive", "12:00:00"),
        b200("il-lo", "12:00:00"),
        b200("il-lo", "12:00:00"),
        a100("il-lo", "12:00:00"),
    ],
    ("rel-stack", "post-votes"): [a100("il-lo", "2-00:00:00")] * 4,
    ("rel-hm", "item-sales"): [a100("il-lo", "2-00:00:00")] * 4,
    ("rel-stack", "user-engagement"): [a100("il-lo", "2-00:00:00")] * 4,
    ("rel-hm", "user-churn"): [a100("il-lo", "2-00:00:00")] * 4,
    ("rel-avito", "user-clicks"): [a100("il-lo", "2-00:00:00")] * 4,
    ("rel-avito", "user-visits"): [a100("il-lo", "2-00:00:00")] * 4,
    ("rel-trial", "site-success"): [a100("il-lo", "2-00:00:00")] * 4,
    ("rel-trial", "study-adverse"): [a100("il-lo", "2:00:00")] * 4,
    ("rel-event", "user-attendance"): [a100("il-lo", "2:00:00")] * 4,
    ("rel-event", "user-ignore"): [a100("il-lo", "2:00:00")] * 4,
    ("rel-avito", "ad-ctr"): [a100("il-lo", "2:00:00")] * 4,
    ("rel-trial", "study-outcome"): [a100("il-lo", "2:00:00")] * 4,
    ("rel-f1", "driver-position"): [a100("il-lo", "2:00:00")] * 4,
    ("rel-f1", "driver-top3"): [a100("il-lo", "2:00:00")] * 4,
    ("rel-f1", "driver-dnf"): [a100("il-lo", "2:00:00")] * 4,
    ("rel-event", "user-repeat"): [a100("il-lo", "2:00:00")] * 4,
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
    for db, task in TASKS:
        run_id = f"tune-{db}-{task}"
        name = f"icl-{run_id}"
        if name in busy:
            print(f"  {name:44s} queued already")
            continue
        if (stage_dir(run_id) / "tuning.json").exists():
            print(f"  {name:44s} done already")
            continue
        resources = TUNE[db, task]
        print(f"  {name:44s} {resources.gpus} {resources.qos:15s} {resources.time}")
        submit(
            "rt.eval:main",
            args=dict(
                load_ckpt_path=checkpoint(db, task),
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
                run_name=f"{db}/{task}/tune",
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

    tuned_path = HERE / "tuned_configs.json"
    tuned = json.loads(tuned_path.read_text()) if tuned_path.exists() else {}
    for db, task in TASKS:
        rec = tuned.get(f"{db}/{task}")
        if rec is None:
            print(f"  {db}/{task}: not in tuned_configs.json yet")
            continue
        for rank, (ctx, lcs, bw, pl) in enumerate(rec["top_cfgs"]):
            run_id = f"ens-{db}-{task}-cfg{rank}"
            name = f"icl-{run_id}"
            if name in busy:
                print(f"  {name:44s} queued already")
                continue
            if (stage_dir(run_id) / "result.json").exists():
                print(f"  {name:44s} done already")
                continue
            resources = ENS[db, task][rank]
            print(
                f"  {name:44s} {resources.gpus} {resources.qos:15s} {resources.time}"
                f"  {(ctx, lcs, bw, pl)}"
            )
            submit(
                "expts.icl.run:main",
                args=dict(
                    db=db,
                    task=task,
                    load_ckpt_path=checkpoint(db, task),
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
                    run_name=f"{db}/{task}/cfg{rank}",
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
