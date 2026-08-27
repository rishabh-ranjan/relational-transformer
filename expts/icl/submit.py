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
# 16:50: item-churn's cfg0 unit freed the b200 under il; rel-hm/user-churn
# (~8h left on its il a100) swaps onto it -- and back at 17:00: the card was
# taken by il-tier jobs slurm will not preempt (planned start 22:40), so it
# returns to an il a100, taking the slot user-engagement's cfg0 unit had.
# 17:10: user-badge tuned with il at 8/10 and nothing long left on il-lo, so
# two of its four units (255k rows, hours each) take the free il a100s.
# 17:20: fine_tune's il b200 refit ended and the sub-cap slot is released to
# this sweep: rel-hm/user-churn (~7h left on its il a100) swaps onto it.
# 17:20: item-ltv tuned on the il-interactive b200; its cfg0 unit keeps that
# slot, cfg1 takes an il a100, the rest il-lo at 6h.
# 18:25: item-ltv cfg1 handed an il slot back with nothing queued; user-visits
# (112/120, the tuning job with the most left) takes it off its il-lo chunks.
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
    ("rt-j", "rel-amazon", "user-churn"): a100("il", "2-00:00:00"),
    ("rt-j", "rel-amazon", "user-ltv"): a100("il", "2-00:00:00"),
    ("rt-j", "rel-amazon", "item-ltv"): b200("il-interactive", "12:00:00"),
    ("rt-j", "rel-amazon", "item-churn"): b200("il", "2-00:00:00"),
    ("rt-j", "rel-stack", "user-badge"): a100("il", "2-00:00:00"),
    ("rt-j", "rel-stack", "post-votes"): a100("il", "2:00:00"),
    ("rt-j", "rel-hm", "item-sales"): a100("il", "2-00:00:00"),
    ("rt-j", "rel-stack", "user-engagement"): a100("il", "2-00:00:00"),
    ("rt-j", "rel-hm", "user-churn"): b200("il", "2-00:00:00"),
    ("rt-j", "rel-avito", "user-clicks"): a100("il-lo", "2:00:00"),
    ("rt-j", "rel-avito", "user-visits"): a100("il", "1-00:00:00"),
    ("rt-j", "rel-trial", "site-success"): a100("il", "2-00:00:00"),
    ("rt-j", "rel-trial", "study-adverse"): a100("il", "2-00:00:00"),
    ("rt-j", "rel-event", "user-attendance"): a100("il-lo", "2:00:00"),
    ("rt-j", "rel-event", "user-ignore"): a100("il", "2-00:00:00"),
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
# went back to il-lo minutes later: fine_tune wanted that il slot for a b200
# -- and moved up again at 04:20 when item-churn cfg0 freed a slot of mine.
# 04:45: user-ltv cfg3 (1024, 352k rows) next. 04:55: rt-plurel's units are
# all placed, so the il slots its units free go to rt-j's tuning chunks,
# biggest database first: rel-amazon/user-churn, then user-ltv (05:20),
# then rel-stack/user-badge and rel-hm/item-sales (05:50, two slots freed),
# rel-stack/user-engagement (06:20), rel-trial/site-success (07:00),
# rel-event/user-ignore (07:25, rt-plurel's last unit done), rel-amazon/
# item-churn (08:10, off a running chunk: ~9h of grid left, one entry lost),
# rel-amazon/item-ltv (10:45, likewise, ~5h left). 11:05: fine_tune's last
# il b200 refit but one ended and it handed the slot over: rel-amazon/
# item-churn (~10h left on its il a100) swaps onto the b200, and its a100
# slot goes to rel-hm/user-churn off its chunks. 14:50: item-churn tuned;
# its cfg0 unit keeps that b200 under il, the rest go to il-lo at 6h. 15:25:
# fine_tune released an il-interactive b200: rel-amazon/item-ltv (~9h left on
# its il a100) swaps onto it, and its il a100 goes to rel-trial/study-adverse
# off its chunks.
# 17:45: a b200 that shows free is gone by the time a job asks, and the il-tier
# holders are not preempted (planned start 22:40 twice today); user-ltv's cfg0
# goes back to an il a100 -- no more b200 requests from this sweep unless
# fine_tune hands one over explicitly.
# 18:05: two b200s freed with il full, but user-churn's cfg1/cfg2 had just
# backfilled onto il-lo a100s, so they stay there.
# 18:15: fine_tune is done and three b200s sit idle with nothing of mine
# queued: item-sales' units take them (il, il-interactive, il-lo).
# 19:35: item-ltv cfg2 handed an il slot back: post-votes cfg1-s2 takes it.
# 19:30: the rolling 12 h il b200 array is back and preempts il-lo b200 jobs within
# minutes: post-votes cfg1-s1 goes back to an il-lo a100.
# 19:15: the rolling 12 h b200 jobs ended early and a b200 is idle again: post-votes
# cfg1-s1 (its ctx 8192 seed, ~2 h on a b200) takes it under il-lo.
# 19:10: post-votes tuned to ctx 4096/8192/1024/2048 over 161k rows -- 3.5 h /
# 4.3 h / 2.3 h / 2.8 h a seed pass on an a100 (tuning: 320/390/210/260 s per
# 4096 rows) -- so every rank runs one job per seed, sized to backfill; the
# longest seed takes the il a100 slot post-votes' own tuning just gave back.
# 19:05: a b200 is idle and my il-b200 and il-interactive slots are full: an
# il-lo b200 seed job (~3.3 h, 5 h limit) ends before the rolling 12 h il
# b200 jobs rotate at ~06:30 and could preempt it. post-votes' last tuning
# entry keeps being preempted off il-lo (by my own il jobs), so it takes the
# il a100 slot user-visits' tuning left free.
# 19:35: user-ltv cfg3-s0 was preempted off the il-lo b200 twice in 40 minutes by
# the il b200 array: back to an il-lo a100 like its siblings.
# 21:20: item-churn cfg1 was preempted off il-lo a second time, minutes before its
# last seed finished; il has room, so it resumes there.
# 21:05: user-ltv cfg2 finished on il-interactive: user-badge cfg1 moves there
# right after its second seed saved.
# 20:30: the b200 units calibrate a ctx 8192 seed pass over 352k rows at ~1.05 h
# (a100 ~2.3 h), a third of the tuning-based estimate, so the a100 seed jobs
# end ~22:30 and the long pole is user-badge cfg0-2 (1.95 h a seed on an a100,
# done ~01:00). user-visits cfg3-s3 freed an il-interactive b200 with nothing
# queued: user-badge cfg0 moves there now (one seed saved; the pass in flight
# is lost, but an idle b200 gets taken by the il array within minutes); cfg1
# and cfg2 follow onto b200s as user-ltv cfg2 / cfg0 finish theirs.
# 20:20: user-ltv cfg1 finished on il-interactive: the one job still queued,
# user-visits cfg3-s3, takes that b200.
# 20:15: user-visits tuned last (21/21): 36129 rows at ctx 4096/2048/8192/8192,
# up to ~47 min a seed pass on an a100. Its own tuning and post-votes
# cfg1-s0 left two il slots: the first two ranks run whole there under 12 h;
# the two ctx 8192 ranks run one job per seed on il-lo, sized to backfill.
# 20:05: hm/user-churn cfg0 handed an il slot back: user-ltv cfg3-s0, the only job
# still queued, takes it.
# 18:55: study-adverse's tuning handed an il slot back; cfg2-s2 backfilled onto
# il-lo before it could take it, so user-churn-cfg2-s3 does.
# 18:45: only my own four b200 slots ever free up (the other four are held by
# 7-day and rolling 12-hour il jobs), so an il-lo b200 request never starts,
# and a 2-day il-lo a100 job cannot backfill either: the free a100s are planned
# for a whole-node job at 06:27 tomorrow. A rank placed as four Resources runs
# one seed per job (run.py seed_offset), so the four 8192 units without a b200
# become sixteen ~7.2 h a100 jobs that fit the backfill window as 10 h il-lo.
# user-clicks (48k rows, ~1 h a pass): 6 h il-lo units.
# 18:35: rt-j's amazon user tasks tuned to ctx 8192/4096, where one seed pass
# over the 352k test rows is ~7.2 h / ~4.8 h on an a100 (tuning: ~300 s /
# ~200 s per 4096 rows): a 6 h or 12 h chunk cannot even finish a seed, so
# those eight units go to b200s (~3.3 h / ~2.2 h a pass) with 2-day limits --
# il-b200 for the two that already hold il slots, il-interactive for the two
# 4096 units (8.8 h fits the 12 h limit), il-lo for the rest -- and the two
# that cannot get a b200 yet run on a 2-day il-lo a100 until one frees.
ENS: dict[tuple[str, str, str], list[Resources | list[Resources]]] = {
    ("rt-plurel", "rel-amazon", "user-churn"): [
        b200("il", "1-00:00:00"),
        a100("il-lo", "1-00:00:00"),
        a100("il-lo", "1-00:00:00"),
        a100("il-lo", "1-00:00:00"),
    ],
    ("rt-plurel", "rel-amazon", "user-ltv"): [
        a100("il-lo", "6:00:00"),
        a100("il-lo", "6:00:00"),
        a100("il-lo", "6:00:00"),
        a100("il", "12:00:00"),
    ],
    ("rt-plurel", "rel-amazon", "item-ltv"): [a100("il-lo", "6:00:00")] * 4,
    ("rt-plurel", "rel-amazon", "item-churn"): [
        a100("il", "12:00:00"),
        a100("il", "12:00:00"),
        a100("il-lo", "6:00:00"),
        a100("il", "12:00:00"),
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
    ("rt-j", "rel-amazon", "user-churn"): [
        b200("il", "2-00:00:00"),
        [a100("il-lo", "10:00:00")] * 4,
        [
            a100("il-lo", "10:00:00"),
            a100("il-lo", "10:00:00"),
            a100("il-lo", "10:00:00"),
            a100("il", "10:00:00"),
        ],
        [a100("il-lo", "10:00:00")] * 4,
    ],
    ("rt-j", "rel-amazon", "user-ltv"): [
        b200("il", "2-00:00:00"),
        b200("il-interactive", "12:00:00"),
        b200("il-interactive", "12:00:00"),
        [
            a100("il", "10:00:00"),
            a100("il-lo", "10:00:00"),
            a100("il-lo", "10:00:00"),
            a100("il-lo", "10:00:00"),
        ],
    ],
    ("rt-j", "rel-amazon", "item-ltv"): [
        b200("il-interactive", "12:00:00"),
        a100("il", "12:00:00"),
        a100("il", "12:00:00"),
        a100("il-lo", "6:00:00"),
    ],
    ("rt-j", "rel-amazon", "item-churn"): [
        b200("il", "12:00:00"),
        a100("il", "6:00:00"),
        a100("il-lo", "6:00:00"),
        a100("il-lo", "6:00:00"),
    ],
    ("rt-j", "rel-stack", "user-badge"): [
        b200("il-interactive", "6:00:00"),
        b200("il-interactive", "6:00:00"),
        a100("il", "12:00:00"),
        a100("il-lo", "12:00:00"),
    ],
    ("rt-j", "rel-stack", "post-votes"): [
        [a100("il-lo", "6:00:00")] * 4,
        [
            a100("il", "6:00:00"),
            a100("il-lo", "6:00:00"),
            a100("il", "6:00:00"),
            a100("il-lo", "6:00:00"),
        ],
        [a100("il-lo", "4:00:00")] * 4,
        [a100("il-lo", "4:00:00")] * 4,
    ],
    ("rt-j", "rel-hm", "item-sales"): [
        b200("il", "12:00:00"),
        b200("il-interactive", "12:00:00"),
        b200("il-lo", "6:00:00"),
        a100("il-lo", "6:00:00"),
    ],
    ("rt-j", "rel-stack", "user-engagement"): [
        a100("il-lo", "6:00:00"),
        a100("il-lo", "6:00:00"),
        a100("il-lo", "6:00:00"),
        a100("il-lo", "6:00:00"),
    ],
    ("rt-j", "rel-hm", "user-churn"): [
        a100("il", "12:00:00"),
        a100("il-lo", "6:00:00"),
        a100("il-lo", "6:00:00"),
        a100("il-lo", "6:00:00"),
    ],
    ("rt-j", "rel-avito", "user-clicks"): [a100("il-lo", "6:00:00")] * 4,
    ("rt-j", "rel-avito", "user-visits"): [
        a100("il", "12:00:00"),
        a100("il", "12:00:00"),
        [a100("il-lo", "4:00:00")] * 4,
        [
            a100("il-lo", "4:00:00"),
            a100("il-lo", "4:00:00"),
            a100("il-lo", "4:00:00"),
            b200("il-interactive", "4:00:00"),
        ],
    ],
    ("rt-j", "rel-trial", "site-success"): [a100("il-lo", "3:00:00")] * 4,
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
                placement = ENS[model, db, task][rank]
                jobs = (
                    [(f"cfg{rank}", 0, 4, placement)]
                    if isinstance(placement, Resources)
                    else [(f"cfg{rank}-s{k}", k, 1, r) for k, r in enumerate(placement)]
                )
                for unit, seed_offset, n_seeds, resources in jobs:
                    run_id = f"ens-{model}-{db}-{task}-{unit}"
                    name = f"icl-{run_id}"
                    if name in busy:
                        print(f"  {name:48s} queued already")
                        continue
                    if (stage_dir(run_id) / "result.json").exists():
                        print(f"  {name:48s} done already")
                        continue
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
                            n_seeds=n_seeds,
                            seed_offset=seed_offset,
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
                            run_name=f"{model}/{db}/{task}/{unit}",
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
