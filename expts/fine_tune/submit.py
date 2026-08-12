"""Submit one fine-tuning job per task. See [README.md](README.md).

The base config, in the values below rather than in prose elsewhere -- it
changes every submission, and a description that lives in another file goes
stale the moment one of these does:

- **delta fine-tuning from RT-J**: the published weights are what decay pulls
  back to, and the update is the ordinary one (see `rt.train`'s
  `delta_finetune`);
- 25 epochs at batch 256, lr 5e-4 held constant -- no warmup, no decay -- and
  weight decay 0.1, Muon;
- a **fixed** context -- ctx and local ctx 1024, bfs width 128, prefer-latest
  off, no walks, the eval grid the same one -- and no masking;
- trained on the splits the arm names, and evaluated on the one it names, with
  `db_cutoff` at that split's own timestamp. The reporting arm trains on
  train+val and scores test; the selection arm trains on train and scores val,
  to ask whether val picks the same epoch test would;
- **equal-weight SWA** (`swa_momentum=1.0`, an fp32 running mean over every
  step) evaluated and saved beside the live net, which is what stands in for a
  decayed learning rate here;
- every eval is a **4-seed context ensemble** (`eval_ensemble_size`), for the
  live net and the SWA net alike, so the logged test curves are ensembled
  numbers rather than single-context ones;
- no early stopping: the budget is the budget.

An arm is this file with one value changed and a `tag` that names it; the base
carries no tag at all.
"""

import functools
import json
import math
from pathlib import Path

from roach.slurm import Resources, submit

HERE = Path(__file__).parent

TASKS = (
    ("rel-event", "user-repeat"),
    ("rel-f1", "driver-dnf"),
    ("rel-f1", "driver-top3"),
    ("rel-f1", "driver-position"),
    # ("rel-trial", "study-outcome"),
    ("rel-avito", "ad-ctr"),
    ("rel-event", "user-attendance"),
    # ("rel-event", "user-ignore"),
    # ("rel-trial", "study-adverse"),
    # ("rel-trial", "site-success"),
    # ("rel-avito", "user-visits"),
    # ("rel-avito", "user-clicks"),
    # ("rel-hm", "user-churn"),
    # ("rel-stack", "user-engagement"),
    # ("rel-hm", "item-sales"),
    # ("rel-stack", "post-votes"),
    # ("rel-amazon", "item-churn"),
    # ("rel-amazon", "item-ltv"),
    # ("rel-stack", "user-badge"),
    # ("rel-amazon", "user-churn"),
    # ("rel-amazon", "user-ltv"),
)


@functools.cache
def published_best() -> dict[str, float]:
    """The best published number per wandb metric key, from results.csv.

    Computed the same way `make_results.py` builds results.md -- over the
    default and the HPO arm of every model, AUROC as a percent and MAE
    normalized by the train-target std and taken as a percent -- so a target
    is literally the bolded number in that table, and lands on the same axis
    as the curve `rt.train` logs beside it. Derive it, never paste it: the
    published tables carry raw MAE where the run logs nMAE.

    `{metric}/{split}/mean` comes along too: the best mean over that table's
    whole task set. A single-task run's own "mean" is that one task, so this
    line says where the field's best all-round model sits, not what this run
    is being asked to beat.
    """
    import pandas as pd
    from huggingface_hub import hf_hub_download

    stds = json.load(
        open(
            hf_hub_download(
                "stanford-star/relbench", "regression_stds.json", repo_type="dataset"
            )
        )
    )["stds"]

    raw = pd.read_csv(HERE / "results.csv")
    raw["pair"] = raw.dataset + "/" + raw.task
    dflt = raw[raw.config_tag == "default"].assign(arm="D")
    hpo = raw[raw.selected].assign(arm="H")
    d = pd.concat([dflt, hpo])
    d["row"] = d.model + " (" + d.arm + ")"

    out: dict[str, float] = {}
    for task_type, metric, higher in [
        ("BINARY_CLASSIFICATION", "auroc", True),
        ("REGRESSION", "nmae", False),
    ]:
        sub = d[d.task_type == task_type]
        best = max if higher else min
        for split in ("val", "test"):
            v = sub[f"{split}_score"] * 100
            if not higher:
                v = v / sub.pair.map(stds)
            for pair, x in v.groupby(sub.pair):
                out[f"{metric}/{split}/{pair}"] = float(best(x))
            out[f"{metric}/{split}/mean"] = float(best(v.groupby(sub.row).mean()))
    return out


@functools.cache
def nsplit() -> dict[str, dict[str, float]]:
    """Rows per split per `{db}/{task}`, from RelBench's own task stats.

    What an epoch is: `rt.train`'s stream is every row of the splits in
    `train_splits`, uncapped here (`items_per_task` is far above any of them),
    so an epoch is their sum and a step covers `total_bs` of it.
    """
    import pandas as pd
    from huggingface_hub import hf_hub_download

    stats = pd.read_parquet(
        hf_hub_download(
            "stanford-star/relbench", "STATS/tasks.parquet", repo_type="dataset"
        )
    )
    return {
        f"{r.database}/{r.task}": {
            "train": float(r.num_rows_train),
            "val": float(r.num_rows_val),
            "test": float(r.num_rows_test),
        }
        for r in stats.itertuples()
    }


def steps_for(db: str, task: str, splits: list[str], total_bs: int, epochs: int) -> int:
    """`epochs` passes over this task, or 50k steps, whichever comes first.

    The two ends of the task-size range want different things: rel-f1 is a few
    thousand rows, where 100 epochs is under a thousand steps, and rel-amazon
    is millions, where 100 epochs is more compute than the answer is worth.
    Training on train+val is a bigger epoch and so a longer run -- which is why
    this reads `splits` rather than assuming the train split.

    The step budget, not `total_steps`: the caller rounds it up to the eval
    cadence it picks.
    """
    rows = sum(nsplit()[f"{db}/{task}"][s] for s in splits)
    return min(math.ceil(epochs * rows / total_bs), 50_000)


def targets_for(db: str, task: str) -> dict[str, float]:
    """The published bests this task's run should draw as reference lines.

    Test only, and only this task's entries plus the `mean` line for the metric
    it is scored by: a target for a split or a task a run never evaluates draws
    a line in a panel that has no curve. `rt.train` logs each as a constant at
    every step (wandb has no reference-line primitive, a flat series is the
    line) under a `target/` prefix, and `workspace.py` pairs it with the curve
    it bounds.
    """
    keys = {k: v for k, v in published_best().items() if "/test/" in k}
    metrics = {k.split("/")[0] for k in keys if k.endswith(f"/{db}/{task}")}
    return {
        k: v
        for k, v in keys.items()
        if k.endswith(f"/{db}/{task}")
        or (k.endswith("/mean") and k.split("/")[0] in metrics)
    }


def task_type_for(db: str, task: str) -> str:
    """This task's RelBench task type, from results.csv."""
    import pandas as pd

    raw = pd.read_csv(HERE / "results.csv")
    (task_type,) = set(raw[(raw.dataset == db) & (raw.task == task)].task_type)
    return task_type


def ckpt_for(db: str, task: str, release: str | None) -> str | None:
    """The published weights this task warm-starts from: RT-P, RT-J, or None
    for a randomly initialized net.

    One head per task type, each in its own subdirectory, so which one a run
    loads follows from the task's `task_type`. A local mirror rather than
    `stanford-star/rt-{p,j}`: a compute node has no Hub access. Refresh either
    with `huggingface_hub.snapshot_download("stanford-star/rt-j", local_dir=...)`.
    """
    if release is None:
        return None
    sub = {"BINARY_CLASSIFICATION": "classification", "REGRESSION": "regression"}
    return f"/dfs/user/ranjanr/share/stanford-star/{release}/{sub[task_type_for(db, task)]}"


def loss_fn_for(db: str, task: str) -> str:
    """The loss this task trains under: the one its metric is scored by."""
    return {"BINARY_CLASSIFICATION": "bce", "REGRESSION": "l1"}[task_type_for(db, task)]


def b200(qos: str, time: str, dependency: str | None = None) -> Resources:
    """One B200. 36 cpus is blackwell1's 288 cores split eight ways, and the
    memory is that share of the node -- under the site's MaxMemPerCPU of 10700M
    times 36, which is what an explicit --mem is capped at."""
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
        dependency=dependency,
    )


def a100(
    qos: str, time: str, reservation: str | None = None, dependency: str | None = None
) -> Resources:
    """One A100. 14 cpus is what the site allows per gpu on a job that is not
    --exclusive; no --mem, so the partition's DefMemPerGPU (240000M) applies,
    which is more than an explicit request would be given.

    `reservation` is how a job reaches a node held for us -- see
    [the reservation rule](../README.md#a-reservation-is-il-lo-only)."""
    assert reservation is None or qos == "il-lo", (
        "a reserved node is ours whatever the qos, so a high tier spent there "
        "buys nothing; see ../README.md#a-reservation-is-il-lo-only"
    )
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
        reservation=reservation,
        dependency=dependency,
    )


# Which slot each job goes in, laid out by hand -- one line per task.
# Commenting a line out is how a job is left out of a submission.
#
# NOT A DEFAULT TO INHERIT: whatever the last submission put here is a record of
# a different cluster and a different instruction. Work the assignment out again
# every time, following [Allocating a sweep](../README.md#allocating-a-sweep) --
# read the cluster, subtract what your own jobs already hold, spend the tiers
# top down.
#
# A task with no line here stops the submission rather than taking a slot
# nobody chose for it.
#
# 02:20: and batch 1024 at lr 1e-3 -- the lr the batch would want if the ratio
# were held. On the reservation, which has room as the bs512 arm drains.
#
# 02:15: batch 1024 as well, on `il` -- the reservation is holding the bs512
# arm and the val-selection runs have the rest.
#
# 02:10: batch 512 on the reporting arm (train+val, test, test cutoff), on
# ampere8 so the val-selection runs keep their `il` cards.
#
# 01:20: the val-selection arm -- train on train, score val, database cut at
# the val timestamp so val stands in the same relation to its labels that test
# does to its own. Same six tasks, same everything else; the queue is empty.
#
# 00:45: the new base -- lr 5e-4 at batch 256, with a 4-seed ensemble at every
# eval. The six tasks have test splits of a few hundred to two thousand rows,
# so four passes over them costs seconds, not the minutes it would on the big
# tasks. Six `il` amperes; the queue is empty again.
#
# 00:20: the fourth corner -- lr 5e-4 at batch 256. Four on `il`, which has
# room as the base sweep's short tasks finish, two on the reservation.
#
# 00:15: two variants off the SWA base -- lr 5e-4, and batch 256 -- both on
# ampere8, which is idle, so the base sweep on `il` keeps its cards.
#
# 00:05: constant lr with SWA instead of a decay. The queue is empty again, so
# six `il` amperes; ctx is back to 1024 and the batch to 128, which is the
# cheapest step this base has had.
#
# 23:35: ctx 1024 at batch 512 -- the same context as the arm above at the
# same batch as the ctx-2048 base, and the first arm on the un-rounded step
# budget. `il` has slots as the ctx-2048 runs finish.
#
# 23:25: ctx 1024 at batch 1024, alongside the ctx-2048 runs rather than
# instead of them -- ampere8 is idle, so all six go on the reservation and
# nothing already running gives up a card.
#
# 23:20: rel-event/user-attendance moves to a b200. It is 1200 steps at 9.4s
# on an ampere -- three hours, against about one there -- and `sbatch
# --test-only` puts a b200 job of mine at 23:34, so the fifteen minutes it
# waits are bought back many times over. It resumes from its own run id.
#
# 23:20: the queue is mine again and empty. blackwell is planned for someone
# else until 23:26, so no b200: six `il` amperes, which start now. ctx 2048 is
# four times the attention of the last base, so these are the slowest short
# runs yet.
RESOURCES: dict[tuple[str, str], Resources] = {
    # 22:20: delta fine-tuning -- the pretrained weights frozen as the point
    # decay pulls back to, the update itself unchanged (see `delta_finetune`).
    #
    # 22:10: the same warm-start pair on driver-position.
    #
    # 22:05: and the same again from a random init -- `load_ckpt_path` None,
    # so the only thing the arm changes is what the run starts from.
    #
    # 22:00: RT-J instead of RT-P, base config, no masking. The mirror is
    # `/dfs/user/ranjanr/share/stanford-star/rt-j`, fetched from the login node.
    #
    # 21:50: `num_walks` is 0 from here on, train and eval alike -- so nothing
    # submitted before this is comparable, and the masking arm needs its own
    # unmasked control at the same setting. Both go out together.
    #
    # 21:40: driver-top3 alone, with 10% token masking. `il-interactive`'s
    # second gpu is free and takes a card of any type, so it starts now.
    #
    # 21:25: a fourth arm, arm A at half the context -- ctx and local ctx both
    # 512, evaluated at the same 512, so the curve measures what it trains
    # under. `il` has five slots free as the first arms finish; the two
    # shortest take reserved cards.
    #
    # 21:10: a focused iteration -- six short tasks, three arms, a project of
    # its own. The queue is empty and the whole budget is free: `il`'s ten take
    # the first arm and four of the second, ampere8's eight reserved cards the
    # rest. blackwell is planned for someone else until 21:10, so no b200.
    #
    # 21:05: plain AdamW at the 50-epoch shape -- the arm that holds the only
    # first place -- so the pair isolates the optimizer and nothing else. Above
    # the 2e-4 arm, whose pending jobs drop to `il-lo` to make room.
    #
    # 20:45: lr 2e-4, the same shape. Queued on `il` behind the 1e-4 arm rather
    # than displacing anything: the short arms finish every few minutes, so the
    # cards come free faster than a cancel-and-resume would pay for itself.
    #
    # 20:35: lr 1e-4 at batch 512 for 100 epochs -- the arm to answer first, so
    # it takes seven `il` cards from the 100-epoch runs (which resume on
    # `il-lo`) and the two idle reserved ones. Same step count as 50ep at bs256,
    # a fifth of the learning rate.
    #
    # 20:05: the five 100-epoch runs the bs512 arm displaced, resuming on
    # `il-lo` with their own run ids.
    #
    # 20:00: the bs512 arm -- 50 epochs at twice the batch, so half the steps
    # over the same data. Above the 100-epoch runs and below the other short
    # arms: five take the `il` slots those runs give back, four take reserved
    # cards as the 50-epoch jobs on ampere8 finish.
    #
    # 19:25: the nine displaced 100-epoch runs, resuming on `il-lo`. They keep
    # their run ids, so each picks up its own resume.pt where it stopped.
    #
    # 19:20: the 25-epoch arm, on `il`. Nine of the ten `il` slots were held by
    # the 50k-step 100-epoch runs, which are ~20h from an answer either way, so
    # they step aside and resume on `il-lo`; these nine are under an hour each
    # and `il` preempts to start them now.
    #
    # 19:15: and rel-event/user-attendance too -- the last reserved card the
    # 50-epoch arm needs, with rel-f1/driver-dnf finishing on its own within
    # the minute for the other. A resume costs a restart and a handful of
    # steps: resume.pt is rewritten at every eval, and these evals are frequent.
    #
    # 19:10: these two 100-epoch runs give their reserved cards to the pending
    # 50-epoch jobs and resume in the general `il-lo` pool. Chosen for having
    # real time left (99 and 402 min) where rel-f1/driver-dnf and
    # rel-trial/study-outcome are minutes from finishing -- displacing those
    # would cost a restart to save nothing.
    #
    # 19:05: the whole 50-epoch arm onto the reservation -- it is the arm to
    # answer next, and ampere8's cards are the ones nothing else can take.
    #
    # 19:00: the 50-epoch arm, nine short tasks. ampere8 is ours and has 3 free
    # cards; `il-lo` on the reservation for those, and the plain pool for the
    # rest -- every one of these is under 2h, well inside the reservation's
    # 2026-08-13T00:00 end, and the high tiers stay on the 100-epoch runs.
    #
    # 18:50: back on an ampere. blackwell1 reads 2 b200 free and the node is not
    # flagged RESERVED, but the cards are held for another job all the same --
    # `AllocTRES` does not show that, and only the pending reason does. A b200
    # job is not placed until it has been seen to start.
    #
    # 16:56, budget unspent: blackwell1 has 3 free b200 and is not flagged
    # RESERVED, so 2 go to `il-interactive` and the third to `il`'s own 2-b200
    # sub-cap. The three longest runs take them -- 4.4h each there against
    # 11.8h on an ampere.
    ("rel-amazon", "user-churn"): b200("il-interactive", "12:00:00"),
    ("rel-amazon", "user-ltv"): b200("il-interactive", "12:00:00"),
    ("rel-stack", "user-badge"): b200("il", "1-00:00:00"),
    # `il`'s nine remaining slots, next-longest first. Every ampere outside
    # the reservation is full, but an `il` job preempts the `il-lo` ones
    # holding them; the longest of these is 11.8h against a 1d wall.
    ("rel-amazon", "item-ltv"): a100("il-lo", "2-00:00:00"),
    ("rel-amazon", "item-churn"): a100("il-lo", "2-00:00:00"),
    ("rel-stack", "post-votes"): a100("il-lo", "2-00:00:00"),
    ("rel-hm", "item-sales"): a100("il-lo", "2-00:00:00"),
    ("rel-stack", "user-engagement"): a100("il-lo", "2-00:00:00"),
    ("rel-hm", "user-churn"): a100("il-lo", "2-00:00:00"),
    ("rel-trial", "site-success"): a100("il-lo", "2-00:00:00"),
    ("rel-avito", "user-visits"): a100("il-lo", "2-00:00:00"),
    ("rel-avito", "user-clicks"): a100("il-lo", "2-00:00:00"),
    # ampere8 is reserved for us and completely idle -- 8 cards nobody can
    # take and nothing preempts. `il-lo` only (a high tier there would buy a
    # card we already have), and the nine shortest runs fit it: all under 4h,
    # well inside the reservation's 2026-08-13T00:00 end.
    ("rel-trial", "study-adverse"): a100("il", "8:00:00"),
    ("rel-event", "user-attendance"): a100("il-lo", "8:00:00", "ranjanr_deadline"),
    ("rel-event", "user-ignore"): a100("il", "8:00:00"),
    ("rel-trial", "study-outcome"): a100("il", "8:00:00"),
    ("rel-f1", "driver-dnf"): a100("il-lo", "8:00:00", "ranjanr_deadline"),
    ("rel-f1", "driver-position"): a100("il-lo", "8:00:00", "ranjanr_deadline"),
    ("rel-avito", "ad-ctr"): a100("il-lo", "8:00:00", "ranjanr_deadline"),
    ("rel-event", "user-repeat"): a100("il-lo", "8:00:00", "ranjanr_deadline"),
    ("rel-f1", "driver-top3"): a100("il-lo", "8:00:00", "ranjanr_deadline"),
}


# Resume an existing run instead of starting a new one: the run whose
# `out_dir` this is picks its `resume.pt` back up. Empty when nothing resumes.
RUN_IDS: dict[tuple[str, str], str] = {}


def main() -> None:
    # The two the schedule is derived from, and the same for every task, so
    # they are read once here rather than per job: editing `train_splits` moves
    # every task's `total_steps` with it, because the val rows are part of the
    # epoch when they are trained on.
    # The splits and the database cutoff are one choice, not three: a run that
    # selects on val trains on train alone and must see the database only up to
    # the val timestamp -- the same rule a test-split run gets one split later.
    # `("train", "test")` is the arm that reports; `("val", "val")` is the arm
    # that asks whether val can pick the epoch for it.
    # train_splits, eval_split, cutoff = ["train"], "val", "val"
    train_splits, eval_split, cutoff = ["train", "val"], "test", "test"

    # How long a run is, and the tag that keeps its curves apart from the other
    # length's in the same project -- the comparison is one panel per task with
    # a group per epoch budget.
    # fmt: off
    epochs, total_bs, lr, opt, ctx, mask, release, delta, wd, lcs, bw, pl, nw, tag = 25, 1024, 1e-3, "muon", 1024, 0.0, "rt-j", True, 0.1, 1024, 128, False, 0, "-bs1024-lr1e-3"  # noqa: E501
    # epochs, total_bs, lr, opt, ctx, mask, release, delta, wd, lcs, bw, pl, nw, tag = 25, 1024, 5e-4, "muon", 1024, 0.0, "rt-j", True, 0.1, 1024, 128, False, 0, "-bs1024"  # noqa: E501
    # epochs, total_bs, lr, opt, ctx, mask, release, delta, wd, lcs, bw, pl, nw, tag = 25, 512, 5e-4, "muon", 1024, 0.0, "rt-j", True, 0.1, 1024, 128, False, 0, "-bs512"  # noqa: E501
    # epochs, total_bs, lr, opt, ctx, mask, release, delta, wd, lcs, bw, pl, nw, tag = 25, 256, 5e-4, "muon", 1024, 0.0, "rt-j", True, 0.1, 1024, 128, False, 0, "-valsel"  # noqa: E501
    # epochs, total_bs, lr, opt, ctx, mask, release, delta, wd, lcs, bw, pl, nw, tag = 25, 256, 5e-4, "muon", 1024, 0.0, "rt-j", True, 0.1, 1024, 128, False, 0, ""  # noqa: E501
    # fmt: on
    # epochs, total_bs, lr, opt, ctx, mask, release, delta, tag = 50, 256, 5e-4, "muon", 1024, 0.0, "rt-p", False, ""
    # epochs, total_bs, lr, opt, ctx, mask, release, delta, tag = 50, 256, 1e-3, "muon", 1024, 0.0, "rt-p", False, "-bs256-lr1e-3"
    # epochs, total_bs, lr, opt, ctx, mask, release, delta, tag = 50, 512, 1e-3, "muon", 1024, 0.0, "rt-p", False, "-bs512-lr1e-3"
    # epochs, total_bs, lr, opt, ctx, mask, release, delta, tag = 50, 256, 5e-4, "muon", 512, 0.0, "rt-p", False, "-ctx512"
    # Shortest run first, so the fastest answers land first. The step budget,
    # not the test split: what a job costs here is overwhelmingly its training.
    for db, task in sorted(
        TASKS, key=lambda p: steps_for(*p, train_splits, total_bs, epochs)
    ):
        resources = RESOURCES[db, task]
        name = f"{db}/{task}{tag}"
        steps = steps_for(db, task, train_splits, total_bs, epochs)
        # The budget is the budget: `epochs` passes over the stream, not a
        # rounded-up multiple of some cadence. Two arms that ask for the same
        # epochs at different batch sizes then train the same amount of data,
        # which rounding to a round number of steps quietly broke.
        total_steps = steps
        # Ten evals across the run, wherever that falls, plus the one
        # `rt.train` always does at the last step.
        eval_freq = max(1, total_steps // 10)
        # No schedule: `lr` from the first step to the last. SWA is what this
        # base averages over instead of a decay.
        lr_warmup_steps = 0
        print(f"  {name:38s} {resources.gpus} {resources.qos:15s} {resources.time}")
        submit(
            "rt.train:main",
            # Do not put comments inside this dict: it is a config block,
            # and reading it means scanning the values.
            args=dict(
                embedder="all-MiniLM-L12-v2",
                d_text=384,
                num_blocks=12,
                d_model=512,
                num_heads=8,
                d_ff=2048,
                compile=True,
                materialize_attn_masks=True,
                loss_fn=loss_fn_for(db, task),
                load_ckpt_path=ckpt_for(db, task, release),
                db_task_list=[(db, task)],
                train_splits=train_splits,
                pre_dir="/dfs/user/ranjanr/share/stanford-star/relbench-preprocessed",
                tokens_per_gpu=2**18 if resources.gpus.startswith("b200") else 2**17,
                num_workers=resources.cpus_per_task,
                prefetch_factor=2,
                ctx_size_list=[ctx],
                local_ctx_size_list=[lcs],
                bfs_width_list=[bw],
                prefer_latest_list=[pl],
                num_walks=nw,
                walk_length=20,
                mask_prob_max=mask,
                items_per_task=1_000_000_000,
                delta_finetune=delta,
                optimizer=opt,
                lr=lr,
                wd=wd,
                lr_warmup_steps=lr_warmup_steps,
                lr_decay_steps=0,
                grad_norm_max=1.0,
                total_bs=total_bs,
                total_steps=total_steps,
                early_stop_after_steps=None,
                swa_momentum=1.0,
                seed=0,
                mmap_populate=True,
                timeout_per_item=10.0,
                eval_freq=eval_freq,
                keep_all_ckpts=False,
                vector_db_path=None,
                db_cutoff=cutoff,
                resume_save_mins=20.0,
                eval_splits=[eval_split],
                eval_db_task_list=[(db, task)],
                eval_pre_dir="/dfs/user/ranjanr/share/stanford-star/relbench-preprocessed",
                eval_tokens_per_gpu=2**18,
                eval_num_workers=resources.cpus_per_task,
                eval_prefetch_factor=2,
                eval_num_walks=nw,
                eval_walk_length=20,
                eval_items_per_task=2**16,
                eval_ctx_size_list=[ctx],
                eval_mmap_populate=True,
                eval_shuffle_seed=0,
                eval_context_seed=0,
                eval_ensemble_size=4,
                eval_vector_db_path=None,
                eval_lcs_bw_pl_grid=[(lcs, bw, pl)],
                targets=targets_for(db, task),
                project="2026-08-11-iteration",
                entity="rtv2",
                run_name=name,
                wandb_disabled=False,
                out_root="/dfs/user/ranjanr/ckpts",
            ),
            resources=resources,
            name=f"{db}-{task}{tag}",
            repo_root="/lfs/hyperturing1/0/ranjanr/clones/rishabh-ranjan/relational-transformer",
            log_root="/dfs/user/ranjanr/slurm-logs/rishabh-ranjan/relational-transformer/expts/fine-tune",
            clone_root="/lfs/local/0/roach_clones",
            secrets_dir="/dfs/user/ranjanr/.secrets",
            run_id=RUN_IDS.get((db, task)),
        )


if __name__ == "__main__":
    main()
