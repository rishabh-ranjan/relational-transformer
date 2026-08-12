"""Submit one fine-tuning job per task. See [README.md](README.md).

What this arm is, in the values below rather than in prose elsewhere -- it
changes every submission, and a description that lives in another file goes
stale the moment one of these does:

- warm-started from RT-P (`ckpt_for`), one head per task type;
- **fixed** eval context, not stochastic: the `*_list` knobs are single-valued,
  and `eval_lcs_bw_pl_grid` is the same configuration, so the curve measures
  what the model is trained under;
- no token masking (`mask_prob_max=0.0`);
- trained on train **and** val, so nothing selects a checkpoint: `eval_splits`
  is test alone, `swa_momentum` is None, and the run keeps its last step
  (`latest.safetensors`, and the one surviving `steps=` file);
- a finite budget -- `steps_for` steps, the lr warmed up over a fifth of it
  and decayed to zero by the end -- instead of early stopping.
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
    ("rel-trial", "study-outcome"),
    ("rel-avito", "ad-ctr"),
    ("rel-event", "user-attendance"),
    ("rel-event", "user-ignore"),
    ("rel-trial", "study-adverse"),
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


def ckpt_for(db: str, task: str) -> str:
    """The RT-P weights this task warm-starts from.

    One head per task type, each in its own subdirectory, so which one a run
    loads follows from the task's `task_type`. A local mirror rather than
    `stanford-star/rt-p`: a compute node has no Hub access.
    """
    sub = {"BINARY_CLASSIFICATION": "classification", "REGRESSION": "regression"}
    return f"/dfs/user/ranjanr/share/stanford-star/rt-p/{sub[task_type_for(db, task)]}"


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
RESOURCES: dict[tuple[str, str], Resources] = {
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
    ("rel-event", "user-attendance"): a100("il", "8:00:00"),
    ("rel-event", "user-ignore"): a100("il", "8:00:00"),
    ("rel-trial", "study-outcome"): a100("il", "8:00:00"),
    ("rel-f1", "driver-dnf"): a100("il", "8:00:00"),
    ("rel-f1", "driver-position"): a100("il", "8:00:00"),
    ("rel-avito", "ad-ctr"): a100("il", "8:00:00"),
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
    train_splits = ["train", "val"]

    # How long a run is, and the tag that keeps its curves apart from the other
    # length's in the same project -- the comparison is one panel per task with
    # a group per epoch budget.
    epochs, total_bs, lr, tag = 100, 512, 1e-4, "-100ep-bs512-lr1e-4"
    # epochs, total_bs, lr, tag = 50, 512, 5e-4, "-50ep-bs512"
    # epochs, total_bs, lr, tag = 25, 256, 5e-4, "-25ep"
    # epochs, total_bs, lr, tag = 50, 256, 5e-4, "-50ep"
    # epochs, total_bs, lr, tag = 100, 256, 5e-4, ""
    # Shortest run first, so the fastest answers land first. The step budget,
    # not the test split: what a job costs here is overwhelmingly its training.
    for db, task in sorted(
        TASKS, key=lambda p: steps_for(*p, train_splits, total_bs, epochs)
    ):
        resources = RESOURCES[db, task]
        name = f"{db}/{task}{tag}"
        steps = steps_for(db, task, train_splits, total_bs, epochs)
        # Eleven points of curve on every run, whatever its length, at a round
        # number of steps: an eval reads `eval_items_per_task` rows of the test
        # split, which on the big tasks is minutes, so a fixed cadence either
        # costs more than the training or leaves the short runs with two
        # points. `total_steps` is rounded up to it, which keeps the last step
        # on the cadence -- it is evaluated either way.
        eval_freq = math.ceil(steps / 1_000) * 100
        total_steps = math.ceil(steps / eval_freq) * eval_freq
        # Long enough to matter, short enough to leave a decay on the shortest
        # runs -- 50 epochs of rel-f1/driver-top3 is a few hundred steps.
        lr_warmup_steps = min(1_000, total_steps // 5)
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
                load_ckpt_path=ckpt_for(db, task),
                db_task_list=[(db, task)],
                train_splits=train_splits,
                pre_dir="/dfs/user/ranjanr/share/stanford-star/relbench-preprocessed",
                tokens_per_gpu=2**18 if resources.gpus.startswith("b200") else 2**17,
                num_workers=resources.cpus_per_task,
                prefetch_factor=2,
                ctx_size_list=[1024],
                local_ctx_size_list=[1024],
                bfs_width_list=[128],
                prefer_latest_list=[False],
                num_walks=10_000,
                walk_length=20,
                mask_prob_max=0.0,
                items_per_task=1_000_000_000,
                lr=lr,
                wd=0.1,
                lr_warmup_steps=lr_warmup_steps,
                lr_decay_steps=total_steps - lr_warmup_steps,
                grad_norm_max=1.0,
                total_bs=total_bs,
                total_steps=total_steps,
                early_stop_after_steps=None,
                swa_momentum=None,
                seed=0,
                mmap_populate=True,
                timeout_per_item=10.0,
                eval_freq=eval_freq,
                keep_all_ckpts=False,
                vector_db_path=None,
                db_upto_test_timestamp=True,
                resume_save_mins=20.0,
                eval_splits=["test"],
                eval_db_task_list=[(db, task)],
                eval_pre_dir="/dfs/user/ranjanr/share/stanford-star/relbench-preprocessed",
                eval_tokens_per_gpu=2**18,
                eval_num_workers=resources.cpus_per_task,
                eval_prefetch_factor=2,
                eval_num_walks=10_000,
                eval_walk_length=20,
                eval_items_per_task=2**16,
                eval_ctx_size_list=[1024],
                eval_mmap_populate=True,
                eval_shuffle_seed=0,
                eval_context_seed=0,
                eval_vector_db_path=None,
                eval_lcs_bw_pl_grid=[(1024, 128, False)],
                targets=targets_for(db, task),
                project="2026-08-12-fine_tune",
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
