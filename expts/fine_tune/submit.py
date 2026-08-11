"""Submit one fine-tuning job per task. See [README.md](README.md)."""

import functools
import itertools
import json
from pathlib import Path

from roach.slurm import Resources, submit

HERE = Path(__file__).parent

TASKS = (
    # ("rel-event", "user-repeat"),
    # ("rel-f1", "driver-dnf"),
    # ("rel-f1", "driver-top3"),
    ("rel-f1", "driver-position"),
    ("rel-trial", "study-outcome"),
    # ("rel-avito", "ad-ctr"),
    # ("rel-event", "user-attendance"),
    # ("rel-event", "user-ignore"),
    # ("rel-trial", "study-adverse"),
    # ("rel-trial", "site-success"),
    # ("rel-avito", "user-visits"),
    # ("rel-avito", "user-clicks"),
    # ("rel-hm", "user-churn"),
    ("rel-stack", "user-engagement"),
    # ("rel-hm", "item-sales"),
    # ("rel-stack", "post-votes"),
    # ("rel-amazon", "item-churn"),
    # ("rel-amazon", "item-ltv"),
    ("rel-stack", "user-badge"),
    # ("rel-amazon", "user-churn"),
    # ("rel-amazon", "user-ltv"),
)

# Random init instead of RT-P, as the control for a task whose fine-tuned
# numbers are far better than every published method's: if a model that never
# saw pretraining lands in the same place, what it knows came from the context
# it is given, not from what it was pretrained on.
RANDOM_INIT = False


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
def ntrain() -> dict[str, float]:
    """Train-set size per `{db}/{task}`, from RelBench's own task stats.

    The same `num_rows_train` column `make_results.py` orders its table columns
    by, so anything ordered by this reads in the order results.md does. A pair
    the stats do not cover sorts last rather than raising -- this only decides
    a display order, and `mean` is such a pair.
    """
    import pandas as pd
    from huggingface_hub import hf_hub_download

    stats = pd.read_parquet(
        hf_hub_download(
            "stanford-star/relbench", "STATS/tasks.parquet", repo_type="dataset"
        )
    )
    return {
        f"{r.database}/{r.task}": float(r.num_rows_train) for r in stats.itertuples()
    }


@functools.cache
def ntest() -> dict[str, float]:
    """Test-set size per `{db}/{task}`, from the same RelBench task stats.

    What this sweep's submission order is keyed on: the test split is what an
    eval pass walks, so smallest-first is fastest-to-an-answer-first.
    """
    import pandas as pd
    from huggingface_hub import hf_hub_download

    stats = pd.read_parquet(
        hf_hub_download(
            "stanford-star/relbench", "STATS/tasks.parquet", repo_type="dataset"
        )
    )
    return {
        f"{r.database}/{r.task}": float(r.num_rows_test) for r in stats.itertuples()
    }


def targets_for(db: str, task: str) -> dict[str, float]:
    """The published bests this task's run should draw as reference lines.

    Only this task's entries, plus the `mean` line for the metric it is scored
    by: a target for a task a run never evaluates would draw a line in a panel
    that has no curve. `rt.train` logs each as a constant at every step (wandb
    has no reference-line primitive, a flat series is the line) under a
    `target/` prefix, and `workspace.py` pairs it with the curve it bounds.
    """
    keys = published_best()
    metrics = {k.split("/")[0] for k in keys if k.endswith(f"/{db}/{task}")}
    return {
        k: v
        for k, v in keys.items()
        if k.endswith(f"/{db}/{task}")
        or (k.endswith("/mean") and k.split("/")[0] in metrics)
    }


def ckpt_for(db: str, task: str) -> str:
    """The RT-P weights this task warm-starts from.

    One head per task type, each in its own subdirectory, so which one a run
    loads follows from the task's `task_type` in results.csv. A local mirror
    rather than `stanford-star/rt-p`: a compute node has no Hub access.
    """
    import pandas as pd

    raw = pd.read_csv(HERE / "results.csv")
    (task_type,) = set(raw[(raw.dataset == db) & (raw.task == task)].task_type)
    sub = {"BINARY_CLASSIFICATION": "classification", "REGRESSION": "regression"}
    return f"/dfs/user/ranjanr/share/stanford-star/rt-p/{sub[task_type]}"


def b200(qos: str, time: str) -> Resources:
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
    )


def a100(qos: str, time: str) -> Resources:
    """One A100. 14 cpus is what the site allows per gpu on a job that is not
    --exclusive; no --mem, so the partition's DefMemPerGPU (240000M) applies,
    which is more than an explicit request would be given."""
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
    )


# Which slot each job goes in, laid out by hand -- one line per job, keyed by
# `(arm, db, task)`. Commenting a line out is how a job is left out of a
# submission.
#
# NOT A DEFAULT TO INHERIT: whatever the last submission put here is a record of
# a different cluster and a different instruction. Work the assignment out again
# every time, following [Allocating a sweep](../README.md#allocating-a-sweep) --
# read the cluster, subtract what your own jobs already hold, spend the tiers
# top down.
#
# Read at submission time: I hold no gpu jobs. blackwell1 has 3 of 8 b200 free,
# and the soonest of the 5 running frees in ~9h. Of those 3, only the two
# `il-interactive` jobs actually start: a reservation holds the third
# (`ReqNodeNotAvail`), so the `il` tier is all ampere. All 64 usable a100 are
# allocated (ampere7 is down) but ~50 of them are one user's `il-lo`, which `il`
# preempts, so the `il` amperes start and the `il-lo` tail queues behind. That
# gives:
#
#   il-interactive  2 b200            the two smallest-test jobs
#   il              8 a100            two slots of the ten are left free for
#                                     another experiment; the eight go to the
#                                     two largest-test tasks (rel-stack
#                                     user-engagement, user-badge), which are
#                                     the slowest and lose the most to `il-lo`
#                                     preemption, plus the smallest four still
#                                     running
#   il-lo           30 a100           everything from rel-event/user-attendance on
#
# Both arms of a task sit in the same tier: the pair is the comparison, and a
# half-finished pair answers nothing. Order is ascending test-set size
# (`ntest()`), so the fastest answers land first.
#
# A job with no line here stops the submission rather than taking a slot
# nobody chose for it.
RESOURCES: dict[tuple[str, str, str], Resources] = {
    ("train", "rel-event", "user-repeat"): b200("il-interactive", "12:00:00"),
    ("trainval", "rel-event", "user-repeat"): b200("il-interactive", "12:00:00"),
    ("train", "rel-f1", "driver-dnf"): a100("il-lo", "1-00:00:00"),
    ("trainval", "rel-f1", "driver-dnf"): a100("il", "1-00:00:00"),
    ("train", "rel-f1", "driver-top3"): a100("il", "1-00:00:00"),
    ("trainval", "rel-f1", "driver-top3"): a100("il", "1-00:00:00"),
    ("train", "rel-f1", "driver-position"): a100("il-lo", "1-00:00:00"),
    ("trainval", "rel-f1", "driver-position"): a100("il-lo", "1-00:00:00"),
    ("train", "rel-trial", "study-outcome"): a100("il-lo", "1-00:00:00"),
    ("trainval", "rel-trial", "study-outcome"): a100("il-lo", "1-00:00:00"),
    ("train", "rel-avito", "ad-ctr"): a100("il", "1-00:00:00"),
    ("trainval", "rel-avito", "ad-ctr"): a100("il-lo", "1-00:00:00"),
    ("train", "rel-event", "user-attendance"): a100("il-lo", "1-00:00:00"),
    ("trainval", "rel-event", "user-attendance"): a100("il-lo", "1-00:00:00"),
    ("train", "rel-event", "user-ignore"): a100("il-lo", "1-00:00:00"),
    ("trainval", "rel-event", "user-ignore"): a100("il-lo", "1-00:00:00"),
    ("train", "rel-trial", "study-adverse"): a100("il-lo", "1-00:00:00"),
    ("trainval", "rel-trial", "study-adverse"): a100("il-lo", "1-00:00:00"),
    ("train", "rel-trial", "site-success"): a100("il-lo", "1-00:00:00"),
    ("trainval", "rel-trial", "site-success"): a100("il-lo", "1-00:00:00"),
    ("train", "rel-avito", "user-visits"): a100("il-lo", "1-00:00:00"),
    ("trainval", "rel-avito", "user-visits"): a100("il-lo", "1-00:00:00"),
    ("train", "rel-avito", "user-clicks"): a100("il-lo", "1-00:00:00"),
    ("trainval", "rel-avito", "user-clicks"): a100("il-lo", "1-00:00:00"),
    ("train", "rel-hm", "user-churn"): a100("il-lo", "1-00:00:00"),
    ("trainval", "rel-hm", "user-churn"): a100("il-lo", "1-00:00:00"),
    ("train", "rel-stack", "user-engagement"): a100("il", "1-00:00:00"),
    ("trainval", "rel-stack", "user-engagement"): a100("il", "1-00:00:00"),
    ("train", "rel-hm", "item-sales"): a100("il-lo", "1-00:00:00"),
    ("trainval", "rel-hm", "item-sales"): a100("il-lo", "1-00:00:00"),
    ("train", "rel-stack", "post-votes"): a100("il-lo", "1-00:00:00"),
    ("trainval", "rel-stack", "post-votes"): a100("il-lo", "1-00:00:00"),
    ("train", "rel-amazon", "item-churn"): a100("il-lo", "1-00:00:00"),
    ("trainval", "rel-amazon", "item-churn"): a100("il-lo", "1-00:00:00"),
    ("train", "rel-amazon", "item-ltv"): a100("il-lo", "1-00:00:00"),
    ("trainval", "rel-amazon", "item-ltv"): a100("il-lo", "1-00:00:00"),
    ("train", "rel-stack", "user-badge"): a100("il", "1-00:00:00"),
    ("trainval", "rel-stack", "user-badge"): a100("il", "1-00:00:00"),
    ("train", "rel-amazon", "user-churn"): a100("il-lo", "1-00:00:00"),
    ("trainval", "rel-amazon", "user-churn"): a100("il-lo", "1-00:00:00"),
    ("train", "rel-amazon", "user-ltv"): a100("il-lo", "1-00:00:00"),
    ("trainval", "rel-amazon", "user-ltv"): a100("il-lo", "1-00:00:00"),
}


def main() -> None:
    tasks = sorted(TASKS, key=lambda p: ntest()[f"{p[0]}/{p[1]}"])
    arms = [("train", ["train"]), ("trainval", ["train", "val"])]
    for (db, task), (arm, train_splits) in itertools.product(tasks, arms):
        resources = RESOURCES[arm, db, task]
        name = f"{db}/{task}-{arm}" + ("-rand" if RANDOM_INIT else "")
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
                loss_fn="huber",
                load_ckpt_path=None if RANDOM_INIT else ckpt_for(db, task),
                db_task_list=[(db, task)],
                train_splits=train_splits,
                pre_dir="/dfs/user/ranjanr/share/stanford-star/relbench-preprocessed",
                tokens_per_gpu=2**17 if resources.gpus.startswith("b200") else 2**16,
                num_workers=resources.cpus_per_task,
                prefetch_factor=2,
                ctx_size_list=[1024],
                local_ctx_size_list=[1024],
                bfs_width_list=[128],
                prefer_latest_list=[True],
                num_walks=0,
                walk_length=20,
                mask_prob_max=0.0,
                items_per_task=1_000_000_000,
                lr=5e-4,
                wd=0.1,
                warmup_steps=500,
                grad_norm_max=1.0,
                total_bs=256,
                total_steps=10_001,
                swa_momentum=0.9999,
                seed=0,
                mmap_populate=True,
                timeout_per_item=10.0,
                eval_freq=500,
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
                eval_num_walks=0,
                eval_walk_length=20,
                eval_items_per_task=2**16,
                eval_ctx_size_list=[1024],
                eval_mmap_populate=True,
                eval_shuffle_seed=0,
                eval_context_seed=0,
                eval_vector_db_path=None,
                eval_lcs_bw_pl_grid=[(1024, 128, False)],
                targets=targets_for(db, task),
                project="2026-08-10-fine_tune",
                entity="rtv2",
                run_name=name,
                wandb_disabled=False,
                out_root="/dfs/user/ranjanr/ckpts",
            ),
            resources=resources,
            name=f"{db}-{task}-{arm}" + ("-rand" if RANDOM_INIT else ""),
            repo_root="/lfs/hyperturing1/0/ranjanr/clones/rishabh-ranjan/relational-transformer",
            log_root="/dfs/user/ranjanr/slurm-logs/rishabh-ranjan/relational-transformer/expts/fine-tune",
            clone_root="/lfs/local/0/roach_clones",
            secrets_dir="/dfs/user/ranjanr/.secrets",
            run_id=None,
        )


if __name__ == "__main__":
    main()
