"""Submit one fine-tuning job per task. See [README.md](README.md)."""

import functools
import json
from pathlib import Path

from roach.slurm import Resources, submit

HERE = Path(__file__).parent

# The forecast tasks of these four databases whose type RT models: predict a
# label at a timestamp from what is known before it. Nothing else is here --
# recommendation tasks are not modeled and autocomplete tasks complete a
# column rather than forecast one. Written out rather than discovered at submit time --
# the list is what ran, and a database gaining a task should not silently
# change a sweep.
TASKS = (
    ("rel-avito", "ad-ctr"),
    # ("rel-avito", "user-clicks"),
    # ("rel-avito", "user-visits"),
    # ("rel-f1", "driver-dnf"),
    # ("rel-f1", "driver-position"),
    # ("rel-f1", "driver-top3"),
    ("rel-event", "user-attendance"),
    # ("rel-event", "user-ignore"),
    ("rel-event", "user-repeat"),
    # ("rel-trial", "site-success"),
    # ("rel-trial", "study-adverse"),
    ("rel-trial", "study-outcome"),
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
    # Both arms of every model, exactly as make_results.py splits them: the
    # default config, and the trial the search selected. A row can be both
    # (the default won the search), in which case it stands in both arms.
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
            # A model's mean is over that table's whole task set, so the means
            # are taken per model+arm and the best of those is the target.
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
        # Every ampere but 9, whose node-local disk answers mkdir with "Input/
        # output error": a job landing there dies in its first second, before
        # the clone exists. Put it back once the disk is replaced.
        nodelist="ampere1,ampere2,ampere3,ampere4,ampere5,ampere6,ampere7,ampere8",
    )


def plan(n: int) -> list[Resources]:
    """The best n slots this cluster will give one-GPU jobs, best first.

    Everything sits on `il-lo`: preemptible, effectively uncapped, 21d wall.
    The capped tiers (`il-interactive` at 2 GPUs, `il` at 2 b200 and 10 a100)
    are what a sweep reaches for when it wants to be un-preemptible, and the
    price of that is a per-user ceiling that leaves the tail of the sweep
    queued behind its own head. This sweep would rather hold every card it can
    and give them back when someone with priority asks: each run checkpoints
    and resumes, at a preemption and at its time limit alike, so the
    low-priority queue costs wall clock, not work.

    Blackwell before Ampere within that, and the b200 share stops at 4:
    blackwell1 is one node whose eight cards the rest of the cluster wants too,
    and b200 `il-lo` jobs queue behind other users' reservations while a100s sit
    free. A whole card that starts now beats a faster one that does not.
    """
    out = [b200("il-lo", "21-00:00:00")] * min(n, 4)
    # whatever is left goes to the ampere queue, which has no cap of its own
    out += [a100("il-lo", "21-00:00:00")] * (n - len(out))
    return out


def main() -> None:
    # Smallest train set first: the fine-tuning question is sharpest where a
    # task has the least to learn from, and a queue that starts there answers
    # it before the long runs take the cluster.
    tasks = sorted(TASKS, key=lambda p: ntrain()[f"{p[0]}/{p[1]}"])
    for (db, task), resources in zip(tasks, plan(len(tasks)), strict=True):
        name = f"{db}/{task}"
        print(f"  {name:28s} {resources.gpus} {resources.qos:15s} {resources.time}")
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
                skip_full_attn=True,
                loss_fn="huber",
                load_ckpt_path=None,
                db_task_list=[(db, task)],
                pre_dir="/dfs/user/ranjanr/share/stanford-star/relbench-preprocessed",
                tokens_per_gpu=2**17 if resources.gpus.startswith("b200") else 2**16,
                num_workers=resources.cpus_per_task,
                prefetch_factor=2,
                ctx_size_list=[1024],
                local_ctx_size_list=[128, 256, 512, 1024],
                bfs_width_list=[8, 16, 32, 64, 128, 256],
                prefer_latest_list=[False, True],
                num_walks=10_000,
                walk_length=20,
                mask_prob_max=0.5,
                items_per_task=1_000_000_000,
                lr=5e-4,
                wd=0.1,
                warmup_steps=100,
                grad_norm_max=1.0,
                total_bs=128,
                total_steps=2_000,
                swa_momentum=0.999,
                seed=0,
                mmap_populate=True,
                timeout_per_item=10.0,
                eval_freq=100,
                keep_all_ckpts=False,
                vector_db_path=None,
                resume_save_mins=20.0,
                eval_splits=["val", "test"],
                eval_db_task_list=[(db, task)],
                eval_pre_dir="/dfs/user/ranjanr/share/stanford-star/relbench-preprocessed",
                eval_tokens_per_gpu=2**18,
                eval_num_workers=1,
                eval_prefetch_factor=2,
                eval_num_walks=10_000,
                eval_walk_length=20,
                eval_items_per_task=1024,
                eval_ctx_size_list=[1024],
                eval_mmap_populate=True,
                eval_shuffle_seed=0,
                eval_context_seed=0,
                eval_vector_db_path=None,
                eval_lcs_bw_pl_grid=[(1024, 1024, True)],
                targets=targets_for(db, task),
                project="2026-08-07-fine_tune",
                entity="rtv2",
                run_name=name,
                wandb_disabled=False,
                out_root="/dfs/user/ranjanr/ckpts",
            ),
            # one GPU per job: the slot `plan` picked for it
            resources=resources,
            name=f"{db}-{task}",
            repo_root="/lfs/hyperturing1/0/ranjanr/clones/rishabh-ranjan/relational-transformer",
            log_root="/dfs/user/ranjanr/slurm-logs/rishabh-ranjan/relational-transformer/expts/fine-tune",
            # the node's own big disk, not /tmp (the 280G root filesystem)
            clone_root="/lfs/local/0/roach_clones",
            secrets_dir="/dfs/user/ranjanr/.secrets",
            # a run_id here names an existing run instead of minting a new one:
            # how a job that was cancelled comes back and resumes from its own
            # checkpoint rather than from step 0
            run_id=None,
            # No setup: `pixi install` already builds the rustler extension into
            # src/rt/ -- the project is an editable dependency of its own
            # environment.
        )


if __name__ == "__main__":
    main()
