"""Submit one fine-tuning job per task. See [README.md](README.md)."""

import functools
import json
from pathlib import Path

from roach.slurm import Resources, submit

HERE = Path(__file__).parent

TASKS = (
    ("rel-trial", "study-adverse"),
    ("rel-event", "user-attendance"),
    ("rel-event", "user-ignore"),
    ("rel-trial", "study-outcome"),
    ("rel-f1", "driver-dnf"),
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


def plan(n: int) -> list[Resources]:
    """The best n slots this cluster will give one-GPU jobs, best first.

    Blackwell before Ampere, and the higher qos before the lower, per
    [../README.md](../README.md). The per-user caps are what bound each tier
    and they come straight from `sacctmgr show qos`: `il-interactive` is 2 gpus
    of any type at 12h, `il` is 10 gpus together but only 2 of them b200 at 7d,
    `il-lo` is preemptible, effectively uncapped and 21d.

    So `il` goes entirely to amperes: it caps b200 at 2 either way, and its
    general cap buys more by being spent on the card it does not restrict. The
    blackwell share is what blackwell1 has idle at submission time -- 2 here,
    which `il-interactive` covers on its own. Recount and rewrite this before
    every submission.

    A run checkpoints and resumes, at a preemption and at its wall limit alike,
    so a short or low-priority slot costs wall clock rather than work.
    """
    out = [b200("il-interactive", "12:00:00")] * min(n, 2)
    out += [b200("il-lo", "21-00:00:00")] * (n - len(out))
    return out


def main() -> None:
    tasks = sorted(TASKS, key=lambda p: -ntrain()[f"{p[0]}/{p[1]}"])
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
                loss_fn="huber",
                load_ckpt_path=ckpt_for(db, task),
                db_task_list=[(db, task)],
                pre_dir="/dfs/user/ranjanr/share/stanford-star/relbench-preprocessed",
                tokens_per_gpu=2**17 if resources.gpus.startswith("b200") else 2**16,
                num_workers=resources.cpus_per_task,
                prefetch_factor=2,
                ctx_size_list=[512, 1024, 2048],
                local_ctx_size_list=[256, 512, 1024, 2048],
                bfs_width_list=[32, 64, 128, 256],
                prefer_latest_list=[False, True],
                num_walks=10_000,
                walk_length=20,
                mask_prob_max=0.0,
                items_per_task=1_000_000_000,
                lr=5e-4,
                wd=0.1,
                warmup_steps=500,
                grad_norm_max=1.0,
                total_bs=256,
                total_steps=10_001,
                swa_momentum=0.999,
                seed=0,
                mmap_populate=True,
                timeout_per_item=10.0,
                eval_freq=500,
                keep_all_ckpts=False,
                vector_db_path=None,
                resume_save_mins=20.0,
                eval_splits=["val", "test"],
                eval_db_task_list=[(db, task)],
                eval_pre_dir="/dfs/user/ranjanr/share/stanford-star/relbench-preprocessed",
                eval_tokens_per_gpu=2**18,
                eval_num_workers=resources.cpus_per_task,
                eval_prefetch_factor=2,
                eval_num_walks=10_000,
                eval_walk_length=20,
                eval_items_per_task=2**16,
                eval_ctx_size_list=[2048],
                eval_mmap_populate=True,
                eval_shuffle_seed=0,
                eval_context_seed=0,
                eval_vector_db_path=None,
                eval_lcs_bw_pl_grid=[(2048, 128, True)],
                targets=targets_for(db, task),
                project="2026-08-08-fine_tune",
                entity="rtv2",
                run_name=name,
                wandb_disabled=False,
                out_root="/dfs/user/ranjanr/ckpts",
            ),
            resources=resources,
            name=f"{db}-{task}",
            repo_root="/lfs/hyperturing1/0/ranjanr/clones/rishabh-ranjan/relational-transformer",
            log_root="/dfs/user/ranjanr/slurm-logs/rishabh-ranjan/relational-transformer/expts/fine-tune",
            clone_root="/lfs/local/0/roach_clones",
            secrets_dir="/dfs/user/ranjanr/.secrets",
            run_id=None,
        )


if __name__ == "__main__":
    main()
