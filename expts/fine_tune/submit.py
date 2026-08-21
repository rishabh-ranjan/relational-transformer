"""Submit one fine-tuning job per task. See [README.md](README.md).

The config, in the values below rather than in prose elsewhere -- it changes
every submission, and a description that lives in another file goes stale the
moment one of these does:

- **delta fine-tuning from RT-PluRel**: the published weights are what decay pulls
  back to, and the update is the ordinary one (see `rt.train`'s
  `delta_finetune`);
- 25k steps at batch 256, lr 5e-4 held constant -- no warmup, no decay -- and
  weight decay 0.1, Muon;
- a **fixed** context: ctx and local ctx 1024, bfs width 128, prefer-latest
  off, no walks, and the eval grid the same one. No token masking;
- trained on train **and** val and scored on test, with `db_cutoff` at the test
  timestamp, so the model stands in the same relation to its labels that
  inference does. Nothing selects a checkpoint: the run keeps its last step
  (`latest.safetensors`);
- **EMA of the weights** (`swa_momentum=0.9995`, an fp32 running average with
  a ~2k-step horizon), evaluated and saved beside the live net, standing in for
  a decay;
- every eval is a **4-seed context ensemble** (`eval_ensemble_size`), live net
  and SWA net alike, so the logged test curves are ensembled numbers;
- an eval every 100 steps over 4096 test rows -- the same rows every time,
  `eval_shuffle_seed` fixes them -- so the curve is dense and cheap on every
  task. The reportable number comes from `submit_ens.py` over the whole split;
- no early stopping: the budget is the budget.
"""

import functools
import json
from pathlib import Path

from roach.slurm import Resources, submit
from roach.slurm.clusters.ilc import ILC

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
    ("rel-trial", "site-success"),
    ("rel-avito", "user-visits"),
    ("rel-avito", "user-clicks"),
    ("rel-hm", "user-churn"),
    ("rel-stack", "user-engagement"),
    ("rel-hm", "item-sales"),
    ("rel-stack", "post-votes"),
    ("rel-amazon", "item-churn"),
    ("rel-amazon", "item-ltv"),
    ("rel-stack", "user-badge"),
    ("rel-amazon", "user-churn"),
    ("rel-amazon", "user-ltv"),
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
    """The published weights this task warm-starts from: RT-PluRel, RT-J, or
    None for a randomly initialized net.

    One head per task type, each in its own subdirectory, so which one a run
    loads follows from the task's `task_type`. A local mirror rather than
    `stanford-star/rt-{plurel,j}`: a compute node has no Hub access. Refresh
    either with
    `huggingface_hub.snapshot_download("stanford-star/rt-j", local_dir=...)`.
    """
    if release is None:
        return None
    sub = {"BINARY_CLASSIFICATION": "classification", "REGRESSION": "regression"}
    head = sub[task_type_for(db, task)]
    return f"/dfs/user/ranjanr/share/stanford-star/{release}/{head}"


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
    # 04:40: I hold nothing. `sbatch --test-only` says a b200 on `il-interactive`
    # starts now, a b200 or an ampere on `il` at 05:48, and anything on `il-lo` not
    # until 12:28 -- including the reservation, which expires at 2026-08-13T00:00
    # and so cannot hold a 16h job at all. Every task costs the same here (fixed
    # steps, capped eval), so the fast slots go to the tasks tonight's iteration
    # was tracking.
    #
    # 04:45: the two b200 on `il`'s sub-cap never started -- blackwell's spare
    # cards are planned for someone else, which only the pending reason shows.
    # They move to amperes, where an `il` job starts at once by preempting.
    ("rel-f1", "driver-top3"): b200("il-interactive", "12:00:00"),
    ("rel-f1", "driver-dnf"): b200("il-interactive", "12:00:00"),
    ("rel-f1", "driver-position"): a100("il", "1-00:00:00"),
    ("rel-avito", "ad-ctr"): a100("il", "1-00:00:00"),
    ("rel-event", "user-repeat"): a100("il", "1-00:00:00"),
    ("rel-trial", "study-outcome"): a100("il", "1-00:00:00"),
    ("rel-event", "user-ignore"): a100("il", "1-00:00:00"),
    ("rel-event", "user-attendance"): a100("il", "1-00:00:00"),
    ("rel-trial", "study-adverse"): a100("il", "1-00:00:00"),
    ("rel-trial", "site-success"): a100("il", "1-00:00:00"),
    ("rel-avito", "user-visits"): a100("il", "1-00:00:00"),
    ("rel-avito", "user-clicks"): a100("il", "1-00:00:00"),
    ("rel-hm", "user-churn"): a100("il-lo", "2-00:00:00"),
    ("rel-stack", "user-engagement"): a100("il-lo", "2-00:00:00"),
    ("rel-hm", "item-sales"): a100("il-lo", "2-00:00:00"),
    ("rel-stack", "post-votes"): a100("il-lo", "2-00:00:00"),
    ("rel-amazon", "item-churn"): a100("il-lo", "2-00:00:00"),
    ("rel-amazon", "item-ltv"): a100("il-lo", "2-00:00:00"),
    ("rel-stack", "user-badge"): a100("il-lo", "2-00:00:00"),
    ("rel-amazon", "user-churn"): a100("il-lo", "2-00:00:00"),
    ("rel-amazon", "user-ltv"): a100("il-lo", "2-00:00:00"),
}


# Resume an existing run instead of starting a new one: the run whose
# `out_dir` this is picks its `resume.pt` back up. Empty when nothing resumes.
RUN_IDS: dict[tuple[str, str], str] = {}


def main() -> None:
    # Every task trains the same number of steps, so submission order is the
    # order TASKS is written in.
    for db, task in TASKS:
        resources = RESOURCES[db, task]
        name = f"{db}/{task}"
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
                load_ckpt_path=ckpt_for(db, task, "rt-plurel"),
                db_task_list=[(db, task)],
                train_splits=["train", "val"],
                pre_dir="/dfs/user/ranjanr/share/stanford-star/relbench-preprocessed",
                tokens_per_gpu=2**18 if resources.gpus.startswith("b200") else 2**17,
                num_workers=resources.cpus_per_task,
                prefetch_factor=2,
                ctx_size_list=[1024],
                local_ctx_size_list=[1024],
                bfs_width_list=[128],
                prefer_latest_list=[False],
                num_walks=0,
                walk_length=20,
                mask_prob_max=0.0,
                items_per_task=1_000_000_000,
                delta_finetune=True,
                optimizer="muon",
                lr=5e-4,
                wd=0.1,
                lr_warmup_steps=0,
                lr_decay_steps=0,
                grad_norm_max=1.0,
                total_bs=256,
                total_steps=25_000,
                early_stop_after_steps=None,
                can_select_init_model=False,
                swa_momentum=0.9995,
                seed=0,
                mmap_populate=True,
                timeout_per_item=10.0,
                eval_freq=100,
                keep_all_ckpts=False,
                vector_db_path=None,
                db_cutoff="test",
                resume_save_mins=20.0,
                eval_splits=["test"],
                eval_db_task_list=[(db, task)],
                eval_pre_dir="/dfs/user/ranjanr/share/stanford-star/relbench-preprocessed",
                eval_tokens_per_gpu=2**18,
                eval_num_workers=resources.cpus_per_task,
                eval_prefetch_factor=2,
                eval_num_walks=0,
                eval_walk_length=20,
                eval_items_per_task=2**12,
                eval_ctx_size_list=[1024],
                eval_mmap_populate=True,
                eval_shuffle_seed=0,
                eval_context_seed=0,
                eval_ensemble_size=4,
                eval_vector_db_path=None,
                eval_lcs_bw_pl_grid=[(1024, 128, False)],
                targets=targets_for(db, task),
                project="2026-08-13-fine_tune",
                entity="rtv2",
                run_name=name,
                wandb_disabled=False,
                out_root="/dfs/user/ranjanr/ckpts",
            ),
            resources=resources,
            name=f"{db}-{task}",
            repo_root="/lfs/hyperturing1/0/ranjanr/clones/rishabh-ranjan/relational-transformer",
            cluster=ILC,
            job_env="expts/job_env.sh",
            log_root="/dfs/user/ranjanr/slurm-logs/rishabh-ranjan/relational-transformer/expts/fine-tune",
            clone_root="/lfs/local/0/roach_clones",
            secrets_dir="/dfs/user/ranjanr/.secrets",
            run_id=RUN_IDS.get((db, task)),
        )


if __name__ == "__main__":
    main()
