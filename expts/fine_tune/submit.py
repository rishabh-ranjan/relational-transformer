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
    # rel-avito is what the sampler fix was for: its p2f hubs (a `Category`
    # node has ~10^6 children) are what made a run produce no step at all, so
    # these three go first, to see the fix hold before the rest of the sweep
    # is worth submitting.
    ("rel-avito", "ad-ctr"),
    ("rel-avito", "user-clicks"),
    ("rel-avito", "user-visits"),
    # ("rel-f1", "driver-dnf"),
    # ("rel-f1", "driver-position"),
    # ("rel-f1", "driver-top3"),
    # ("rel-event", "user-attendance"),
    # ("rel-event", "user-ignore"),
    # ("rel-event", "user-repeat"),
    # ("rel-trial", "site-success"),
    # ("rel-trial", "study-adverse"),
    # ("rel-trial", "study-outcome"),
)


@functools.cache
def published_best() -> dict[str, float]:
    """The best published number per wandb metric key, from results.csv.

    Computed the same way `make_results.py` builds results.md -- over the
    default and the HPO arm of every model, AUROC as a percent and MAE
    normalized by the train-target std and taken as a percent -- so a target
    is literally the bolded number in that table, and lands on the same axis
    as the curve `rt.train` logs beside it. Derived rather than pasted: the
    pasted list had raw MAE where the run logs nMAE (`rel-trial/study-adverse`
    read 40.4 against a curve that lives near 12).

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


def default_loss_fn(db: str, task: str) -> str:
    """`bce` for a classification task, `huber` for a regression one.

    The heads are the same regression heads either way -- the label rides in
    the sign of the z-scored cell (see `RelationalTransformer.loss`) -- so the
    choice is only which loss reads it, and a binary target read as a logit is
    the one the metric (AUROC) is actually scoring. The task's type comes from
    `results.csv`, the same column `make_results.py` splits the table by."""
    import pandas as pd

    raw = pd.read_csv(HERE / "results.csv")
    types = set(raw[(raw.dataset == db) & (raw.task == task)].task_type)
    if types == {"BINARY_CLASSIFICATION"}:
        return "bce"
    if types == {"REGRESSION"}:
        return "huber"
    raise ValueError(f"no single task_type for {db}/{task}: {types}")


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
        # output error" -- three jobs landed there and all three died in the
        # first second, before the clone existed. Put it back once the disk is
        # replaced.
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

    Blackwell before Ampere within that, and the b200 share stops at 4 --
    blackwell1 is one node whose eight cards the rest of the cluster wants too,
    and measured, three b200 `il-lo` jobs sat pending behind another user's
    reservation while a100s were free. A whole card that starts now beats a
    faster one that does not.
    """
    out = [b200("il-lo", "21-00:00:00")] * min(n, 4)
    # whatever is left goes to the ampere queue, which has no cap of its own
    out += [a100("il-lo", "21-00:00:00")] * (n - len(out))
    return out


def main() -> None:
    # One arm per job. The two classification tasks are submitted under both
    # losses: `bce` is what a binary target should be read by and is the
    # default this file picks, `huber` is what every run before it used, and
    # nothing has yet measured the difference on a task. ad-ctr is regression
    # and has only the one loss to run.
    arms = [
        ("rel-avito", "ad-ctr", None),
        ("rel-avito", "user-clicks", "bce"),
        ("rel-avito", "user-clicks", "huber"),
        ("rel-avito", "user-visits", "bce"),
        ("rel-avito", "user-visits", "huber"),
    ]
    for (db, task, loss_fn), resources in zip(arms, plan(len(arms)), strict=True):
        name = f"{db}/{task}" + (f"/{loss_fn}" if loss_fn else "")
        print(f"  {name:28s} {resources.gpus} {resources.qos:15s} {resources.time}")
        submit_one(db, task, resources, loss_fn=loss_fn, run_name=name)


def submit_one(
    db: str,
    task: str,
    resources: Resources,
    run_id: str | None = None,
    total_steps: int = 10_001,
    loss_fn: str | None = None,
    run_name: str | None = None,
):
    """One job. `run_id` names an existing run instead of minting a new one,
    which is how a job that was cancelled -- moved to another queue, say --
    comes back and resumes from its own checkpoint rather than from step 0.

    `run_name` overrides the wandb name, which is the task by default: two
    arms of the same task -- the two losses, say -- want to be told apart in
    the workspace."""
    return submit(
        "rt.train:main",
        args=dict(
            # model: RT-J's dims, so a fine-tuned run and a pretrained
            # checkpoint are the same architecture
            embedder="all-MiniLM-L12-v2",
            d_text=384,
            num_blocks=12,
            d_model=512,
            num_heads=8,
            d_ff=2048,
            compile=True,
            materialize_attn_masks=True,
            loss_fn=loss_fn or default_loss_fn(db, task),
            # the arm: None is random init, a checkpoint path is fine-tuning
            load_ckpt_path=None,
            # data: one task, from the benchmark data rather than the Join
            db_task_list=[(db, task)],
            pre_dir="/dfs/user/ranjanr/share/stanford-star/relbench-preprocessed",
            tokens_per_gpu=2**17,
            # Loader workers are processes, and the job only owns
            # `cpus_per_task` cores: two are held back for the training process
            # itself and the one eval worker below, so a b200 slot (36) keeps
            # the 16 this has always used while an a100 slot (14) asks for 12
            # instead of oversubscribing its allocation.
            num_workers=min(16, resources.cpus_per_task - 2),
            prefetch_factor=2,
            ctx_size_list=[1024],
            local_ctx_size_list=[1024],
            bfs_width_list=[32],
            prefer_latest_list=[True],
            num_walks=0,
            walk_length=20,
            mask_prob_max=0.0,
            items_per_task=1000_000_000,
            # optimization: pretraining's, unchanged
            lr=5e-4,
            wd=0.1,
            warmup_steps=100,
            grad_norm_max=1.0,
            total_bs=128,
            # pretraining's 100k steps is a mixture's worth of data, not one
            # task's: the one number this experiment sets on its own, and the
            # one a shorter probe run overrides
            total_steps=total_steps,
            swa_momentum=1.0,
            seed=0,
            mmap_populate=True,
            timeout_per_item=10.0,
            eval_freq=100,
            keep_all_ckpts=False,
            vector_db_path=None,
            resume_save_mins=20.0,
            # in-loop validation: the task it is trained on, on the val split
            eval_splits=["val", "test"],
            eval_db_task_list=[(db, task)],
            eval_pre_dir="/dfs/user/ranjanr/share/stanford-star/relbench-preprocessed",
            eval_tokens_per_gpu=2**18,
            eval_num_workers=1,
            eval_prefetch_factor=2,
            eval_num_walks=10_000,
            eval_walk_length=20,
            # The in-loop eval is a *trajectory* val, not a final score: it runs
            # at every eval_freq and only has to say which way the curve is
            # going, so it reads a fixed 1024-item prefix of each split, which
            # is what the rt repo's fine-tuning recipe evaluates on too. The
            # whole split is what made this hang: rel-avito/user-clicks is
            # 21_183 val + 47_996 test items, each assembled from
            # eval_num_walks=10_000 random walks by a single loader worker, so
            # the step-0 eval alone was hours and the run logged nothing.
            # A final number comes from scoring the selected checkpoint on the
            # full split, not from this.
            eval_items_per_task=1024,
            eval_ctx_size_list=[1024],
            eval_mmap_populate=True,
            eval_shuffle_seed=0,
            eval_context_seed=0,
            eval_vector_db_path=None,
            eval_lcs_bw_pl_grid=[(1024, 1024, True)],
            # logging
            targets=targets_for(db, task),
            project="2026-08-07-fine_tune",
            entity="rtv2",
            run_name=run_name or f"{db}/{task}",
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
        run_id=run_id,
        # No setup: `pixi install` already builds the rustler extension and puts
        # it in src/rt/ -- the project is an editable dependency of its own
        # environment. Running `pixi run build-sampler` here as well rebuilt the
        # whole pyo3 stack a second time (maturin develop drops
        # pyo3/extension-module, which is a different fingerprint), for ~10s and
        # no different result.
    )


if __name__ == "__main__":
    main()
