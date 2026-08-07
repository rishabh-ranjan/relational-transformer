"""Fine-tune a Relational Transformer on one task, and submit one job per task.

    pixi run python expts/fine_tune/submit.py

One file, and every argument written where it is passed: `rt.train:main` is the
target, so there is no per-experiment wrapper to keep in step with it, and the
call below is both the recipe and the record of what ran. Change a number here
and the diff is the experiment.

The values are pretraining's (the released RT-J recipe, `examples/train.py`).
What fine-tuning changes is the data: one `(db, task)` pair instead of a mixture,
read from the *benchmark* directory rather than the Join, so a run trains where
it is evaluated and train/eval differ only in split. `load_ckpt_path=None` is the
random-init control -- what the architecture learns from the task alone, which is
the number the pretrained arm has to beat.

Run it from a clean, pushed checkout: the job clones the commit you submit from.
Edit it freely -- it takes no arguments, and the next submission wants a
different shape anyway (see expts/README.md).
"""

from roach.slurm import Resources, submit

# The forecast tasks of these four databases whose type RT models: predict a
# label at a timestamp from what is known before it. Nothing else is here --
# link-prediction tasks are not modeled, autocomplete tasks complete a column
# rather than forecast one, and rel-event's user-repeat is an external table
# rather than a forecast. Written out rather than discovered at submit time --
# the list is what ran, and a database gaining a task should not silently
# change a sweep.
TASKS = (
    ("rel-f1", "driver-dnf"),
    ("rel-f1", "driver-position"),
    ("rel-f1", "driver-top3"),
    ("rel-event", "user-attendance"),
    ("rel-event", "user-ignore"),
    ("rel-trial", "site-success"),
    ("rel-trial", "study-adverse"),
    ("rel-trial", "study-outcome"),
    ("rel-avito", "ad-ctr"),
    ("rel-avito", "user-clicks"),
    ("rel-avito", "user-visits"),
)


def targets_for(db: str, task: str) -> dict[str, float]:
    """The best published number for this task, keyed by the wandb metric it
    bounds: the best of every model in expts/fine_tune/results.md, over both
    the default and the HPO arm.

    In the raw units `rt.train` logs (AUC as a fraction, MAE unnormalized), not
    results.md's percentages, so a value sits on the same axis as the curve it
    bounds. `rt.train` logs these at every step, which draws them as horizontal
    lines (wandb has no reference-line primitive); `workspace.py` is what puts
    curve and target in one panel, and reads them from here.

    Only this task's entries: a target for a task a run never evaluates would
    draw a line in a panel that has no curve.
    """
    return {
        k: v
        for k, v in {
            "auc/test/rel-amazon/item-churn": 0.830527,
            "auc/test/rel-amazon/user-churn": 0.704596,
            "auc/test/rel-avito/user-clicks": 0.679955,
            "auc/test/rel-avito/user-visits": 0.669391,
            "auc/test/rel-event/user-ignore": 0.884695,
            "auc/test/rel-event/user-repeat": 0.78496,
            "auc/test/rel-f1/driver-dnf": 0.730298,
            "auc/test/rel-f1/driver-top3": 0.811207,
            "auc/test/rel-hm/user-churn": 0.70569,
            "auc/test/rel-stack/user-badge": 0.888748,
            "auc/test/rel-stack/user-engagement": 0.907587,
            "auc/test/rel-trial/study-outcome": 0.755003,
            "auc/val/rel-amazon/item-churn": 0.824893,
            "auc/val/rel-amazon/user-churn": 0.706883,
            "auc/val/rel-avito/user-clicks": 0.672679,
            "auc/val/rel-avito/user-visits": 0.703479,
            "auc/val/rel-event/user-ignore": 0.8774,
            "auc/val/rel-event/user-repeat": 0.746042,
            "auc/val/rel-f1/driver-dnf": 0.685578,
            "auc/val/rel-f1/driver-top3": 0.73869,
            "auc/val/rel-hm/user-churn": 0.707665,
            "auc/val/rel-stack/user-badge": 0.900339,
            "auc/val/rel-stack/user-engagement": 0.90639,
            "auc/val/rel-trial/study-outcome": 0.706512,
            "mae/test/rel-amazon/item-ltv": 46.7891,
            "mae/test/rel-amazon/user-ltv": 14.3119,
            "mae/test/rel-avito/ad-ctr": 0.0310402,
            "mae/test/rel-event/user-attendance": 0.234435,
            "mae/test/rel-f1/driver-position": 3.74971,
            "mae/test/rel-hm/item-sales": 0.0531676,
            "mae/test/rel-stack/post-votes": 0.0648983,
            "mae/test/rel-trial/site-success": 0.324851,
            "mae/test/rel-trial/study-adverse": 40.3657,
            "mae/val/rel-amazon/item-ltv": 43.235,
            "mae/val/rel-amazon/user-ltv": 12.022,
            "mae/val/rel-avito/ad-ctr": 0.0291652,
            "mae/val/rel-event/user-attendance": 0.228424,
            "mae/val/rel-f1/driver-position": 3.50024,
            "mae/val/rel-hm/item-sales": 0.0640494,
            "mae/val/rel-stack/post-votes": 0.058982,
            "mae/val/rel-trial/site-success": 0.339356,
            "mae/val/rel-trial/study-adverse": 42.9937,
        }.items()
        if k.endswith(f"/{db}/{task}")
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

    Blackwell before Ampere, and within a card the QOS in priority order, each
    taken up to the limit that makes the next one necessary:

    * `il-interactive` caps a user at 2 GPUs of any kind (12h wall, highest
      priority, not preempted). A job that hits that wall stops; resubmit it
      with the same run_id and it resumes from its own checkpoint.
    * `il` caps b200 at 2 and a100 at 10 per user (7d wall, not preempted).
    * `il-lo` is preemptible and effectively uncapped (21d wall).

    A whole card better beats a faster one: an idle a100 on `il` starts now,
    while blackwell1 is one node whose eight cards the rest of the cluster wants
    too -- measured, three b200 `il-lo` jobs sat pending behind another user's
    reservation while a100s were free. So both `il` tiers come before either
    `il-lo` tier, and the b200 `il-lo` share stops at 4, which is what is left of
    blackwell1 once the four above are held.

    Every one of these runs is preemption-safe -- it checkpoints and resumes --
    so the low-priority queue costs wall clock, not work.
    """
    tiers = [
        (2, b200("il-interactive", "12:00:00")),
        (2, b200("il", "7-00:00:00")),
        (10, a100("il", "7-00:00:00")),
        (4, b200("il-lo", "21-00:00:00")),
    ]
    out = [r for count, r in tiers for _ in range(count)][:n]
    # whatever is left goes to the queue with no cap
    out += [a100("il-lo", "21-00:00:00")] * (n - len(out))
    return out


def main() -> None:
    for (db, task), resources in zip(TASKS, plan(len(TASKS)), strict=True):
        print(
            f"  {db}/{task:16s} {resources.gpus} {resources.qos:15s} {resources.time}"
        )
        submit_one(db, task, resources)


def submit_one(db: str, task: str, resources: Resources):
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
            # the arm: None is random init, a checkpoint path is fine-tuning
            load_ckpt_path=None,
            # data: one task, from the benchmark data rather than the Join
            db_task_list=[(db, task)],
            pre_dir="/dfs/user/ranjanr/share/stanford-star/relbench-preprocessed",
            tokens_per_gpu=2**17,
            num_workers=16,
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
            # task's: the one number this experiment sets on its own
            total_steps=10_001,
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
            eval_items_per_task=1_000_000_000,
            eval_ctx_size_list=[1024],
            eval_mmap_populate=True,
            eval_shuffle_seed=0,
            eval_context_seed=0,
            eval_vector_db_path=None,
            eval_lcs_bw_pl_grid=[(1024, 1024, True)],
            # logging
            targets=targets_for(db, task),
            project="2026-08-06-fine_tune",
            entity="rtv2",
            run_name=f"{db}/{task}",
            wandb_disabled=False,
            out_root="/dfs/user/ranjanr/ckpts",
        ),
        # one GPU per job: the slot `plan` picked for it
        resources=resources,
        name=f"{db}-{task}",
        repo_root="/lfs/hyperturing1/0/ranjanr/clones/rishabh-ranjan/relational-transformer",
        log_root="/dfs/user/ranjanr/slurm-logs/rishabh-ranjan/relational-transformer/expts/fine-tune",
        # the node's own big disk, not /tmp (the 280G root filesystem): clones
        # are shared per commit and hold the pixi env, which pixi hardlinks from
        # the package cache only when the two are on the same filesystem
        clone_root="/lfs/local/0/roach_clones",
        secrets_dir="/dfs/user/ranjanr/.secrets",
        # No setup: `pixi install` already builds the rustler extension and puts
        # it in src/rt/ -- the project is an editable dependency of its own
        # environment. Running `pixi run build-sampler` here as well rebuilt the
        # whole pyo3 stack a second time (maturin develop drops
        # pyo3/extension-module, which is a different fingerprint), for ~10s and
        # no different result.
    )


if __name__ == "__main__":
    main()
