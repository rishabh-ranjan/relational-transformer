"""Ensembling on its own: score each task on test at the context the
fine-tuning runs evaluated with, averaged over context seeds, one job per task.
See [README.md](README.md).

No tuning, so nothing reads validation and this does not wait on
`submit_hpo_only.py`. `rt.eval` scores the running average after every seed, so one
job yields the whole test-metric-vs-ensemble-size curve, logged to wandb
against `ens_size` with each task's published target beside it.

Tasks go out largest train set first: the curve that takes longest starts
first, and the small ones fill the slots behind it.
"""

import functools
import json
from pathlib import Path

import wandb
from roach.slurm import Resources, submit

from ens_table import ENTITY, PROJECT, curves
from submit import b200, ntrain, targets_for

# The 21 RelBench forecast tasks, this experiment's own list rather than
# `submit.TASKS`: that one is whatever the fine-tuning sweep last submitted,
# which a partial resubmission narrows to a handful, and every task fine-tuning
# has reached belongs here.
TASKS = (
    ("rel-amazon", "item-churn"),
    ("rel-amazon", "item-ltv"),
    ("rel-amazon", "user-churn"),
    ("rel-amazon", "user-ltv"),
    ("rel-avito", "ad-ctr"),
    ("rel-avito", "user-clicks"),
    ("rel-avito", "user-visits"),
    ("rel-event", "user-attendance"),
    ("rel-event", "user-ignore"),
    ("rel-event", "user-repeat"),
    ("rel-f1", "driver-dnf"),
    ("rel-f1", "driver-position"),
    ("rel-f1", "driver-top3"),
    ("rel-hm", "item-sales"),
    ("rel-hm", "user-churn"),
    ("rel-stack", "post-votes"),
    ("rel-stack", "user-badge"),
    ("rel-stack", "user-engagement"),
    ("rel-trial", "site-success"),
    ("rel-trial", "study-adverse"),
    ("rel-trial", "study-outcome"),
)

# Where `submit.py`'s fine-tuning runs land, and the wandb project they log to.
CKPT_ROOT = Path("/dfs/user/ranjanr/ckpts/rtv2/2026-08-08-fine_tune")
FINE_TUNE_PROJECT = "rtv2/2026-08-08-fine_tune"


@functools.cache
def run_dirs() -> dict[str, Path]:
    """`{db}/{task}` -> the output directory of its most recent run."""
    out: dict[str, list[Path]] = {}
    for d in sorted(CKPT_ROOT.iterdir()):
        name = json.loads((d / "params.json").read_text())["run_name"]
        out.setdefault(name, []).append(d)
    return {k: v[-1] for k, v in out.items()}


def ckpt_for(db: str, task: str) -> str:
    """The best-on-val checkpoint of that task's fine-tuning run *so far*.

    `best_{clf,reg}.safetensors` is written only when a run reaches its final
    eval, so waiting for it would mean waiting for the whole sweep. Until then
    the same weights are already on disk under the name the periodic save gave
    them: `rt.train` prunes every checkpoint its running best does not point
    at, and keeps the live net's winner and the SWA net's.

    Which of the two is the better -- what `best_{tt}` would end up being -- is
    in the run's own val curve, so it is read from wandb: the highest AUROC or
    the lowest nMAE over both nets and every attempt, and the step it happened
    at names the file.
    """
    run = run_dirs()[f"{db}/{task}"]
    metric, higher = task_metric(db, task)
    key = f"{metric}/val/{db}/{task}"
    best = None  # (value, filename), better first
    for r in wandb.Api().runs(FINE_TUNE_PROJECT, {"config.run_name": f"{db}/{task}"}):
        for row in r.scan_history(keys=["step", key, f"swa/{key}"]):
            for k, name in ((key, "steps"), (f"swa/{key}", "swa_steps")):
                v = row[k]
                if v is None:
                    continue
                cand = (v, f"{name}={row['step']}.safetensors")
                if best is None or (cand[0] > best[0]) == higher:
                    best = cand
    assert best is not None, f"no val curve for {db}/{task} in {FINE_TUNE_PROJECT}"
    path = run / best[1]
    assert path.exists(), f"{path} was pruned; the val curve says it is the best"
    return str(path)


def task_metric(db: str, task: str) -> tuple[str, bool]:
    """The task's metric name and whether higher is better."""
    keys = targets_for(db, task)
    (metric,) = {k.split("/")[0] for k in keys if k.endswith(f"/{db}/{task}")}
    return metric, metric == "auroc"


def ready() -> list[tuple[str, str]]:
    """The tasks that can be ensembled now, largest train set first.

    A task whose fine-tuning run has not written a checkpoint yet -- it has not
    reached its first eval -- is skipped rather than aborting the submission:
    that run is still going, and this is rerun as they land.
    """
    started = {
        t
        for t in TASKS
        if any(run_dirs()[f"{t[0]}/{t[1]}"].glob("*steps=*.safetensors"))
    }
    return sorted(started, key=lambda p: -ntrain()[f"{p[0]}/{p[1]}"])


def plan(n: int) -> list[Resources]:
    """One slot per task, in the order `main` hands out tasks: largest train
    set first.

    An eval run does not checkpoint. A preemption or a wall limit restarts it
    from ensemble size 1, so what a slot is worth here is how sure it is to
    hold the whole run -- the opposite of `submit.plan`, where a short or
    preemptible slot costs minutes. So the safest slots go to the longest runs
    and the 12-hour ones to the shortest, rather than the best slots first.

    `il-lo` is preemptible and uncapped at 21 days, `il-interactive` is 2 gpus
    of any type but only 12 hours, and `il` is not used at all: its cap is 10
    gpus of any kind together, which `submit.py`'s amperes already hold, so an
    `il` job here waits on those rather than on a card. Blackwell throughout
    while blackwell1 has them -- a test pass per context seed is the whole wall
    clock, and there are `test_ensemble_size` of them.

    A whole 16-seed curve on a b200 is minutes, not hours, so 12 hours is no
    constraint on the two `il-interactive` slots and preemption on the rest
    costs one restart at worst.

    Recount and rewrite this before every submission.
    """
    out = [b200("il-interactive", "12:00:00")] * min(n, 2)
    out += [b200("il-lo", "21-00:00:00")] * (n - len(out))
    return out


def main() -> None:
    tasks = ready()
    # A task that already has a curve is not queued a second one. Comment this
    # out to re-ensemble them anyway -- worth it once a run has finished
    # training and the checkpoint under its curve has moved.
    scored = {tuple(name.split("/")) for name in curves(ENTITY, PROJECT)}
    tasks = [t for t in tasks if t not in scored]
    ckpts = {t: ckpt_for(*t) for t in tasks}
    for (db, task), resources in zip(tasks, plan(len(tasks)), strict=True):
        name = f"{db}/{task}"
        print(f"  {name:28s} {resources.gpus} {resources.qos:15s} {resources.time}")
        submit(
            "rt.eval:main",
            # Do not put comments inside this dict: it is a config block,
            # and reading it means scanning the values.
            args=dict(
                load_ckpt_path=ckpts[db, task],
                embedder="all-MiniLM-L12-v2",
                d_text=384,
                num_blocks=12,
                d_model=512,
                num_heads=8,
                d_ff=2048,
                splits=["test"],
                db_task_list=[(db, task)],
                pre_dir="/dfs/user/ranjanr/share/stanford-star/relbench-preprocessed",
                tokens_per_gpu=2**18,
                num_workers=resources.cpus_per_task,
                prefetch_factor=2,
                num_walks=10_000,
                walk_length=20,
                items_per_task=10_000_000,
                ctx_size_list=[2048],
                mmap_populate=True,
                shuffle_seed=0,
                context_seed=0,
                vector_db_path=None,
                lcs_bw_pl_grid=[(2048, 128, True)],
                val_ensemble_size=1,
                test_ensemble_size=16,
                run_name=name,
                targets=targets_for(db, task),
                project="2026-08-10-fine_tune_ens_only",
                entity="rtv2",
                out_root="/dfs/user/ranjanr/ckpts",
                wandb_disabled=False,
            ),
            resources=resources,
            name=f"ens-only-{db}-{task}",
            repo_root="/lfs/hyperturing1/0/ranjanr/clones/rishabh-ranjan/relational-transformer",
            log_root="/dfs/user/ranjanr/slurm-logs/rishabh-ranjan/relational-transformer/expts/fine-tune-ens-only",
            clone_root="/lfs/local/0/roach_clones",
            secrets_dir="/dfs/user/ranjanr/.secrets",
            run_id=None,
        )


if __name__ == "__main__":
    main()
