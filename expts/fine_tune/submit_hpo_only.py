"""Tune the eval context per task on the fine-tuned checkpoints, on validation
only, one job per task. `submit_ens.py` scores the winners on test. See
[README.md](README.md)."""

import json
from pathlib import Path

from roach.slurm import Resources, submit

from submit import TASKS, ntrain

CKPT_ROOT = Path("/dfs/user/ranjanr/ckpts/rtv2/2026-08-08-fine_tune")


def ckpt_for(db: str, task: str) -> str:
    """The fine-tuned weights for this task: the best-on-val checkpoint of the
    most recent run named `{db}/{task}` under `CKPT_ROOT`.

    A weights *file*, not the run directory: `config.json` names
    `model.safetensors`, which a training run never writes, so pointing
    `load_ckpt_path` at the directory fails to resolve. The sibling
    `config.json` still supplies the dims.

    `best_{clf,reg}` is the better of the live and the swa net on validation,
    so which of the two exists follows from the task type; a run that has not
    reached its first eval has neither, and asserting here beats a job that
    dies minutes in. `best_swa_*`/`best_live_*` sit beside it -- glob one of
    those instead to tune one net's weights rather than the winner's.
    """
    runs = sorted(
        d
        for d in CKPT_ROOT.iterdir()
        if json.loads((d / "params.json").read_text())["run_name"] == f"{db}/{task}"
    )
    assert runs, f"no run named {db}/{task} under {CKPT_ROOT}"
    run = runs[-1]
    best = [
        p
        for p in run.glob("best_*.safetensors")
        if p.stem in ("best_clf", "best_reg")
        # if p.stem in ("best_swa_clf", "best_swa_reg")
    ]
    assert len(best) == 1, f"{run} holds {[p.name for p in best]}, want one"
    return str(best[0])


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


# Which slot each job goes in, laid out by hand -- one line per job, keyed by
# `(db, task)`. Commenting a line out is how a job is left out of a submission.
#
# NOT A DEFAULT TO INHERIT, and blank on purpose: whatever the last submission
# put here is a record of a different cluster and a different instruction. Work
# the assignment out again every time, following
# [Allocating a sweep](../README.md#allocating-a-sweep) -- read the cluster,
# subtract what your own jobs already hold, spend the tiers top down -- and
# write today's answer here, one line per job this submission sends:
#
#     ("rel-f1", "driver-dnf"): a100("il", "12:00:00"),
#
# A job with no line here stops the submission rather than taking a slot
# nobody chose for it.
RESOURCES: dict[tuple[str, str], Resources] = {}


def main() -> None:
    tasks = sorted(TASKS, key=lambda p: -ntrain()[f"{p[0]}/{p[1]}"])
    # Every checkpoint before any job: `ckpt_for` asserts, and a task whose
    # fine-tuning run has not reached its first eval must abort the submission
    # rather than leave the tasks ahead of it queued and the rest not.
    ckpts = {t: ckpt_for(*t) for t in tasks}
    for db, task in tasks:
        resources = RESOURCES[db, task]
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
                splits=["val"],
                db_task_list=[(db, task)],
                pre_dir="/dfs/user/ranjanr/share/stanford-star/relbench-preprocessed",
                tokens_per_gpu=2**18,
                num_workers=resources.cpus_per_task,
                prefetch_factor=2,
                num_walks=10_000,
                walk_length=20,
                items_per_task={"val": 2**14},
                mmap_populate=True,
                shuffle_seed=0,
                context_seed=0,
                vector_db_path=None,
                db_upto_test_timestamp=True,
                ctx_size_list=[512, 1024, 2048],
                lcs_bw_pl_grid=[
                    (lcs, bw, pl)
                    for lcs in (512, 1024, 2048)
                    for bw in (64, 128, 256)
                    for pl in (True, False)
                ],
                val_ensemble_size=1,
                test_ensemble_size=1,
                run_name=None,
                targets={},
                project="2026-08-10-fine_tune_hpo",
                entity="rtv2",
                out_root="/dfs/user/ranjanr/ckpts",
                wandb_disabled=True,
            ),
            resources=resources,
            name=f"hpo-{db}-{task}",
            repo_root="/lfs/hyperturing1/0/ranjanr/clones/rishabh-ranjan/relational-transformer",
            log_root="/dfs/user/ranjanr/slurm-logs/rishabh-ranjan/relational-transformer/expts/fine-tune-hpo",
            clone_root="/lfs/local/0/roach_clones",
            secrets_dir="/dfs/user/ranjanr/.secrets",
            run_id=None,
        )


if __name__ == "__main__":
    main()
