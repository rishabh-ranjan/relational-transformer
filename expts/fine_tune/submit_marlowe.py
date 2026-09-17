import dataclasses
import subprocess
from pathlib import Path

from roach.slurm import submit
from roach.slurm.clusters import marlowe

from expts.fine_tune.submit import ENTITY, OUT_ROOT, PROJECT, targets_for

HERE = Path(__file__).parent

TASKS = (
    ("rel-hm", "user-churn"),
    ("rel-stack", "user-badge"),
    ("rel-trial", "site-success"),
    ("rel-stack", "post-votes"),
    ("rel-avito", "user-clicks"),
    ("rel-avito", "ad-ctr"),
    ("rel-trial", "study-outcome"),
    ("rel-event", "user-repeat"),
    ("rel-f1", "driver-dnf"),
    ("rel-f1", "driver-top3"),
    ("rel-f1", "driver-position"),
)


def queued() -> set[str]:
    out = subprocess.run(
        ["ssh", "marlowe", "squeue -h -u $USER -o %j"],
        capture_output=True,
        text=True,
        check=True,
    )
    return set(out.stdout.split())


def main() -> None:
    busy = queued()
    for db, task in TASKS:
        name = f"ft-rt-plurel-{db}-{task}-mw"
        if name in busy:
            print(f"  {name:48s} queued already")
            continue
        submit(
            "expts.fine_tune.run:main",
            args=dict(
                model="rt-plurel",
                db=db,
                task=task,
                load_ckpt_root="~/scratch/hf/stanford-star/rt-plurel",
                pre_dir="~/scratch/hf/stanford-star/relbench-preprocessed",
                tokens_per_gpu=2**17,
                num_workers=14,
                eval_num_workers=2,
                selection_steps=50_000,
                patience_steps=10_000,
                eval_freq=100,
                eval_rows=1024,
                selection_ensemble_size=4,
                tune_rows=4096,
                test_ensemble_size=8,
                seed=0,
                targets=targets_for(db, task),
                project=PROJECT,
                entity=ENTITY,
                wandb_disabled=False,
                out_root=OUT_ROOT,
            ),
            # 2-day walls sat behind a 3-day backfill horizon; a 6-hour slice
            # backfills into today's churn and roach requeues at the limit,
            # so a long stage just runs in slices (the 08-24 sweep's pattern)
            resources=dataclasses.replace(
                marlowe.H100,
                gpus="1",
                exclusive=False,
                cpus_per_task=14,
                time="6:00:00",
            ),
            name=name,
            repo_root=str(HERE.parents[1]),
            cluster=marlowe.MARLOWE,
            job_env="expts/job_env.sh",
            log_root=f"{OUT_ROOT}/slurm-logs",
            clone_root="~/roach_clones",
            secrets_dir="~/scratch/.secrets",
        )


if __name__ == "__main__":
    main()
