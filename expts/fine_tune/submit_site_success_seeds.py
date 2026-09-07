from pathlib import Path

from roach.slurm.clusters.ilc import ILC

from roach.slurm import submit

from expts.fine_tune.submit import ENTITY, OUT_ROOT, PROJECT, a100, queued, targets_for

HERE = Path(__file__).parent

# 2026-09-07: fine-tuned rel-trial/site-success came out 64.45 (rt-j) vs 80.58
# (rt-plurel) nMAE, a 16-point swing on the task where every method lands
# 60-100. Three fresh training seeds per model, the full inner/tune/outer/test
# pipeline each (seed 0 is the submitted board), to see whether the gap is
# larger than the seed noise. `model` carries the seed so every stage id and
# run name is its own; the checkpoints and args are exactly submit.py's.


def main() -> None:
    busy = queued()
    for model, load_ckpt_root in (
        ("rt-plurel", "~/scratch/hf/stanford-star/rt-plurel"),
        ("rt-j", "~/scratch/hf/stanford-star/rt-j"),
    ):
        for seed in (1, 2, 3):
            name = f"ftv-{model}-s{seed}-site-success"
            if name in busy:
                print(f"  {name:44s} queued already")
                continue
            table = (
                Path(OUT_ROOT).expanduser()
                / ENTITY
                / PROJECT
                / f"{model}-s{seed}-rel-trial-site-success-test"
                / "eval_out"
                / "rel-trial__site-success.csv"
            )
            if table.exists():
                print(f"  {name:44s} done already")
                continue
            resources = a100("il", "12:00:00")
            print(f"  {name:44s} {resources.gpus} {resources.qos:15s} {resources.time}")
            submit(
                "expts.fine_tune.run:main",
                args=dict(
                    model=f"{model}-s{seed}",
                    db="rel-trial",
                    task="site-success",
                    load_ckpt_root=load_ckpt_root,
                    pre_dir="~/scratch/hf/stanford-star/relbench-preprocessed",
                    tokens_per_gpu=2**17,
                    num_workers=resources.cpus_per_task,
                    eval_num_workers=2,
                    selection_steps=50_000,
                    patience_steps=10_000,
                    eval_freq=100,
                    eval_rows=1024,
                    selection_ensemble_size=4,
                    tune_rows=4096,
                    test_ensemble_size=8,
                    seed=seed,
                    targets=targets_for("rel-trial", "site-success"),
                    project=PROJECT,
                    entity=ENTITY,
                    wandb_disabled=False,
                    out_root=OUT_ROOT,
                ),
                resources=resources,
                name=name,
                repo_root=str(HERE.parents[1]),
                cluster=ILC,
                job_env="expts/job_env.sh",
                log_root=f"{OUT_ROOT}/slurm-logs",
                clone_root="~/roach_clones",
                secrets_dir="~/scratch/.secrets",
            )


if __name__ == "__main__":
    main()
