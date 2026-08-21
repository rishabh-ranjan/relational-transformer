"""Repeat one task N times to measure rt-plurel's run-to-run spread.

Seeds 1..N, not repeats of seed 0: the pipeline is close to deterministic given
a seed, so repeating one would measure GPU nondeterminism rather than the
variance that matters -- data order, context draws, and the step validation
lands on. Seed 0 stays untouched as the canonical result.

Results land in `results-variance/`, not `results/`. Five extra rows for one
task in the main directory would weight that task five times over in any
leaderboard built from it.

il-lo on blackwell is deliberate here. The rule against it existed because a
preemption restarted the run from step 0; rt 1.7.0 resumes exactly instead, so a
preemption now costs the checkpoint interval and this is the first real exercise
of that path.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from expts.relarena.submit import (  # noqa: E402
    CACHE_DIR,
    REPO_ROOT,
    SECRETS_DIR,
    SHARE,
    b200,
    relarena_setup,
)
from roach.slurm import submit  # noqa: E402
from roach.slurm.clusters.ilc import ILC  # noqa: E402

DATASET, TASK, SEEDS = "rel-avito", "user-clicks", range(1, 6)


def main() -> None:
    for seed in SEEDS:
        job = submit(
            "expts.relarena.run:main",
            args=dict(
                dataset=DATASET,
                task=TASK,
                model="rt-plurel",
                seed=seed,
                n_trials=1,
                cache_dir=CACHE_DIR,
                out_dir=f"{SHARE}/results-variance",
            ),
            resources=b200("il-lo", "21-00:00:00"),
            name=f"relarena-rt-plurel-{DATASET}-{TASK}-var{seed}",
            setup=relarena_setup(),
            repo_root=REPO_ROOT,
            cluster=ILC,
            job_env="expts/job_env.sh",
            log_root=f"{SHARE}/slurm-logs",
            clone_root="/lfs/local/0/roach_clones",
            secrets_dir=SECRETS_DIR,
        )
        print(f"  seed {seed}: job {job.id}")


if __name__ == "__main__":
    main()
