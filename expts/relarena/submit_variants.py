"""Submit the `rt-j` and `rt` sweeps beside a running `rt-plurel` sweep.

Same 21 tasks and the same protocol; the only difference is the checkpoint each
arm starts from (see relarena's `models/rt/model.py`). `rt-plurel` is the run
that matters, so nothing here may take a card from it: these jobs go to the
reservation and to tiers `rt-plurel` is not filling, and never preempt anything.

Two rules learned the hard way this session, both encoded below:

**No `il-lo` on blackwell.** Two `rt-plurel` b200 jobs on that tier were
preempted 3 and 4 times and lost ~13h between them. B200 is reachable here only
through `il` and `il-interactive`, whose slots are capped and cannot be taken.

**Preemption is not a rollback, it is a restart.** `out_dir` is a fresh
TemporaryDirectory per process, so `resume.pt` lands where the next process never
looks and a requeued job resumes from step 0. Anything preemptible therefore
costs its whole elapsed time, not a 20-minute window -- which is why the
reservation and `il` are spent first and `il-lo` is the last resort.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from roach.slurm.clusters.ilc import ILC  # noqa: E402

from expts.relarena.submit import (  # noqa: E402
    CACHE_DIR,
    EXPERIMENTS,
    REPO_ROOT,
    RESERVATION_WALL,
    SECRETS_DIR,
    SHARE,
    a100,
    b200,
    relarena_setup,
    reserved,
)
from roach.slurm import submit  # noqa: E402

#: Tasks longest-first, so the ones that decide the makespan start earliest.
ORDER = [(d, t) for _m, d, t in EXPERIMENTS]


def plan(model: str) -> list:
    """Where each task of one sweep runs, in submission order.

    `rt-j` first: it takes the reservation's eight cards and the `il` slots
    `rt-plurel` is not using. `rt` follows on `il-lo`, ampere only -- it is the
    control, so it is the one that can afford to wait or be restarted.
    """
    if model == "rt-j":
        # blackwell is filled from il-interactive FIRST. Its two gpus are a
        # pool of their own -- left idle that capacity is simply lost -- while
        # il's b200 sub-cap of 2 is drawn from the same 10-gpu budget the a100
        # jobs need. Spending il's b200 slots before il-interactive's costs a
        # card twice: once on blackwell and once against the a100 allowance.
        #
        # The 12h wall binds only on il-interactive, so the longest tasks go to
        # il (a week) and the ones that clear 12h on a b200 go interactive.
        tiers = (
            [b200("il-interactive", "12:00:00")] * 2
            + [b200("il", "7-00:00:00")] * 1
            + [reserved(RESERVATION_WALL)] * 8
            + [a100("il", "7-00:00:00")] * 5
            + [a100("il-lo", "21-00:00:00")] * 5
        )
    else:
        tiers = [a100("il-lo", "21-00:00:00")] * len(ORDER)
    return tiers[: len(ORDER)]


def main() -> None:
    models = sys.argv[1:] or ["rt-j", "rt"]
    for model in models:
        print(f"=== {model} ===")
        for (dataset, task), resources in zip(ORDER, plan(model)):
            job = submit(
                "expts.relarena.run:main",
                args=dict(
                    dataset=dataset,
                    task=task,
                    model=model,
                    seed=0,
                    n_trials=1,
                    cache_dir=CACHE_DIR,
                    out_dir=f"{SHARE}/results",
                ),
                resources=resources,
                name=f"relarena-{model}-{dataset}-{task}",
                setup=relarena_setup(),
                repo_root=REPO_ROOT,
                cluster=ILC,
                job_env="expts/job_env.sh",
                log_root="~/scratch/relational-transformer/relarena/slurm-logs",
                clone_root="~/roach_clones",
                secrets_dir=SECRETS_DIR,
            )
            print(f"  {model}/{dataset}/{task:22s} {resources.qos:15s} {job.id}")


if __name__ == "__main__":
    main()
