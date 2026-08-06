"""Hold an allocation, then run inside it -- the loop for quick iteration.

A batch job is the right shape for a run you leave: it queues, it survives
preemption, it requeues. It is the wrong shape for a session where you run
something, read the traceback, fix a line and run it again, because every one of
those attempts queues again for scarce cards, and a crash gives the node back.

So: take the allocation once and keep it.

    from roach.slurm import BLACKWELL_INTERACTIVE, interactive, submit

    job = interactive.hold(BLACKWELL_INTERACTIVE, log_root=LOG_ROOT)   # once
    submit(..., resources=BLACKWELL_INTERACTIVE, overlap=job)          # per run

`hold` submits a job that does nothing but sit on the hardware, so it is the
allocation that is durable and each run is a disposable step of it. A run that
fails, or that you cancel with `scancel --signal`/`scancel <jobid>.<stepid>`,
leaves the allocation untouched and the next attempt starts in seconds.

What you give up is everything the batch path gives a long run: no requeue after
preemption (the interactive QOS is not preempted, but a node reboot ends it), no
resume, and a wall clock the QOS caps at 12h. Do not park a training run here.

    interactive.find()    # the id of the allocation being held, if there is one
    interactive.release() # give it back
"""

import os
import subprocess

from roach.slurm.resources import Resources

NAME = "roach-hold"
"""Job name for a held allocation, so `find` can recognise one."""


def hold(resources: Resources, *, log_root: str | os.PathLike, name: str = NAME) -> str:
    """Submit a job that holds `resources` and does nothing, and return its id.

    It sleeps for the resources' wall clock and is *not* requeued: a held
    allocation that came back silently after a preemption would be an allocation
    nobody knows the age of.
    """
    if (existing := find(name)) is not None:
        print(f"{name}: already held by job {existing}")
        return existing
    flags = [
        f"--job-name={name}",
        *resources.sbatch_flags(),
        "--chdir=/tmp",
        "--propagate=MEMLOCK",
        "--no-requeue",
        f"--output={os.fspath(log_root)}/{name}_%j.out",
        "--export=NONE",
        # Long enough that the wall clock, not the sleep, ends it.
        "--wrap=sleep infinity",
    ]
    env = {
        k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SBATCH_"))
    }
    out = subprocess.run(
        ["sbatch", *flags], capture_output=True, text=True, check=True, env=env
    )
    job_id = out.stdout.split()[-1]
    print(f"{name}: holding job {job_id} ({resources.gpus} on {resources.nodelist})")
    return job_id


def find(name: str = NAME) -> str | None:
    """The id of the running held allocation, or None.

    Only a RUNNING one counts: a pending hold has no node yet, and a step
    submitted against it would fail rather than wait.
    """
    out = subprocess.run(
        [
            "squeue",
            "-h",
            "-u",
            os.environ.get("USER", ""),
            "-n",
            name,
            "-t",
            "R",
            "-o",
            "%i",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    ids = out.stdout.split()
    return ids[0] if ids else None


def release(name: str = NAME) -> None:
    """Give the allocation back. Steps running inside it die with it."""
    job_id = find(name)
    if job_id is None:
        print(f"{name}: nothing held")
        return
    subprocess.run(["scancel", job_id], check=True)
    print(f"{name}: released job {job_id}")
