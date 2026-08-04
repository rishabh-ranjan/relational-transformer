"""Run python functions on slurm: one function, one job, one rank per GPU.

    from rt.slurm import Resources, submit

    submit("rt.train:main", args={...}, resources=Resources(...), name="ds-10pct",
           repo_root=..., log_root=..., clone_root=..., secrets_dir=...)

The job clones the commit you submitted from, builds its environment on the
node, and calls the target in every rank. Preemption is handled by the target
saving a resumable checkpoint; slurm requeues, and the same run id resumes it.
"""

from rt.slurm.resources import Resources
from rt.slurm.run import resolve
from rt.slurm.submit import Job, check_args, submit, timestamp

__all__ = ["Job", "Resources", "check_args", "resolve", "submit", "timestamp"]
