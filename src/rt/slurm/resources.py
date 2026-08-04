"""What a job asks slurm for.

Deliberately site-agnostic: every field is required and nothing is defaulted, so
a cluster's account/partition/QOS live with the experiment (see expts/site.py),
not in the library. Validation here is structural only -- a cluster's own caps
(memory per cpu, cpus per gpu, gpus per QOS) are policy, and the scheduler
reports them better than a stale copy in here would.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Resources:
    partition: str
    account: str
    qos: str
    time: str
    """Wall clock as slurm spells it, e.g. "7-00:00:00"."""
    gpus: str
    """`<type>:<count>`, e.g. "a100:8". The count is also the number of ranks:
    one task per GPU, which is what makes DDP work (see rt.slurm.run)."""
    cpus_per_task: int
    """Per rank, not per node -- srun starts one task per GPU."""
    exclusive: bool
    mem: str | None
    """None leaves it to the partition default, which is usually what you want;
    an explicit value is capped by the site's MaxMemPerCPU."""
    constraint: str | None
    nodelist: str | None

    def __post_init__(self) -> None:
        kind, _, count = self.gpus.partition(":")
        if not kind or not count.isdigit() or int(count) < 1:
            raise ValueError(f"gpus must be '<type>:<count>', got {self.gpus!r}")
        if self.cpus_per_task < 1:
            raise ValueError(f"cpus_per_task must be >= 1, got {self.cpus_per_task}")

    @property
    def ntasks(self) -> int:
        """One rank per GPU."""
        return int(self.gpus.rpartition(":")[2])

    def sbatch_flags(self) -> list[str]:
        flags = [
            f"--partition={self.partition}",
            f"--account={self.account}",
            f"--qos={self.qos}",
            f"--time={self.time}",
            "--nodes=1",
            f"--ntasks-per-node={self.ntasks}",
            f"--gres=gpu:{self.gpus}",
            f"--cpus-per-task={self.cpus_per_task}",
        ]
        if self.exclusive:
            flags.append("--exclusive")
        if self.mem:
            flags.append(f"--mem={self.mem}")
        if self.constraint:
            flags.append(f"--constraint={self.constraint}")
        if self.nodelist:
            flags.append(f"--nodelist={self.nodelist}")
        return flags
