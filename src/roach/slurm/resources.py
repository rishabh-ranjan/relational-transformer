"""What a job asks slurm for.

Every field is required: a resource request is a deliberate choice, and a
default here would be an experiment silently making it for you. The presets at
the bottom are this cluster's usable shapes, each carrying the constraint that
forced it.

Validation is structural only -- a cluster's own caps (memory per cpu, cpus per
gpu, gpus per QOS) are policy, and the scheduler reports them better than a
stale copy in here would.
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
    """`<type>:<count>`, e.g. "a100:8", or a bare `<count>` for any type the
    eligible nodes offer -- which is what lets one job shape be scheduled across
    nodes that carry different cards. The count is also the number of ranks: one
    task per GPU, which is what makes DDP work (see roach.slurm.run).

    `"0"` asks for no GPU at all, and runs a single rank. A cpu-only stage of a
    pipeline that wants no accelerator should not have to hold one: GPUs are the
    scarcest thing on a node, and a job that keeps one idle is capping how many
    of its siblings can run."""
    cpus_per_task: int
    """Per rank, not per node -- srun starts one task per GPU by default."""
    ntasks: int | None
    """Ranks to start. None means one per GPU, which is what DDP wants and what
    roach.slurm.run assumes. Set it to 1 to give a single process several GPUs:
    a stage that parallelises inside one process (sentence-transformers spawning
    a worker per device, say) wants all of them visible to one rank, not one
    rank each."""
    exclusive: bool
    mem: str | None
    """None leaves it to the partition default, which is usually what you want;
    an explicit value is capped by the site's MaxMemPerCPU."""
    mem_per_gpu: str | None
    """Memory per GPU, as an alternative to `mem`.

    Needed wherever a partition sets DefMemPerGPU: that default is applied when
    working out whether a job fits, and `--mem` does not displace it, so the
    most GPUs a job can hold is RealMemory / DefMemPerGPU however little memory
    it actually wants. On this cluster (DefMemPerGPU=240000M) that is 3 GPUs on
    a 770G node and 8 on a 2T one -- a job that wants a whole node's cards
    cannot ask for them with `mem` at all. `--mem-per-gpu` replaces the default
    and lifts it."""
    constraint: str | None
    nodelist: str | None

    def __post_init__(self) -> None:
        kind, sep, count = self.gpus.rpartition(":")
        if (sep and not kind) or not count.isdigit():
            raise ValueError(
                f"gpus must be '<count>' or '<type>:<count>', got {self.gpus!r}"
            )
        if sep and int(count) == 0:
            raise ValueError(f"no GPUs is '0', not a type with none: {self.gpus!r}")
        if self.cpus_per_task < 1:
            raise ValueError(f"cpus_per_task must be >= 1, got {self.cpus_per_task}")
        if self.ntasks is not None and self.ntasks < 1:
            raise ValueError(f"ntasks must be >= 1 or None, got {self.ntasks}")
        if self.mem and self.mem_per_gpu:
            raise ValueError("give mem or mem_per_gpu, not both")

    @property
    def ranks(self) -> int:
        """How many tasks srun starts: `ntasks` if given, else one per GPU."""
        if self.ntasks is not None:
            return self.ntasks
        return max(1, int(self.gpus.rpartition(":")[2]))

    def sbatch_flags(self) -> list[str]:
        flags = [
            f"--partition={self.partition}",
            f"--account={self.account}",
            f"--qos={self.qos}",
            f"--time={self.time}",
            "--nodes=1",
            f"--ntasks-per-node={self.ranks}",
            f"--cpus-per-task={self.cpus_per_task}",
        ]
        if self.gpus != "0":
            flags.append(f"--gres=gpu:{self.gpus}")
        if self.exclusive:
            flags.append("--exclusive")
        if self.mem:
            flags.append(f"--mem={self.mem}")
        if self.mem_per_gpu:
            flags.append(f"--mem-per-gpu={self.mem_per_gpu}")
        if self.constraint:
            flags.append(f"--constraint={self.constraint}")
        if self.nodelist:
            flags.append(f"--nodelist={self.nodelist}")
        return flags


# --------------------------------------------------------------------------- #
# This cluster's usable shapes. Each one is a scheduler constraint in disguise;
# the comments are what it cost to find out.
# --------------------------------------------------------------------------- #

AMPERE = Resources(
    partition="il",
    account="infolab",
    qos="il",
    time="7-00:00:00",  # the `il` QOS caps wall clock here; the partition allows 21d under il-lo
    gpus="a100:8",
    cpus_per_task=16,  # 128 cores / 8 ranks
    ntasks=None,
    exclusive=True,  # the mixture is populated into the page cache: take the node's memory
    mem=None,  # --exclusive + DefMemPerGPU gives 2017232M; an explicit --mem is capped lower
    mem_per_gpu=None,
    constraint="ampere",
    nodelist=None,
)
"""8xA100 on the fast queue. `il` caps a100 at 10 per user, so one of these at a time."""

AMPERE_LO = Resources(
    partition="il",
    account="infolab",
    qos="il-lo",
    time="21-00:00:00",
    gpus="a100:8",
    cpus_per_task=14,  # the site allows 14 cpus per gpu when not --exclusive
    ntasks=None,
    exclusive=False,  # these nodes carry unrelated cpu-only jobs; demanding the node just queues
    mem=None,
    mem_per_gpu=None,
    constraint="ampere",
    nodelist=None,
)
"""The same hardware on the low-priority queue: preemptible, but outside the
10-a100 cap, so it runs alongside an AMPERE job."""

BLACKWELL = Resources(
    partition="il",
    account="infolab",
    qos="il-lo",  # `il` caps b200 at 2 per user, so four is only reachable here
    time="21-00:00:00",
    gpus="b200:4",
    cpus_per_task=36,
    ntasks=None,
    exclusive=False,
    mem="1500000M",  # the node's default for 4 gpus is below what the mixture needs resident
    mem_per_gpu=None,
    constraint=None,
    nodelist="blackwell1",
)
"""4xB200. Roughly 3x an 8xA100 job's throughput in the runs measured so far."""

BLACKWELL_INTERACTIVE = Resources(
    partition="il",
    account="infolab",
    qos="il-interactive",  # priority 1500 -- the highest here, and it preempts nothing
    time="12:00:00",  # the QOS cap; a held allocation has to be renewed after that
    gpus="b200:2",  # the QOS caps *all* gpus at 2 per user, so this is the whole budget
    cpus_per_task=36,  # 288 cores / 8 gpus on blackwell1, the site's per-gpu share
    ntasks=None,
    exclusive=False,
    mem="750000M",  # 2 gpus' share of the node, under MaxMemPerCPU=10700 x 72 cpus
    mem_per_gpu=None,
    constraint=None,
    nodelist="blackwell1",
)
"""2xB200 on the interactive QOS: the shape to *hold* while iterating.

Use it when the work is a quick edit-run-look loop rather than a run you submit
and leave. Hold it once with ``roach.slurm.interactive.hold`` and put every run
inside it with ``submit(..., overlap=<job id>)``: the runs are steps of one
allocation, so nothing queues between attempts (seconds, not the minutes a fresh
b200 job waits), and a run that crashes -- or that you cancel -- takes its step
down and leaves the allocation standing for the next one.

Two GPUs is the whole interactive budget (`il-interactive` caps gpus at 2 per
user), and 12 hours is the QOS wall clock, so a loop that outlives a working day
needs the allocation renewed. Anything that should survive a night, a preemption
or your laptop closing is a batch job on `BLACKWELL`/`AMPERE` instead: an
interactive allocation is not requeued, and its runs are not resumed for you."""
