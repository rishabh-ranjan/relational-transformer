"""Submit the attention-mask memory probe. See [README.md](README.md)."""

# dataclasses / AMPERE are unused while RESOURCES is blank, and imported so that
# filling it in is one line and not an import hunt.
import dataclasses  # noqa: F401

from roach.slurm.clusters.ilc import AMPERE, ILC  # noqa: F401

from roach.slurm import Resources, submit

# The slot this one job goes in, chosen by hand at submission time.
#
# NOT A DEFAULT TO INHERIT, and blank on purpose: whatever the last submission
# used is a record of a different cluster. Work it out again every time,
# following [Allocating a sweep](../README.md#allocating-a-sweep) -- read the
# cluster, subtract what your own jobs already hold, spend the tiers top down --
# and write today's answer here, e.g.
#
#     RESOURCES = dataclasses.replace(
#         AMPERE, gpus="a100:1", exclusive=False, cpus_per_task=8,
#         qos="il-interactive", time="00:20:00",
#     )
#
# One card, a shared node and a short wall is the shape it wants: the probe
# wants a GPU now, not the run's eight.
RESOURCES: Resources | None = None


def main() -> None:
    assert RESOURCES is not None, (
        "RESOURCES is blank: pick this submission's slot per "
        "../README.md#allocating-a-sweep and write it in"
    )
    submit(
        "expts.mask_mem.probe:main",
        args=dict(
            # The shape the pretraining run is memory-bound at: tokens_per_gpu=2**17 at
            # ctx 8192 gives 16 rows, and MAX_F2P_NBRS is rt.data's.
            batch_size=16,
            seq_len=8192,
            max_f2p_nbrs=5,
            # Roughly a node per 8 feature cells, so `same_node` is a band.
            num_nodes=1024,
            repeats=5,
        ),
        resources=RESOURCES,
        name="mask-mem",
        repo_root="~/clones/rishabh-ranjan/relational-transformer",
        cluster=ILC,
        job_env="expts/job_env.sh",
        log_root="~/scratch/slurm-logs/rishabh-ranjan/relational-transformer/expts/mask-mem",
        clone_root="~/roach_clones",
        secrets_dir="~/scratch/.secrets",
    )


if __name__ == "__main__":
    main()
