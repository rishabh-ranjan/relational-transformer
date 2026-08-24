import dataclasses  # noqa: F401

from roach.slurm.clusters.ilc import AMPERE, ILC  # noqa: F401

from roach.slurm import Resources, submit

RESOURCES: Resources | None = None


def main() -> None:
    assert RESOURCES is not None, (
        "RESOURCES is blank: pick this submission's slot per "
        "../README.md#allocating-a-sweep and write it in"
    )
    submit(
        "expts.mask_mem.probe:main",
        args=dict(
            batch_size=16,
            seq_len=8192,
            max_f2p_nbrs=5,
            num_nodes=1024,
            repeats=5,
        ),
        resources=RESOURCES,
        name="mask-mem",
        repo_root="~/clones/rishabh-ranjan/relational-transformer",
        cluster=ILC,
        job_env="expts/job_env.sh",
        log_root="~/scratch/relational-transformer/mask_mem/slurm-logs",
        clone_root="~/roach_clones",
        secrets_dir="~/scratch/.secrets",
    )


if __name__ == "__main__":
    main()
