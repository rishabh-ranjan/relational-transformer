"""Submit the attention-mask memory probe. See [README.md](README.md)."""

import dataclasses

from roach.slurm import AMPERE, submit


def main() -> None:
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
        resources=dataclasses.replace(
            AMPERE,
            # One card, shared node, short wall: the probe wants a GPU now, not
            # the run's eight.
            gpus="a100:1",
            exclusive=False,
            cpus_per_task=8,
            qos="il-interactive",
            time="00:20:00",
        ),
        name="mask-mem",
        repo_root="/lfs/hyperturing1/0/ranjanr/clones/rishabh-ranjan/relational-transformer",
        log_root="/dfs/user/ranjanr/slurm-logs/rishabh-ranjan/relational-transformer/expts/mask-mem",
        clone_root="/lfs/local/0/roach_clones",
        secrets_dir="/dfs/user/ranjanr/.secrets",
    )


if __name__ == "__main__":
    main()
