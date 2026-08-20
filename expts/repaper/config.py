"""What every experiment under expts/repaper shares, in one place.

A rerun edits this file and nothing else: ``RUN_TAG`` names the wandb projects
(a new tag keeps a new round's runs apart from an old one's), ``CKPT_*`` point
at the checkpoint under evaluation. Paths below are where the artifacts of one
round live; a new round with a new checkpoint must clear the checkpoint-
dependent result directories first (see README.md).
"""

# Prefix of every wandb project this round logs to (rtv2/<RUN_TAG>-repaper-*).
RUN_TAG = "2026-08-19"

# The RT-J checkpoints under evaluation: local mirrors of the Hub repo
# (stanford-star/rt-j), one per task type. Compute nodes have no Hub access.
CKPT_CLF = "/dfs/user/ranjanr/share/stanford-star/rt-j/classification"
CKPT_REG = "/dfs/user/ranjanr/share/stanford-star/rt-j/regression"

# Data every node can read.
PRE_DIR = "/dfs/user/ranjanr/share/stanford-star/relbench-preprocessed"
RAW_DIR = "/dfs/user/ranjanr/share/stanford-star/relbench"
JOIN_PRE_DIR = "/dfs/user/ranjanr/share/stanford-star/the-join-preprocessed"

# Shared artifacts this round produces (features, FAISS indices, TabICL
# checkpoints, the classic-relbench cache, the semantics-ablated data).
SHARE = "/dfs/user/ranjanr/share/relational-transformer/repaper"

# Per-experiment results and logs. `rt.train` / `rt.eval` take CKPT_ROOT and
# write under <CKPT_ROOT>/<entity>/<project>/<run>; the runners here take an
# explicit directory under OUT_ROOT.
CKPT_ROOT = "/dfs/user/ranjanr/ckpts"
OUT_ROOT = f"{CKPT_ROOT}/rtv2"
LOG_ROOT = "/dfs/user/ranjanr/slurm-logs/rishabh-ranjan/relational-transformer/expts"
CLONE_ROOT = "/lfs/local/0/roach_clones"
SECRETS_DIR = "/dfs/user/ranjanr/.secrets"


def project(name: str) -> str:
    """wandb project for one experiment of this round."""
    return f"{RUN_TAG}-repaper-{name}"
