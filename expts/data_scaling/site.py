"""Where this cluster keeps things, and what a job may ask it for.

rt.slurm has no defaults on purpose -- it does not know about any particular
cluster. These constants are that knowledge, in one place, for the experiments
in this directory.
"""

from __future__ import annotations

from rt.slurm import Resources

REPO_ROOT = "/lfs/hyperturing1/0/ranjanr/clones/rishabh-ranjan/relational-transformer"
CLONE_ROOT = "/tmp/ranjanr/clones"
SECRETS_DIR = "/dfs/user/ranjanr/.secrets"
LOG_ROOT = "/dfs/user/ranjanr/slurm-logs/data-scaling"
OUT_ROOT = "/dfs/user/ranjanr/ckpts"
PRE_DIR = "/dfs/user/ranjanr/pre/the-join-preprocessed"
EVAL_PRE_DIR = "/dfs/user/ranjanr/pre/relbench-preprocessed"

SITE = dict(
    repo_root=REPO_ROOT,
    clone_root=CLONE_ROOT,
    secrets_dir=SECRETS_DIR,
    log_root=LOG_ROOT,
)

# 8xA100 on the fast queue. --exclusive because the mixture is populated into
# the page cache and wants the whole node's memory; the `il` QOS caps a100 at 10
# per user, so only one of these runs at a time.
AMPERE = Resources(
    partition="il",
    account="infolab",
    qos="il",
    time="7-00:00:00",
    gpus="a100:8",
    cpus_per_task=16,
    exclusive=True,
    mem=None,
    constraint="ampere",
    nodelist=None,
)

# The same, on the low-priority queue: preemptible, but not subject to the
# 10-a100 cap. Not --exclusive, and 14 cpus per gpu is the site limit for
# non-exclusive ampere jobs.
AMPERE_LO = Resources(
    partition="il",
    account="infolab",
    qos="il-lo",
    time="21-00:00:00",
    gpus="a100:8",
    cpus_per_task=14,
    exclusive=False,
    mem=None,
    constraint="ampere",
    nodelist=None,
)

# 4xB200. `il` caps b200 at 2 per user, so four is only reachable under il-lo.
# Memory is explicit: the site's default for this node is below what the
# mixture needs resident, and 1500000M is the most MaxMemPerCPU allows here.
BLACKWELL = Resources(
    partition="il",
    account="infolab",
    qos="il-lo",
    time="21-00:00:00",
    gpus="b200:4",
    cpus_per_task=36,
    exclusive=False,
    mem="1500000M",
    constraint=None,
    nodelist="blackwell1",
)
