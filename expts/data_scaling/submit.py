"""Submit the data-scaling arms.

The loop is the experiment: same target, same everything, one argument varying.
Run it from a clean, pushed checkout -- the jobs clone the commit you submit
from, so this file is also the record of what was run.

    pixi run python expts/data_scaling/submit.py
"""

from __future__ import annotations

from roach.slurm import AMPERE, BLACKWELL, submit

# Where this cluster keeps things. roach.slurm has no idea about any of it.
REPO_ROOT = "/lfs/hyperturing1/0/ranjanr/clones/rishabh-ranjan/relational-transformer"
# The node's own big disk, not /tmp (which is the 280G root filesystem): clones
# are shared per commit and hold the pixi env, which pixi hardlinks from the
# package cache only when the two are on the same filesystem.
CLONE_ROOT = "/lfs/local/0/roach_clones"
SECRETS_DIR = "/dfs/user/ranjanr/.secrets"
LOG_ROOT = "/dfs/user/ranjanr/slurm-logs/data-scaling"
OUT_ROOT = "/dfs/user/ranjanr/ckpts"
PRE_DIR = "/dfs/user/ranjanr/pre/the-join-preprocessed"
EVAL_PRE_DIR = "/dfs/user/ranjanr/pre/relbench-preprocessed"

WHERE = dict(
    repo_root=REPO_ROOT,
    clone_root=CLONE_ROOT,
    secrets_dir=SECRETS_DIR,
    log_root=LOG_ROOT,
    # a clone survives a week unused before a later job sweeps it; these runs
    # are long, so the arms that follow one another reuse the same one
    clone_ttl_days=7,
)

ARMS = {
    "10pct": "expts/data_scaling/10pct.json",
    "32pct": "expts/data_scaling/32pct.json",
    "all": "expts/data_scaling/all-usable.json",
    "rt-j": f"{PRE_DIR}/db-task-lists/rt-j.json",
}

BASE = dict(
    pre_dir=PRE_DIR,
    eval_pre_dir=EVAL_PRE_DIR,
    out_root=OUT_ROOT,
    project="data-scaling",
)


def main() -> None:
    for arm, task_list in ARMS.items():
        # The control arm gets the fast queue; the rest take what is free.
        resources = AMPERE if arm == "rt-j" else BLACKWELL
        submit(
            "expts.data_scaling.train:train",
            args={**BASE, "db_task_list": task_list},
            resources=resources,
            name=f"ds-{arm}",
            # the rustler sampler is a compiled extension; build it in the clone
            setup=("pixi run build-sampler",),
            **WHERE,
        )


if __name__ == "__main__":
    main()
