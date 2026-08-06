"""Submit the fine-tuning runs.

    pixi run python expts/fine_tune/submit.py            # batch, 4xB200
    pixi run python expts/fine_tune/submit.py --interactive

The second form runs inside a held interactive allocation instead of queuing
(see src/roach/slurm/README.md): the same target with the same arguments, so a
run debugged there is submitted for real by dropping the flag. Use it while the
recipe is still moving; use the batch form for the run whose number you report.

Run it from a clean, pushed checkout -- the jobs clone the commit you submit
from, so this file is also the record of what was run.
"""

from __future__ import annotations

import sys

from roach.slurm import BLACKWELL, BLACKWELL_INTERACTIVE_1GPU, interactive, submit

# Where this cluster keeps things. roach.slurm has no idea about any of it.
REPO_ROOT = "/lfs/hyperturing1/0/ranjanr/clones/rishabh-ranjan/relational-transformer"
# The node's own big disk, not /tmp (which is the 280G root filesystem): clones
# are shared per commit and hold the pixi env, which pixi hardlinks from the
# package cache only when the two are on the same filesystem.
CLONE_ROOT = "/lfs/local/0/roach_clones"
SECRETS_DIR = "/dfs/user/ranjanr/.secrets"
LOG_ROOT = "/dfs/user/ranjanr/slurm-logs/fine-tune"
OUT_ROOT = "/dfs/user/ranjanr/ckpts"
# Fine-tuning trains on the benchmark data, not the Join: the task it is
# evaluated on is the task it is trained on.
PRE_DIR = "/dfs/user/ranjanr/pre/relbench-preprocessed"

WHERE = dict(
    repo_root=REPO_ROOT,
    clone_root=CLONE_ROOT,
    secrets_dir=SECRETS_DIR,
    log_root=LOG_ROOT,
    clone_ttl_days=7,
)

# The task this arm fine-tunes on, and how long for. Pretraining's values
# throughout (expts/data_scaling/train.py); only the mixture and the number of
# steps are this experiment's own.
DB, TASK = "rel-f1", "driver-top3"
TOTAL_STEPS = 10_001

BASE = dict(
    db_name=DB,
    task_name=TASK,
    # random init: the control that says what the task alone teaches
    load_ckpt_path=None,
    pre_dir=PRE_DIR,
    out_root=OUT_ROOT,
    total_steps=TOTAL_STEPS,
    project="fine-tune",
)


def main(use_interactive: bool = False) -> None:
    if use_interactive:
        held = interactive.find()
        if held is None:
            raise SystemExit(
                "no interactive allocation is held; take one with\n"
                "  from roach.slurm import BLACKWELL_INTERACTIVE, interactive\n"
                f"  interactive.hold(BLACKWELL_INTERACTIVE, log_root='{LOG_ROOT}')"
            )
        # One GPU per run, though the allocation holds two: a fine-tuning run
        # is small, and the second card is what lets the next arm start beside
        # this one instead of after it.
        resources, overlap = BLACKWELL_INTERACTIVE_1GPU, held
    else:
        resources, overlap = BLACKWELL, None
    submit(
        "expts.fine_tune.train:train",
        args=BASE,
        resources=resources,
        name=f"ft-{DB}-{TASK}-scratch",
        # the rustler sampler is a compiled extension; build it in the clone
        setup=("pixi run build-sampler",),
        overlap=overlap,
        **WHERE,
    )


if __name__ == "__main__":
    main(use_interactive="--interactive" in sys.argv[1:])
