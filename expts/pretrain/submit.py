"""Submit the pretraining run. See [README.md](README.md)."""

import argparse
import dataclasses
import json
from pathlib import Path

from roach.slurm import AMPERE_LO, Resources, submit

# The checkout this script belongs to, whatever that is, rather than one named
# clone. `submit` refuses a dirty or unpushed tree, and the usual clone is
# shared with other sessions: one of them mid-edit used to take the run down,
# because an autoscale pass cancels before it submits and the submit then
# failed on someone else's uncommitted files. From a `git worktree` (or any
# other clean checkout of the same commit) submitting still works.
REPO_ROOT = Path(__file__).resolve().parents[2]

# The in-loop validation tasks, listed here rather than by path: RelBench's
# published `forecast.json` also carries recommendation tasks, which
# rt.train cannot build a Task from. Read at submit
# time and passed inline -- this repo lives on the submitting host's local
# disk, so a path into it does not resolve on the compute node.
EVAL_TASKS = [
    (db, task)
    for db, task in json.loads(Path(__file__).with_name("eval-tasks.json").read_text())
]


# The pretraining shape, as a function of what the cluster will give right now.
# `nodes` x 8xA100, exclusive: the mixture is populated into each node's page
# cache, which wants the node's whole memory, and cpus_per_task is 128/8 with
# it. See [autoscale.py](autoscale.py), which picks the shape and resubmits.
#
# il-lo is preemptible and effectively uncapped, which is the only way to hold
# more than 10 a100s; `il` is not preemptible but caps a100 at 10 per user, so
# it fits a single node and nothing larger. The run checkpoints and resumes
# either way, so a preemption costs wall clock rather than work.
def resources(nodes: int, qos: str, nodelist: str) -> Resources:
    return dataclasses.replace(
        AMPERE_LO,
        nodes=nodes,
        qos=qos,
        time="21-00:00:00" if qos == "il-lo" else "7-00:00:00",
        exclusive=True,
        cpus_per_task=16,
        nodelist=nodelist,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "run_id",
        nargs="?",
        help="relaunch this run instead of starting a new one; the run id names "
        "the checkpoint directory, so the job resumes where it left off",
    )
    p.add_argument("--nodes", type=int, default=1)
    # No default: the tier is chosen against the cluster at the moment of
    # submission, following
    # [Allocating a sweep](../README.md#allocating-a-sweep) -- read the cluster,
    # subtract what your own jobs already hold, spend the tiers top down.
    # Whatever the last submission used is a record of a different cluster.
    # `autoscale.py` computes it live and always passes it.
    p.add_argument("--qos", required=True, choices=("il-lo", "il"))
    p.add_argument(
        "--nodelist",
        default="ampere1,ampere2,ampere3,ampere4,ampere5,ampere6,ampere7,ampere8,ampere9",
        help="nodes to run on. Naming exactly the free ones is how a job starts "
        "immediately rather than queueing behind whatever else wants them.",
    )
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    submit(
        "rt.train:main",
        args=dict(
            # model: RT-J's dims
            embedder="all-MiniLM-L12-v2",
            d_text=384,
            num_blocks=12,
            d_model=512,
            num_heads=8,
            d_ff=2048,
            compile=True,
            materialize_attn_masks=True,
            loss_fn="huber",
            load_ckpt_path=None,
            # data: the Join's mixture
            db_task_list="/dfs/user/ranjanr/share/stanford-star/the-join-preprocessed/db-task-lists/rt-j.json",
            pre_dir="/dfs/user/ranjanr/share/stanford-star/the-join-preprocessed",
            tokens_per_gpu=2**17,
            # loader workers are processes, and the job only owns
            # `cpus_per_task` cores per task
            num_workers=16,
            prefetch_factor=2,
            ctx_size_list=[512, 1024, 2048, 4096, 8192],
            local_ctx_size_list=[256, 512, 1024, 2048, 4096, 8192],
            bfs_width_list=[8, 16, 32, 64, 128, 256],
            prefer_latest_list=[False, True],
            num_walks=10_000,
            walk_length=20,
            mask_prob_max=0.5,
            items_per_task=100_000,
            # optimization
            lr=5e-4,
            wd=0.1,
            warmup_steps=2_000,
            grad_norm_max=1.0,
            total_bs=1024,
            total_steps=100_001,
            early_stop_after_steps=None,
            swa_momentum=0.9995,
            seed=0,
            mmap_populate=True,
            timeout_per_item=10.0,
            eval_freq=1_000,
            keep_all_ckpts=True,
            vector_db_path=None,
            db_upto_test_timestamp=True,
            resume_save_mins=20.0,
            # in-loop validation: the benchmark's forecast tasks, val split
            eval_splits=["val"],
            eval_db_task_list=EVAL_TASKS,
            eval_pre_dir="/dfs/user/ranjanr/share/stanford-star/relbench-preprocessed",
            # Half of training's: the eval masks are
            # `eval_tokens_per_gpu * 8192` bytes each (2 GB at 2**18), and an
            # eval that fills the card OOMs the training forward after it. Eval
            # is slower for it, and only runs every `eval_freq` steps.
            eval_tokens_per_gpu=2**17,
            eval_num_workers=1,
            eval_prefetch_factor=2,
            eval_num_walks=10_000,
            eval_walk_length=20,
            eval_items_per_task=1024,
            eval_ctx_size_list=[8192],
            eval_mmap_populate=True,
            eval_shuffle_seed=0,
            eval_context_seed=0,
            eval_vector_db_path=None,
            eval_lcs_bw_pl_grid=[(256, 32, True)],
            # logging
            targets={},
            project="2026-08-07-pretrain",
            entity="rtv2",
            run_name="rt-j",
            wandb_disabled=False,
            out_root="/dfs/user/ranjanr/ckpts",
        ),
        resources=resources(args.nodes, args.qos, args.nodelist),
        name="pretrain",
        run_id=args.run_id,
        repo_root=str(REPO_ROOT),
        log_root="/dfs/user/ranjanr/slurm-logs/rishabh-ranjan/relational-transformer/expts/pretrain",
        clone_root="/lfs/local/0/roach_clones",
        secrets_dir="/dfs/user/ranjanr/.secrets",
    )


if __name__ == "__main__":
    main()
