"""Smoke-test the pretraining launch path, multi-node, in about a minute.

Same entrypoint, launcher and cluster shape as [submit.py](submit.py), but on
rel-f1 alone and for 20 steps, so what it exercises is the *launch*: two nodes
joining one process group, the init-time broadcast of the model across them,
and a training step running to completion.

That broadcast is the fragile part: GPUDirect RDMA on the `ampere` nodes wedges
it -- every rank enqueues the collective, none ever starts it, and the job burns
its whole NCCL watchdog timeout looking like a slow start (`NCCL_NET_GDR_LEVEL=0`
in `rt.train.setup_dist` is what keeps it off). Run this before trusting a
multi-node shape after touching anything about distributed setup, or on nodes a
run has just hung on:

```
pixi run python expts/pretrain/smoke.py                          # 2 nodes, wherever
pixi run python expts/pretrain/smoke.py --nodelist ampere3,ampere9
```

A pass is `time_to_first_step` in the log within a minute or so of the ranks
starting, then `saved: best_clf` and a clean exit. A hang is silence, and means
the RDMA workaround is not holding on those nodes -- do not start a real
multi-node run.
Data is small and page-cache population is off, so a slow start is a real
signal here rather than the tens of minutes the full mixture costs on a node
whose page cache is cold.

Checkpoints are throwaway and go to `/tmp`; nothing is logged to wandb.
"""

import argparse
import dataclasses

from roach.slurm import AMPERE_LO, submit

# rel-f1 is the smallest preprocessed database, so loading it is seconds. Two
# tasks rather than one so the eval path builds a real task list.
TASKS = [("rel-f1", "driver-dnf"), ("rel-f1", "driver-top3")]
PRE_DIR = "/dfs/user/ranjanr/share/stanford-star/relbench-preprocessed"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nodes", type=int, default=2)
    p.add_argument("--qos", default="il-lo", choices=("il-lo", "il"))
    p.add_argument(
        "--nodelist",
        default="ampere1,ampere2,ampere3,ampere4,ampere5,ampere6,ampere7,ampere8,ampere9",
        help="nodes to run on. Name the exact pair you want to test when you "
        "are testing nodes rather than code.",
    )
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    submit(
        "rt.train:main",
        args=dict(
            # model: RT-J's dims, so the broadcast under test is the real one
            embedder="all-MiniLM-L12-v2",
            d_text=384,
            num_blocks=12,
            d_model=512,
            num_heads=8,
            d_ff=2048,
            compile=False,
            materialize_attn_masks=True,
            loss_fn="huber",
            load_ckpt_path=None,
            # data: one small database, no page-cache population
            db_task_list=TASKS,
            train_splits=["train"],
            pre_dir=PRE_DIR,
            tokens_per_gpu=2**15,
            num_workers=4,
            prefetch_factor=2,
            ctx_size_list=[512],
            local_ctx_size_list=[256],
            bfs_width_list=[8],
            prefer_latest_list=[False],
            num_walks=1_000,
            walk_length=20,
            mask_prob_max=0.5,
            items_per_task=1_000,
            # optimization: enough steps to prove the loop turns over
            delta_finetune=False,
            optimizer="muon",
            lr=5e-4,
            wd=0.1,
            lr_warmup_steps=10,
            lr_decay_steps=0,
            grad_norm_max=1.0,
            total_bs=64,
            total_steps=20,
            early_stop_after_steps=None,
            swa_momentum=0.9995,
            seed=0,
            mmap_populate=False,
            timeout_per_item=10.0,
            eval_freq=1_000_000,  # the final eval at the last step is enough
            keep_all_ckpts=False,
            vector_db_path=None,
            db_upto_test_timestamp=True,
            resume_save_mins=1e9,
            eval_splits=["val"],
            eval_db_task_list=TASKS,
            eval_pre_dir=PRE_DIR,
            eval_tokens_per_gpu=2**15,
            eval_num_workers=1,
            eval_prefetch_factor=2,
            eval_num_walks=1_000,
            eval_walk_length=20,
            eval_items_per_task=64,
            eval_ctx_size_list=[512],
            eval_mmap_populate=False,
            eval_shuffle_seed=0,
            eval_context_seed=0,
            eval_vector_db_path=None,
            eval_lcs_bw_pl_grid=[(256, 32, True)],
            # throwaway outputs
            targets={},
            project="2026-08-07-pretrain-smoke",
            entity="rtv2",
            run_name="smoke",
            wandb_disabled=True,
            out_root="/tmp/ranjanr/pretrain-smoke/ckpts",
        ),
        resources=dataclasses.replace(
            AMPERE_LO,
            nodes=args.nodes,
            qos=args.qos,
            time="0-01:00:00",
            exclusive=True,
            cpus_per_task=16,
            nodelist=args.nodelist,
        ),
        name="pretrain-smoke",
        repo_root="/lfs/hyperturing1/0/ranjanr/clones/rishabh-ranjan/relational-transformer",
        # Its own log directory: MONITOR.md identifies the live run by the
        # newest log under the run's log_root, and a smoke test landing there
        # would answer with its own run_id.
        log_root="/dfs/user/ranjanr/slurm-logs/rishabh-ranjan/relational-transformer/expts/pretrain/smoke",
        clone_root="/lfs/local/0/roach_clones",
        secrets_dir="/dfs/user/ranjanr/.secrets",
    )


if __name__ == "__main__":
    main()
