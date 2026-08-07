"""Submit the pretraining run. See [README.md](README.md)."""

import dataclasses
import json
from pathlib import Path

from roach.slurm import AMPERE_LO, submit

# The in-loop validation tasks, listed here rather than by path: RelBench's
# published `forecast.json` also carries recommendation (link_prediction)
# tasks, which rt.train cannot build a Task from and dies on. Read at submit
# time and passed inline -- this repo lives on the submitting host's local
# disk, so a path into it does not resolve on the compute node.
EVAL_TASKS = [
    (db, task)
    for db, task in json.loads(Path(__file__).with_name("eval-tasks.json").read_text())
]

# 8xA100, exclusive, on the preemptible il-lo queue. Exclusive takes the whole
# node (and its memory) for the mixture's page cache, so cpus_per_task goes back
# to 128/8. ampere9 is misbehaving: list every other ampere node instead.
AMPERE_LO_EXCL = dataclasses.replace(
    AMPERE_LO,
    exclusive=True,
    cpus_per_task=16,
    nodelist="ampere1,ampere2,ampere3,ampere4,ampere5,ampere6,ampere7,ampere8",
)


def main() -> None:
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
            swa_momentum=0.9995,
            seed=0,
            mmap_populate=True,
            timeout_per_item=10.0,
            eval_freq=1_000,
            keep_all_ckpts=True,
            vector_db_path=None,
            resume_save_mins=20.0,
            # in-loop validation: the benchmark's forecast tasks, val split
            eval_splits=["val"],
            eval_db_task_list=EVAL_TASKS,
            eval_pre_dir="/dfs/user/ranjanr/share/stanford-star/relbench-preprocessed",
            eval_tokens_per_gpu=2**18,
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
        # 8xA100 on il-lo: preemptible, and the run checkpoints, so the low
        # priority queue costs wall clock, not work
        resources=AMPERE_LO_EXCL,
        name="pretrain",
        repo_root="/lfs/hyperturing1/0/ranjanr/clones/rishabh-ranjan/relational-transformer",
        log_root="/dfs/user/ranjanr/slurm-logs/rishabh-ranjan/relational-transformer/expts/pretrain",
        clone_root="/lfs/local/0/roach_clones",
        secrets_dir="/dfs/user/ranjanr/.secrets",
    )


if __name__ == "__main__":
    main()
