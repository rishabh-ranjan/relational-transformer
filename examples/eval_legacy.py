"""Reproduce the published RT-v1 / RT-PluRel leaderboard submissions.

The context configuration below is the one those papers used: the whole 1024
token context is a single BFS neighborhood around the seed (local_ctx_size ==
ctx_size), width 256, no random-walk tier and no recency-sorted neighbors.

    pixi run python examples/eval_legacy.py
"""

from rt.eval.legacy import run
from rt.model.legacy.plurel import PLUREL_HUB_REPO, PluRelTransformer
from rt.model.legacy.v1 import V1_HUB_REPO, V1Transformer

# Legacy data: RelBench re-preprocessed with the RT-v1-era boolean typing, which
# is what those nets' BCE-trained boolean head expects.
PRE_DIR = "data/relbench-preprocessed/legacy"

CONTEXT = dict(
    ctx_size=1024,
    local_ctx_size=1024,
    num_walks=0,
    walk_length=0,
    bfs_width=256,
    prefer_latest=False,
    tokens_per_gpu=2**18,
    items_per_task=10_000_000,
    num_workers=2,
    prefetch_factor=2,
    shuffle_seed=0,
    context_seed=0,
    mmap_populate=True,
    vector_db_path=None,
)


def eval_v1(out_dir: str = "eval_v1_out") -> None:
    """Task-wise checkpoints, each pretrained with its target database held out."""

    def model_for_task(task):
        filename = f"pretrain_{task.db_name}_{task.table_name}.pt"
        print(f"loading {V1_HUB_REPO}/{filename}")
        return V1Transformer.from_pretrained(filename, repo_id=V1_HUB_REPO)

    run(
        model_for_task,
        out_dir=out_dir,
        pre_dir=PRE_DIR,
        db_task_list=f"{PRE_DIR}/db-task-lists/forecast.json",
        **CONTEXT,
    )


def eval_plurel(mode: str = "synth", out_dir: str = "eval_plurel_out") -> None:
    """`synth` is one synthetic-only checkpoint for every task; `synth-real` is
    the task-wise continued-pretraining checkpoint."""

    def model_for_task(task):
        filename = (
            "synth.pt"
            if mode == "synth"
            else f"synth_real_{task.db_name}_{task.table_name}.pt"
        )
        print(f"loading {PLUREL_HUB_REPO}/{filename}")
        return PluRelTransformer.from_pretrained(filename, repo_id=PLUREL_HUB_REPO)

    run(
        model_for_task,
        out_dir=out_dir,
        pre_dir=PRE_DIR,
        db_task_list=f"{PRE_DIR}/db-task-lists/forecast.json",
        **{**CONTEXT, "bfs_width": 128},  # the PluRel paper's width
    )


if __name__ == "__main__":
    eval_v1()
