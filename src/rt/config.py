from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime


def timestamp() -> str:
    """Unique run id: ``yy-mm-dd_hh-mm-ss_ns``.

    Every rank generates its own, so under DDP the ranks disagree -- only rank
    0's is used (it owns the output directory and hands the resume checkpoint
    to the others). Pass ``--logger.id`` to name a run explicitly, which is
    also how you resume one.

    The time is punctuated with ``-`` rather than ``:``: the id names a wandb
    run (which rejects ``:``) and an output directory, and a build under a path
    containing ``:`` fails in cargo.
    """
    now = time.time_ns()
    return f"{datetime.fromtimestamp(now / 1e9):%y-%m-%d_%H-%M-%S}_{now % 1_000_000_000:09d}"


@dataclass
class ModelConfig:
    embedder: str
    d_text: int
    num_blocks: int
    d_model: int
    num_heads: int
    d_ff: int
    compile: bool
    materialize_attn_masks: bool
    load_ckpt_path: str | None


@dataclass
class TrainConfig:
    db_task_list: list[tuple[str, str]] | str
    """(db, task) pairs, or a path to a JSON file of pairs. The released lists
    ship with the data: <pre_dir>/db-task-lists/<name>.json."""
    pre_dir: str
    """Local directory of preprocessed datasets (one subdir per db). Download it
    up front with `hf download` -- see docs/train.md; it is not fetched on
    demand."""
    tokens_per_gpu: int
    num_workers: int
    prefetch_factor: int
    ctx_size_list: list[int]
    local_ctx_size_list: list[int]
    bfs_width_list: list[int]
    prefer_latest_list: list[bool]
    num_walks: int
    walk_length: int
    mask_prob_max: float
    items_per_task: int
    lr: float
    wd: float
    warmup_steps: int
    grad_norm_max: float
    total_bs: int
    total_steps: int
    swa_momentum: float
    seed: int
    mmap_populate: bool
    timeout_per_item: float
    # How often (in steps) to run in-loop validation. None = never.
    eval_freq: int | None
    # When set, Tier 1 same-table seed selection switches from random
    # walks to FAISS-similarity lookups. Layout is
    # `<vector_db_path>/<db>/<table>.index` and
    # `<vector_db_path>/<db>/<table>_vectors.bin`. When None, behavior
    # is unchanged (random walk + same-table fallback).
    vector_db_path: str | None
    # Also write resume.pt every this many minutes of wall-clock
    # (preemption resilience), on top of the eval-freq save.
    resume_save_mins: float


@dataclass
class EvalConfig:
    # which task splits to evaluate, e.g. ["test"] or ["val", "test"]
    splits: list[str]
    db_task_list: list[tuple[str, str]] | str
    """(db, task) pairs, or a path to a JSON file of pairs. The released lists
    ship with the data: <pre_dir>/db-task-lists/<name>.json."""
    pre_dir: str
    """Local directory of preprocessed datasets (one subdir per db). Download it
    up front with `hf download` -- see docs/train.md; it is not fetched on
    demand."""
    tokens_per_gpu: int
    num_workers: int
    prefetch_factor: int
    num_walks: int
    walk_length: int
    items_per_task: int
    ctx_size_list: list[int]
    mmap_populate: bool
    shuffle_seed: int
    context_seed: int
    # See TrainConfig.vector_db_path.
    vector_db_path: str | None
    # --- standalone evaluation (rt.eval) ---
    # Candidate (local_ctx_size, bfs_width, prefer_latest) context configs.
    # A single entry is used directly; multiple entries are tuned per task on
    # the validation split. In-loop training eval uses the first entry.
    lcs_bw_pl_grid: list[tuple[int, int, bool]]
    # Number of context seeds whose test predictions are averaged; 1 = no
    # ensembling. Grid tuning and/or ensembling engage the val-tuned test path.
    ensemble_size: int


@dataclass
class LoggerConfig:
    project: str
    # wandb entity (team/user). None = the wandb default entity; it is resolved
    # from the live run when wandb is enabled.
    entity: str | None
    # Unique run id; also names the output directory (see ``out_root``). Never
    # None -- CLI entry points default it to a timestamp via ``timestamp()``.
    # Reuse it to resume a run.
    id: str
    # Human-readable label only; id identifies the run.
    run_name: str | None
    wandb_disabled: bool
    # Root for run outputs. The actual output directory is
    # ``<out_root>/<entity>/<project>/<id>/`` and holds checkpoints,
    # resume.pt, config.json and val metrics.
    out_root: str


@dataclass
class Config:
    model: ModelConfig
    train: TrainConfig | None
    eval: EvalConfig
    logger: LoggerConfig
