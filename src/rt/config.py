from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime


def default_wandb_id() -> str:
    """Unique run id: ``yy-mm-dd_hh:mm:ss_ns``.

    Every rank builds its own config, so a freshly generated timestamp would
    differ per rank -- and the ranks must agree, since the id names the shared
    output dir. The launcher therefore generates the timestamp once and exports
    it as ``RT_WANDB_ID`` (see the ``train`` pixi task); an explicit
    ``--logger.wandb-id`` still wins over both.
    """
    run_id = os.environ.get("RT_WANDB_ID")
    if run_id:
        return run_id
    now = time.time_ns()
    return f"{datetime.fromtimestamp(now / 1e9):%y-%m-%d_%H:%M:%S}_{now % 1_000_000_000:09d}"


@dataclass
class ModelConfig:
    embedding_model: str
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
    """(db, task) pairs, a local JSON file of pairs, or a Hub path like
    stanford-star/the-join/db-task-lists/forecast.json (only that file downloads)."""
    pre_dir: str
    tokens_per_gpu: int
    num_workers: int
    prefetch_factor: int
    ctx_sizes: list[int]
    local_ctx_sizes: list[int]
    bfs_widths: list[int]
    num_walks: int
    walk_length: int
    prefer_latest: list[bool]
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
    bool_as_num: bool
    skip_text_cols: bool
    mmap_populate: bool
    balance_labels: list[bool]
    timeout_per_item: float
    # When set, Tier 1 same-table seed selection switches from random
    # walks to FAISS-similarity lookups. Layout is
    # `<vector_db_path>/<db>/<table>.index` and
    # `<vector_db_path>/<db>/<table>_vectors.bin`. When None, behavior
    # is unchanged (random walk + same-table fallback).
    vector_db_path: str | None
    # Root for run outputs. The actual output directory is
    # ``<out_root>/<wandb entity>/<wandb project>/<wandb run id>/`` and holds
    # checkpoints, resume.pt, config.json and val metrics.
    out_root: str
    # Also write resume.pt every this many minutes of wall-clock
    # (preemption resilience), on top of the eval-freq save.
    resume_save_mins: float
    # Restrict the pretraining mixture to the databases listed in this
    # file (one db name per line; '#' comments and blank lines ignored).
    # None = use every preprocessed db under pre_dir.


@dataclass
class EvalConfig:
    # which task splits to evaluate, e.g. ["test"] or ["val", "test"]
    splits: list[str]
    db_task_list: list[tuple[str, str]] | str
    """(db, task) pairs, a local JSON file of pairs, or a Hub path like
    stanford-star/relbench/db-task-lists/forecast.json."""
    pre_dir: str
    tokens_per_gpu: int
    num_workers: int
    prefetch_factor: int
    num_walks: int
    walk_length: int
    freq: int | None
    items_per_task: int
    ctx_sizes: list[int]
    bool_as_num: bool
    skip_text_cols: bool
    mmap_populate: bool
    balance_labels: bool
    ablate_schema_semantics: bool
    reg_metric: str
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
    # output directory for prediction CSVs (a RelBench submission dir).
    out_dir: str
    # skip writing per-item prediction CSVs.
    write_csv: bool


@dataclass
class LoggerConfig:
    project: str
    # wandb entity (team/user). None = the wandb default entity; it is resolved
    # from the live run when wandb is enabled.
    wandb_entity: str | None
    # Unique wandb run id; also names the output directory (see
    # TrainConfig.out_root). Never None -- CLI entry points default it to a
    # timestamp via ``default_wandb_id()``. Reuse it to resume a run.
    wandb_id: str
    # Human-readable label only; wandb_id identifies the run.
    wandb_run_name: str | None
    wandb_disabled: bool


@dataclass
class Config:
    model: ModelConfig
    train: TrainConfig | None
    eval: EvalConfig
    logger: LoggerConfig
