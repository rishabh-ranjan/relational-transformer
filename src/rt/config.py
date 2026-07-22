from __future__ import annotations

from dataclasses import dataclass
# rel2tab config modules are lazy-import-cheap: the heavy deps (sklearn,
# xgboost, relbench, ...) are imported inside build()/fit(), not at module load.
from rt.rel2tab.config import Rel2TabModelConfig


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
    # Output directory for checkpoints, resume.pt, config.json, val metrics.
    out_dir: str
    # Also write resume.pt every this many minutes of wall-clock
    # (preemption resilience), on top of the eval-freq save.
    resume_save_mins: float
    # Restrict the pretraining mixture to the databases listed in this
    # file (one db name per line; '#' comments and blank lines ignored).
    # None = use every preprocessed db under pre_dir.
    include_dbs_file: str | None


@dataclass
class EvalConfig:
    # which task splits to evaluate, e.g. ["test"] or ["val", "test"]
    splits: list[str]
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
    # restrict to these tasks; each entry is a 'db' (all its tasks) or
    # 'db/task-table' (one task), e.g. rel-f1/driver-top3. None = all tasks.
    tasks: list[str] | None
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
    # restrict tasks by type (baselines): clf | reg | both.
    task_type: str


@dataclass
class LoggerConfig:
    project: str
    wandb_run_name: str | None
    wandb_disabled: bool


@dataclass
class Config:
    model: ModelConfig | Rel2TabModelConfig
    train: TrainConfig | None
    eval: EvalConfig
    logger: LoggerConfig
