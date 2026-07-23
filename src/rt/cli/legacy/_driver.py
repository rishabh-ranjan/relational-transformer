"""Shared driver for the legacy (RT-v1 / RT-PluRel) eval CLIs.

Evaluates a possibly per-task family of legacy checkpoints on the RelBench
tasks with the published legacy context configuration (ctx=1024, one BFS
context around the seed with bfs_width=256, boolean targets kept boolean),
and writes a RelBench leaderboard submission directory of prediction CSVs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from rt.data import get_tasks
from rt.eval.main import build_evaluator
from rt.eval.metrics import metric_for
from rt.eval.relbench import _emit_and_score
from rt.model.legacy._common import LEGACY_EMBEDDING_MODEL


@dataclass
class LegacyEvalConfig:
    # output directory for prediction CSVs (a RelBench submission dir).
    out_dir: str
    # legacy/ holds RelBench re-preprocessed with RT-v1 boolean typing
    # (booleans are a real sem type instead of z-scored numbers).
    pre_dir: str = "stanford-star/relbench-preprocessed/legacy"
    db_task_list: str = "stanford-star/relbench/db-task-lists/forecast.json"
    # False reads clf targets from the boolean head (BCE-trained), matching
    # the legacy models' training. Requires boolean-typed data (legacy/).
    bool_as_num: bool = False
    # published legacy eval context: the whole 1024-token context is one BFS
    # neighborhood around the seed (local_ctx_size == ctx_size), width 256,
    # no random-walk tier (num_walks=0) and no recency-sorted neighbors
    # (prefer_latest=False), matching the original samplers.
    ctx_size: int = 1024
    local_ctx_size: int = 1024
    num_walks: int = 0
    walk_length: int = 0
    bfs_width: int = 256
    prefer_latest: bool = False
    tokens_per_gpu: int = 2**17
    num_workers: int = 2
    # effectively the full test split.
    items_per_task: int = 10_000_000
    shuffle_seed: int = 0
    context_seed: int = 0
    reg_metric: str = "mae"


def run_legacy_eval(cfg: LegacyEvalConfig, model_for_task) -> dict:
    """``model_for_task(task) -> nn.Module`` supplies the (possibly per-task)
    legacy checkpoint, already on-device in bf16."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(cfg.out_dir).expanduser()

    tasks = get_tasks(cfg.pre_dir, cfg.db_task_list, ("test",))
    if not tasks:
        raise SystemExit(f"no tasks found in {cfg.pre_dir}")

    by_metric: dict[str, list[float]] = {}
    results = {}
    print(f"\n{'task':40} {'metric':8} {'value':>9} {'n':>7}  {'align':>11}  debug")
    for task in tasks:
        model = model_for_task(task).to(device).to(torch.bfloat16)
        ev = build_evaluator(
            [task],
            cfg.pre_dir,
            embedding_model=LEGACY_EMBEDDING_MODEL,
            d_text=384,
            device=device,
            ctx_size=cfg.ctx_size,
            local_ctx_size=cfg.local_ctx_size,
            bfs_width=cfg.bfs_width,
            prefer_latest=cfg.prefer_latest,
            num_walks=cfg.num_walks,
            walk_length=cfg.walk_length,
            tokens_per_gpu=cfg.tokens_per_gpu,
            items_per_task=cfg.items_per_task,
            num_workers=cfg.num_workers,
            shuffle_seed=cfg.shuffle_seed,
            context_seed=cfg.context_seed,
            bool_as_num=cfg.bool_as_num,
        )
        for _task, _ctx, labels, preds_by_prefix, _nl, node_idxs in ev.evaluate_raw(
            [(model, "")], [cfg.ctx_size], with_node_idxs=True
        ):
            preds = preds_by_prefix[""]
            mname, mval, n, align, _ = _emit_and_score(
                out_dir, task, cfg.pre_dir, LEGACY_EMBEDDING_MODEL, labels, preds,
                node_idxs, keep_csv=True,
            )
            nm, nv = metric_for(task.task_type, labels, preds, cfg.reg_metric)
            by_metric.setdefault(mname, []).append(mval)
            results[f"{task.db_name}/{task.table_name}"] = {
                "metric": mname, "value": mval, "n": n,
            }
            print(
                f"{task.db_name + '/' + task.table_name:40} {mname:8} {mval:>9.4f} "
                f"{n:>7}  {align:>11}  norm[{nm}]={nv:.4f}",
                flush=True,
            )
        del model, ev
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"\n{'mean':40}")
    for name, vals in by_metric.items():
        print(f"  {name:10} {sum(vals) / len(vals):>9.4f}  (over {len(vals)} tasks)")
    print(f"\nsubmission CSVs written to {out_dir}/  "
          f"(validate: python -m relbench.submit {out_dir})")
    return results
