"""Context-scaling eval: one (method, task) job, per-ctx test metric to disk.

The compute half of the scaling family (baselines figures, retriever and
schema-semantics ablations); ``reduce.py`` aggregates the JSONs into the wandb
runs the paper's figures read. For one method and one task it measures, at
every requested context size, the metric on the normalized scale (NMAE for
regression -- rustler normalizes the target by the same train std RelBench's
NMAE divides out -- and AUROC for classification, which the sigmoid cannot
change) together with the mean number of in-context labeled rows.

The whole ctx sweep is one pass: the evaluator builds each row's context once
at ``max(ctx_size_list)`` and scores every size off a prefix of it.

Methods share the evaluator through one ``predict(batch, ctx_sizes, device,
task)`` contract:

* ``rt``            -- RT-J, routed to the clf / reg checkpoint by task type.
* ``rdblearn_tabicl`` / ``sql_tabicl``  -- batched TabICL v2 on precomputed
  RDBLearn / SQL features (GPU).
* ``rdblearn_lgbm`` / ``sql_lgbm``      -- a stock-defaults LightGBM fit per
  (row, ctx) on the same features (CPU).

Writes ``<out_dir>/<db>__<table>.json`` atomically and skips if it already
exists, so a sweep resubmits idempotently.
"""

import json
import os
import uuid
from collections import OrderedDict
from pathlib import Path

BASELINE_EMBEDDER = "all-MiniLM-L12-v2"
BASELINE_D_TEXT = 384

FEATURES_SUBDIR = {
    "rdblearn_tabicl": "rdblearn_features",
    "rdblearn_lgbm": "rdblearn_features",
    "sql_tabicl": "sql_features",
    "sql_lgbm": "sql_features",
}


def _atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".tmp.{os.getpid()}.{uuid.uuid4().hex}.json"
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    os.replace(tmp, path)


def main(
    *,
    method: str,
    db: str,
    table: str,
    split: str,
    pre_dir: str,
    features_root: str | None,
    out_dir: str,
    ctx_size_list: list[int],
    items_per_task: int,
    local_ctx_size: int,
    bfs_width: int,
    prefer_latest: bool,
    num_walks: int,
    walk_length: int,
    shuffle_seed: int,
    context_seed: int,
    tokens_per_gpu: int,
    num_workers: int,
    prefetch_factor: int,
    mmap_populate: bool,
    db_cutoff: str | int | None,
    vector_db_path: str | None,
    ckpt_clf: str | None,
    ckpt_reg: str | None,
    tabicl_dir: str | None,
    tabicl_max_batch_size: int,
    tabicl_min_bin_size: int,
    tabicl_softmax_temperature: float,
    lgbm_n_jobs: int,
) -> None:
    out_path = Path(out_dir).expanduser() / f"{db}__{table}.json"
    if out_path.exists():
        print(f"{out_path} exists; nothing to do", flush=True)
        return

    import numpy as np
    import torch

    from rt.data import get_tasks
    from rt.eval import build_evaluator
    from rt.eval.metrics import metric_for

    (task,) = get_tasks(pre_dir, [(db, table)], (split,))
    ctx_sizes = sorted(ctx_size_list)

    if method == "rt":
        from rt.model import load_rt_model

        # The compiled net recompiles once per ctx size (each is a distinct
        # sequence-length shape) within the pass; keep the dynamo cache above
        # the sweep's shape count so it never thrashes.
        torch._dynamo.config.cache_size_limit = max(16, 2 * len(ctx_sizes))
        device = "cuda"
        ckpt = ckpt_clf if task.task_type == "clf" else ckpt_reg
        model, config = load_rt_model(ckpt, device=device, compile=True)
        model = model.to(torch.bfloat16)
        embedder, d_text = config["embedder"], config["d_text"]
    else:
        from expts.repaper.baselines.rel2tab.model import Rel2TabModel
        from expts.repaper.baselines.rel2tab.precomputed import PrecomputedFeaturizer

        featurizer = PrecomputedFeaturizer(
            features_root, FEATURES_SUBDIR[method], [(db, table)]
        )
        if method.endswith("_tabicl"):
            from expts.repaper.baselines.rel2tab.tabicl_batched import (
                TabICLBatchedPredictor,
            )

            device = "cuda"
            predictor = TabICLBatchedPredictor(
                max_batch_size=tabicl_max_batch_size,
                min_bin_size=tabicl_min_bin_size,
                softmax_temperature=tabicl_softmax_temperature,
                checkpoint_dir=tabicl_dir,
                device=device,
            )
        else:
            from expts.repaper.baselines.rel2tab.lgbm import LGBMPredictor

            device = "cpu"
            predictor = LGBMPredictor(n_jobs=lgbm_n_jobs)
        model = Rel2TabModel(featurizer, predictor)
        embedder, d_text = BASELINE_EMBEDDER, BASELINE_D_TEXT

    ev = build_evaluator(
        [task],
        pre_dir,
        embedder=embedder,
        d_text=d_text,
        device=device,
        ctx_size_list=ctx_sizes,
        local_ctx_size=local_ctx_size,
        bfs_width=bfs_width,
        prefer_latest=prefer_latest,
        num_walks=num_walks,
        walk_length=walk_length,
        tokens_per_gpu=tokens_per_gpu,
        items_per_task=items_per_task,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        context_seed=context_seed,
        shuffle_seed=shuffle_seed,
        mmap_populate=mmap_populate,
        vector_db_path=vector_db_path,
        db_cutoff=db_cutoff,
    )

    per_ctx: dict[int, dict] = OrderedDict()
    for _task, ctx, labels, preds_by_prefix, num_labels in ev.evaluate_raw(
        [(model, "")], ctx_sizes
    ):
        metric_name, metric_value = metric_for(
            task.task_type, labels, preds_by_prefix[""]
        )
        per_ctx[int(ctx)] = {
            "metric_name": metric_name,
            "metric_value": metric_value if np.isfinite(metric_value) else None,
            "n": int(labels.shape[0]),
            "mean_labels": float(np.mean(num_labels)),
        }
        print(
            f"{db}/{table} ctx={ctx}: {metric_name}={metric_value:.4f} "
            f"(n={labels.shape[0]}, labels={per_ctx[int(ctx)]['mean_labels']:.1f})",
            flush=True,
        )

    _atomic_write_json(
        out_path,
        {
            "method": method,
            "task": f"{db}/{table}",
            "db": db,
            "table": table,
            "task_type": task.task_type,
            "per_ctx": {str(c): per_ctx[c] for c in sorted(per_ctx)},
            "config": {
                "method": method,
                "split": split,
                "ctx_sizes": ctx_sizes,
                "items_per_task": items_per_task,
                "local_ctx_size": local_ctx_size,
                "bfs_width": bfs_width,
                "prefer_latest": prefer_latest,
                "num_walks": num_walks,
                "walk_length": walk_length,
                "shuffle_seed": shuffle_seed,
                "context_seed": context_seed,
                "tokens_per_gpu": tokens_per_gpu,
                "db_cutoff": db_cutoff,
                "vector_db_path": vector_db_path,
                "pre_dir": pre_dir,
                "features_root": features_root,
                "ckpt_clf": ckpt_clf,
                "ckpt_reg": ckpt_reg,
            },
        },
    )
    print(f"wrote {out_path}", flush=True)
