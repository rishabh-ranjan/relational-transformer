import json
import os
import pickle
import uuid
from collections import OrderedDict
from pathlib import Path

# The preprocessed collection's text embedder. build_evaluator and the global
# retriever's label source both read the same data, so they take it from here
# rather than each carrying a literal that could drift.
EMBEDDER = "all-MiniLM-L12-v2"
D_TEXT = 384


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
    features_root: str,
    out_dir: str,
    ctx_size_list: list[int],
    sampler_ctx_size: int,
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
    tabicl_dir: str,
    tabicl_max_batch_size: int,
    tabicl_min_bin_size: int,
    tabicl_softmax_temperature: float,
    lgbm_n_jobs: int,
    exaone_ensemble_count: int,
    tabfm_backend: str,
    tabpfn_dir: str,
    tabpfn_n_estimators: int | str,
    tabpfn_fit_mode: str,
    retriever: str,
    context_sampler: str,
    context_split: str,
    tags: dict,
) -> None:
    out_path = Path(out_dir).expanduser() / f"{db}__{table}.json"
    if out_path.exists():
        print(f"{out_path} exists; nothing to do", flush=True)
        return

    import numpy as np

    from expts.repaper.baselines.rel2tabv2.build import build_rel2tab
    from rt.data import get_tasks
    from rt.eval import build_evaluator
    from rt.eval.metrics import metric_for

    (task,) = get_tasks(pre_dir, [(db, table)], (split,))
    ctx_sizes = sorted(ctx_size_list)

    # What the evaluator is told its context size is, which is NOT the
    # predictor's context size under a query-independent retriever. The
    # evaluator's number is a per-query cell count: the rustler sampler pads
    # every batch to it (3.2 GiB for eight queries at 262144) and
    # eval_bs = tokens_per_gpu // it. The global retriever discards that context
    # entirely, so a run passes something small here and keys its real row
    # counts under it. For retriever="sampler" the context IS the evaluator's,
    # so the two must agree.
    if retriever == "sampler":
        assert sampler_ctx_size == max(ctx_sizes), (
            f"retriever='sampler' uses the evaluator's context, so "
            f"sampler_ctx_size ({sampler_ctx_size}) must be max(ctx_size_list) "
            f"({max(ctx_sizes)})"
        )
        eval_ctx_sizes = ctx_sizes
        ctx_keys = ctx_sizes
    else:
        assert len(ctx_sizes) == 1, (
            f"a query-independent retriever keys its context by the evaluator's "
            f"single context size, so pass one row count, got {ctx_sizes}"
        )
        eval_ctx_sizes = [sampler_ctx_size]
        ctx_keys = [sampler_ctx_size]

    model, device = build_rel2tab(
        method=method,
        db=db,
        table=table,
        features_root=features_root,
        tabicl_dir=tabicl_dir,
        tabicl_max_batch_size=tabicl_max_batch_size,
        tabicl_min_bin_size=tabicl_min_bin_size,
        tabicl_softmax_temperature=tabicl_softmax_temperature,
        lgbm_n_jobs=lgbm_n_jobs,
        exaone_ensemble_count=exaone_ensemble_count,
        tabfm_backend=tabfm_backend,
        tabpfn_dir=tabpfn_dir,
        tabpfn_n_estimators=tabpfn_n_estimators,
        tabpfn_fit_mode=tabpfn_fit_mode,
        retriever=retriever,
        context_sampler=context_sampler,
        context_seed=context_seed,
        context_split=context_split,
        # for the global retriever these are context ROW counts, not cells
        n_rows_list=ctx_sizes,
        ctx_keys=ctx_keys,
        pre_dir=pre_dir,
        embedder=EMBEDDER,
        d_text=D_TEXT,
    )

    ev = build_evaluator(
        [task],
        pre_dir,
        embedder=EMBEDDER,
        d_text=D_TEXT,
        device=device,
        ctx_size_list=eval_ctx_sizes,
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
    saved_preds: dict[str, np.ndarray] = {}
    # rows_for maps the evaluator's context key back to the number of context
    # rows the predictor actually saw, so a result is readable without knowing
    # how the run was configured.
    rows_for = dict(zip(ctx_keys, ctx_sizes))
    effective = getattr(model.retriever, "effective_rows", None)
    for _task, ctx, labels, preds_by_prefix, num_labels in ev.evaluate_raw(
        [(model, "")], eval_ctx_sizes
    ):
        metric_name, metric_value = metric_for(
            task.task_type, labels, preds_by_prefix[""]
        )
        assert np.isfinite(metric_value), (
            f"{db}/{table} ctx={ctx}: {metric_name}={metric_value}"
        )
        per_ctx[int(ctx)] = {
            "metric_name": metric_name,
            "metric_value": metric_value,
            "n": int(labels.shape[0]),
            "mean_labels": float(np.mean(num_labels)),
            # The experiment's axis. n_context_rows is what was asked for and
            # n_context_rows_effective what the train split could supply.
            "n_context_rows": int(rows_for[int(ctx)]),
            "n_context_rows_effective": (
                int(effective[int(ctx)]) if effective else int(rows_for[int(ctx)])
            ),
        }
        preds = np.asarray(preds_by_prefix[""], dtype=np.float64)
        saved_preds[f"preds_{int(ctx)}"] = preds
        saved_preds["labels"] = np.asarray(labels, dtype=np.float64)
        print(
            f"{db}/{table} ctx={ctx}: {metric_name}={metric_value:.4f} "
            f"(n={labels.shape[0]}, labels={per_ctx[int(ctx)]['mean_labels']:.1f})",
            flush=True,
        )
        print(
            f"  preds: {len(np.unique(preds))} distinct, "
            f"min {preds.min():.4f} p50 {np.median(preds):.4f} "
            f"max {preds.max():.4f} mean {preds.mean():.4f} std {preds.std():.4f}",
            flush=True,
        )

    result = {
        "tags": tags,
        "method": method,
        "task": f"{db}/{table}",
        "db": db,
        "table": table,
        "task_type": task.task_type,
        "per_ctx": {int(c): per_ctx[c] for c in sorted(per_ctx)},
        "labels": saved_preds.get("labels"),
        "preds": {
            int(k.removeprefix("preds_")): v
            for k, v in saved_preds.items()
            if k.startswith("preds_")
        },
    }

    _atomic_write_json(
        out_path,
        {
            "method": method,
            "task": f"{db}/{table}",
            "db": db,
            "table": table,
            "task_type": task.task_type,
            "per_ctx": {str(c): per_ctx[c] for c in sorted(per_ctx)},
            # Whatever the sweep varies, named by the sweep rather than inferred
            # from features_root's spelling downstream.
            "tags": tags,
            "config": {
                "method": method,
                "retriever": retriever,
                "context_sampler": context_sampler,
                "context_split": context_split,
                "split": split,
                "ctx_sizes": ctx_sizes,
                "sampler_ctx_size": sampler_ctx_size,
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
            },
        },
    )
    # The metric alone cannot distinguish a real fit from a degenerate one, and
    # rerunning an arm to find out costs more than the 702 floats do.
    preds_path = out_path.with_name(f"{db}__{table}_preds.npz")
    np.savez(preds_path, **saved_preds)

    pickle_path = out_path.with_name(f"{db}__{table}.pkl")
    result["config"] = json.loads(out_path.read_text())["config"]
    tmp = pickle_path.parent / f".tmp.{os.getpid()}.{uuid.uuid4().hex}.pkl"
    with open(tmp, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, pickle_path)

    print(f"wrote {out_path}, {preds_path} and {pickle_path}", flush=True)
