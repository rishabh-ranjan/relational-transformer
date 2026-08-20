"""Context-ensembling curve: one (variant, task) job, metric at every size.

Evaluates one task at one fixed context configuration with ``n_seeds``
independent context seeds (the same seed family ``rt.eval`` ensembles draw
from), averaging the raw per-row predictions over the first k seeds and
scoring the average on the normalized scale at every k = 1..n_seeds. Writes

    <out_dir>/<db>__<table>.json     {"curve": {k: metric}, ...}
    <out_dir>/<db>__<table>.state.npz  (resume: per-seed prediction sums)

A preemption costs at most one seed. ``reduce.py`` aggregates the per-task
curves into the two wandb runs the ensembling figure reads.
"""

import json
import os
import uuid
from pathlib import Path


def main(
    *,
    variant: str,
    db: str,
    table: str,
    ctx_size: int,
    local_ctx_size: int,
    bfs_width: int,
    prefer_latest: bool,
    n_seeds: int,
    items_per_task: int,
    split: str,
    pre_dir: str,
    out_dir: str,
    num_walks: int,
    walk_length: int,
    shuffle_seed: int,
    context_seed: int,
    tokens_per_gpu: int,
    num_workers: int,
    prefetch_factor: int,
    mmap_populate: bool,
    db_cutoff: str | int | None,
    ckpt_clf: str,
    ckpt_reg: str,
) -> None:
    out = Path(out_dir).expanduser()
    final_path = out / f"{db}__{table}.json"
    if final_path.exists():
        print(f"{final_path} exists; nothing to do", flush=True)
        return
    state_path = out / f"{db}__{table}.state.npz"

    import numpy as np
    import torch

    from rt.data import get_tasks
    from rt.eval import build_evaluator
    from rt.eval._eval import member_context_seed
    from rt.eval.metrics import metric_for
    from rt.model import load_rt_model

    (task,) = get_tasks(pre_dir, [(db, table)], (split,))
    ckpt = ckpt_clf if task.task_type == "clf" else ckpt_reg
    model, config = load_rt_model(ckpt, device="cuda", compile=True)
    model = model.to(torch.bfloat16)

    curve: dict[str, float] = {}
    sum_preds = labels0 = nodes0 = None
    start = 0
    if state_path.exists():
        st = np.load(state_path)
        sum_preds, labels0, nodes0 = st["sum_preds"], st["labels"], st["node_idxs"]
        start = int(st["seeds"])
        curve = json.loads(str(st["curve"]))
        print(f"resumed at seed {start}", flush=True)

    for seed in range(start, n_seeds):
        ev = build_evaluator(
            [task],
            pre_dir,
            embedder=config["embedder"],
            d_text=config["d_text"],
            device="cuda",
            ctx_size_list=[ctx_size],
            local_ctx_size=local_ctx_size,
            bfs_width=bfs_width,
            prefer_latest=prefer_latest,
            num_walks=num_walks,
            walk_length=walk_length,
            tokens_per_gpu=tokens_per_gpu,
            items_per_task=items_per_task,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            context_seed=member_context_seed(context_seed, seed),
            shuffle_seed=shuffle_seed,
            mmap_populate=mmap_populate,
            vector_db_path=None,
            db_cutoff=db_cutoff,
        )
        ((_t, _ctx, labels, preds_by_prefix, _nl, node_idxs),) = list(
            ev.evaluate_raw([(model, "")], [ctx_size], with_node_idxs=True)
        )
        p = preds_by_prefix[""].astype(np.float64)
        if sum_preds is None:
            sum_preds, labels0, nodes0 = np.zeros_like(p), labels, node_idxs
        # shuffle_seed fixes the subset and in_order=True the sequence, so
        # every seed sees the same rows in the same order.
        assert np.array_equal(nodes0, node_idxs) and np.array_equal(labels0, labels)
        sum_preds += p
        k = seed + 1
        mname, mval = metric_for(task.task_type, labels0, sum_preds / k)
        curve[str(k)] = mval
        print(f"{db}/{table} [{variant}] ens={k}: {mname}={mval:.4f}", flush=True)

        out.mkdir(parents=True, exist_ok=True)
        tmp = out / f".tmp.{os.getpid()}.{uuid.uuid4().hex}.npz"
        np.savez(
            tmp,
            sum_preds=sum_preds,
            labels=labels0,
            node_idxs=nodes0,
            seeds=k,
            curve=json.dumps(curve),
        )
        os.replace(tmp, state_path)
        del ev

    final = {
        "variant": variant,
        "task": f"{db}/{table}",
        "db": db,
        "table": table,
        "task_type": task.task_type,
        "metric": "roc_auc" if task.task_type == "clf" else "mae",
        "curve": curve,
        "config": {
            "ctx_size": ctx_size,
            "local_ctx_size": local_ctx_size,
            "bfs_width": bfs_width,
            "prefer_latest": prefer_latest,
            "n_seeds": n_seeds,
            "items_per_task": items_per_task,
            "shuffle_seed": shuffle_seed,
            "context_seed": context_seed,
            "db_cutoff": db_cutoff,
            "split": split,
        },
    }
    tmp = out / f".tmp.{os.getpid()}.{uuid.uuid4().hex}.json"
    tmp.write_text(json.dumps(final, indent=2, sort_keys=True))
    os.replace(tmp, final_path)
    print(f"wrote {final_path}", flush=True)
