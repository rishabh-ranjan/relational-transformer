#!/usr/bin/env python
"""Evaluate an RT checkpoint on the RelBench benchmark tasks.

Loads a checkpoint -- a local dir/file or a Hub model repo such as
``stanford-star/rt-j/classification`` -- evaluates every task of its kind (clf/reg) in
the preprocessed RelBench data (``--pre-dir``, local or Hub), and reports metrics
through RelBench's own leaderboard evaluator: regression predictions are
denormalized to the original target scale, classification logits are sigmoided
to probabilities, and each prediction is keyed back to its relbench
``(entity_col, time_col)``. ``--out-dir`` becomes a valid RelBench *submission
directory* -- one ``<dataset>__<task>.csv`` prediction table per task -- that is
scored with ``relbench.leaderboard.evaluate_task`` (AUROC for clf, NMAE for reg)
and can be re-validated with ``python -m relbench.leaderboard <out-dir>``.

Single-process (one GPU). Example:

    pixi run eval --checkpoint stanford-star/rt-j/classification \\
        --pre-dir stanford-star/relbench-preprocessed --out-dir eval_out
"""

from __future__ import annotations

import torch

import tempfile
import time
from functools import cache
from pathlib import Path

import lazy_loader as lazy
import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from rt.data import EvalDataset, RustlerDataset, read_meta, resolve_pre_dir
from rt.model import load_rt_model
from rt.rel2tab.config import Rel2TabModelConfig
from rt.config import Config
from rt.data import eval_tasks


def main(cfg: Config) -> None:
    ev_cfg = cfg.eval
    assert cfg.logger.wandb_disabled, "standalone eval does not log to wandb"
    assert ev_cfg.freq is None, "eval.freq is an in-loop training knob"
    assert len(ev_cfg.ctx_sizes) == 1, (
        "standalone eval writes one submission per run and needs exactly one "
        "ctx size; multi-size ctx_sizes is an in-loop training-eval feature"
    )
    ctx_size = ev_cfg.ctx_sizes[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if isinstance(cfg.model, Rel2TabModelConfig):
        # rel2tab tabular baseline: (featurizer, predictor) through the same
        # eval path as RT.
        net = cfg.model.build(device)
        embedding_model = cfg.model.embedding_model
        d_text = cfg.model.d_text
        print(f"baseline: {type(cfg.model.featurizer).__name__} + "
              f"{type(cfg.model.predictor).__name__} on {device}")
    else:
        checkpoint = cfg.model.load_ckpt_path
        assert checkpoint is not None, "model.load_ckpt_path is required"
        net, config = load_rt_model(checkpoint, device=device, compile=False)
        net = net.to(torch.bfloat16)
        embedding_model = config["embedding_model"]
        d_text = config["d_text"]
        print(f"loaded {config.get('name', checkpoint)} "
              f"(embed={embedding_model}) on {device}")
        if config.get("task_type") in ("clf", "reg"):
            print(f"warning: this checkpoint was selected/trained for "
                  f"task_type={config['task_type']}; it will be evaluated on "
                  f"both clf and reg tasks", flush=True)
        # The checkpoint's own config drives model construction; cfg.model dims
        # are ignored here. Warn when they disagree so a stale CLI default is
        # visible rather than silently shadowed.
        ckpt_model = config.get("model", {})
        mismatches = [
            f"{k}: config={v} checkpoint={ckpt_model[k]}"
            for k, v in (("num_blocks", cfg.model.num_blocks),
                         ("d_model", cfg.model.d_model),
                         ("d_text", cfg.model.d_text),
                         ("num_heads", cfg.model.num_heads),
                         ("d_ff", cfg.model.d_ff))
            if k in ckpt_model and ckpt_model[k] != v
        ]
        if cfg.model.embedding_model != embedding_model:
            mismatches.append(f"embedding_model: config={cfg.model.embedding_model} "
                              f"checkpoint={embedding_model}")
        if mismatches:
            print("warning: model config ignored for checkpoint eval; differs from "
                  "the checkpoint's own config: " + "; ".join(mismatches), flush=True)


    eval_kwargs = dict(
        embedding_model=embedding_model, d_text=d_text, device=device,
        num_walks=ev_cfg.num_walks, walk_length=ev_cfg.walk_length,
        tokens_per_gpu=ev_cfg.tokens_per_gpu, items_per_task=ev_cfg.items_per_task,
        num_workers=ev_cfg.num_workers, shuffle_seed=ev_cfg.shuffle_seed,
        prefetch_factor=ev_cfg.prefetch_factor, bool_as_num=ev_cfg.bool_as_num,
        skip_text_cols=ev_cfg.skip_text_cols, mmap_populate=ev_cfg.mmap_populate,
        balance_labels=ev_cfg.balance_labels,
        ablate_schema_semantics=ev_cfg.ablate_schema_semantics,
        vector_db_path=ev_cfg.vector_db_path,
    )
    grid = ev_cfg.lcs_bw_pl_grid

    if len(grid) > 1 or ev_cfg.ensemble_size > 1:
        assert ev_cfg.context_seed == 0, (
            "ensembling sweeps context seeds 0..ensemble_size-1; a fixed "
            "eval.context_seed only applies to single-config runs"
        )

        val_tasks = eval_tasks(ev_cfg.pre_dir, splits=("val",))
        test_tasks = eval_tasks(ev_cfg.pre_dir, splits=("test",))
        if not test_tasks:
            raise SystemExit(f"no tasks found in {ev_cfg.pre_dir}")
        run_ensemble(net, ev_cfg.pre_dir, val_tasks, test_tasks, grid=grid,
                     ensemble_size=ev_cfg.ensemble_size, ctx_size=ctx_size,
                     reg_metric=ev_cfg.reg_metric, out_dir=ev_cfg.out_dir, no_csv=not ev_cfg.write_csv,
                     **eval_kwargs)
        return

        tasks = eval_tasks(ev_cfg.pre_dir, splits=tuple(ev_cfg.splits))
    if not tasks:
        raise SystemExit(f"no tasks found in {ev_cfg.pre_dir}")
    lcs, bw, pl = grid[0]
    ev = build_evaluator(tasks, ev_cfg.pre_dir, ctx_size=ctx_size,
                         local_ctx_size=lcs, bfs_width=bw, prefer_latest=pl,
                         context_seed=ev_cfg.context_seed, **eval_kwargs)
    run_and_report(net, tasks, ev_cfg.pre_dir, ctx_size=ctx_size,
                   reg_metric=ev_cfg.reg_metric, out_dir=ev_cfg.out_dir, no_csv=not ev_cfg.write_csv,
                   evaluator=ev, embedding_model=embedding_model)


def metric_for(task_type: str, labels, preds, reg_metric: str = "mae") -> tuple[str, float]:
    """rt-internal metric on the *normalized* scale (val-set tuning + debug).

    Kept for ensemble context-tuning on the validation split (a relative
    comparison, scale-invariant) and as a labeled debug number alongside the
    authoritative RelBench metric. Not the submission metric.
    """
    import sklearn.metrics as M

    if task_type == "reg":
        if reg_metric == "r2":
            return "r2", float(M.r2_score(labels, preds))
        return "mae", float(M.mean_absolute_error(labels, preds))
    return "roc_auc", float(M.roc_auc_score((labels > 0).astype(int), preds))


# --------------------------------------------------------------------------- #
# RelBench submission: denormalize / sigmoid, key by node index, score.
# --------------------------------------------------------------------------- #
def _relbench():
    try:
        import relbench  # noqa: F401
        from relbench.leaderboard import evaluate_task

        return relbench, evaluate_task
    except Exception as e:  # pragma: no cover - import-time guidance
        raise RuntimeError(
            "relbench is required for evaluation. It is a declared dependency "
            "(relbench @ relbench-hf), e.g. `pixi install`."
        ) from e


@cache
def _seed_offset(pre_dir: str, db: str, table: str, split: str, embedding_model: str) -> int:
    """Global rustler ``node_idx`` of the first row of ``table``'s ``split``
    (so ``node_idx - offset`` is the relbench parquet row index)."""
    local = resolve_pre_dir(pre_dir, [db], embedding_model)
    ti = json.loads((Path(local) / db / "table_info.json").read_text())
    split_cap = {"train": "Train", "val": "Val", "test": "Test"}.get(split, split.capitalize())
    key = f"{table}:Db" if f"{table}:Db" in ti else f"{table}:{split_cap}"
    return int(ti[key]["node_idx_offset"])


@cache
def _load_relbench_task(source: str, table: str):
    relbench, _ = _relbench()
    return relbench.load_task(source, table)


def _train_stats(rtask) -> tuple[float, float]:
    """Train-split target ``(mean, std)`` -- the exact inverse of rustler's
    ``(val - mean) / std`` normalization (``std`` ddof=1, 0 -> 1.0)."""
    df = rtask.get_table("train").df
    col = rtask.target_col
    mean = float(df[col].mean())
    std = float(df[col].std(ddof=1))
    return mean, (std if std != 0.0 else 1.0)


def _emit_and_score(out_dir: Path, task, pre_dir: str, embedding_model: str,
                    labels, preds, node_idxs, *, keep_csv: bool):
    """Denormalize/sigmoid ``preds``, write a relbench prediction-table CSV keyed
    by ``(entity_col, time_col)``, and score it with relbench's evaluator.

    Returns ``(metric_name, metric_value, n, align_str, csv_path | None)``.
    ``align_str`` is a built-in alignment guard: denormalized rt labels are
    compared row-for-row to the relbench ground truth (max abs diff for reg,
    class agreement for clf) -- both should be ~perfect when the node-index
    join is correct.
    """
    import numpy as np

    relbench, evaluate_task = _relbench()

    meta = read_meta(pre_dir, task.db_name)
    source = meta.get("source")
    if not source:
        raise RuntimeError(
            f"{task.db_name}/meta.json has no 'source'; cannot locate the relbench task"
        )
    rtask = _load_relbench_task(source, task.table_name)
    offset = _seed_offset(pre_dir, task.db_name, task.table_name, task.split, embedding_model)

    node_idxs = np.asarray(node_idxs, dtype=np.int64)
    rowidx = node_idxs - offset
    masked = rtask.get_table("test", mask_input_cols=True).df.reset_index(drop=True)
    gt = rtask.get_table("test", mask_input_cols=False).df.reset_index(drop=True)
    n_test = len(masked)
    if rowidx.size and (rowidx.min() < 0 or rowidx.max() >= n_test):
        raise RuntimeError(
            f"{task.db_name}/{task.table_name}: seed row indices out of range "
            f"[{int(rowidx.min())}, {int(rowidx.max())}] vs {n_test} relbench test rows"
        )

    preds = np.asarray(preds, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    gt_vals = gt[rtask.target_col].to_numpy(dtype=np.float64)[rowidx]
    if task.task_type == "reg":
        mean, std = _train_stats(rtask)
        out_preds = preds * std + mean
        align = f"|dy|<={float(np.max(np.abs(labels * std + mean - gt_vals))):.1e}" if rowidx.size else "n/a"
    else:  # clf -> probability in [0, 1]; AUROC is invariant to the sigmoid.
        out_preds = 1.0 / (1.0 + np.exp(-preds))
        agree = float(np.mean((labels > 0).astype(int) == (gt_vals > 0).astype(int))) if rowidx.size else float("nan")
        align = f"cls={agree:.3f}"

    sub = masked.iloc[rowidx][[rtask.entity_col, rtask.time_col]].copy()
    sub[rtask.target_col] = out_preds

    if keep_csv:
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"{task.db_name}__{task.table_name}.csv"
        sub.to_csv(csv_path, index=False)
        score_path, ret_path = csv_path, csv_path
    else:
        tf = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
        sub.to_csv(tf.name, index=False)
        tf.close()
        score_path, ret_path = Path(tf.name), None

    metrics = evaluate_task(f"{task.db_name}/{task.table_name}", str(score_path), dataset=source)
    if ret_path is None:
        Path(score_path).unlink(missing_ok=True)
    mname, mval = next(iter(metrics.items()))
    return mname, float(mval), int(rowidx.size), align, ret_path


def build_evaluator(tasks, pre_dir, *, embedding_model, d_text, device, ctx_size=8192,
                    local_ctx_size=256, bfs_width=32, num_walks=10_000, walk_length=20,
                    tokens_per_gpu=2**18, items_per_task=None, num_workers=2, context_seed=0,
                    prefer_latest=True, shuffle_seed=0, mmap_populate=True,
                    prefetch_factor=2, bool_as_num=True, skip_text_cols=False,
                    balance_labels=False, ablate_schema_semantics=False,
                    vector_db_path=None):
    # mmap_populate=True by default: pre-fault the eval data into RAM so the
    # context build is fed instead of cold-faulting it from shared storage per
    # item (the same starvation that hits training without it).
    #
    # prefer_latest / shuffle_seed are real knobs (not constants): prefer_latest
    # picks the same-table neighbor sort (recency vs frequency); shuffle_seed
    # fixes the val/test subset selection + item shuffle, so an --items-per-task
    # subsample stays the *same* rows across configs (context tuning, ensembling).
    return Evaluator(
        tasks=tasks, pre_dir=pre_dir, eval_bs=max(1, tokens_per_gpu // ctx_size),
        ctx_sizes=[ctx_size], items_per_task=items_per_task, num_workers=num_workers,
        prefetch_factor=prefetch_factor, persistent_workers=False, local_ctx_size=local_ctx_size,
        bfs_width=bfs_width, num_walks=num_walks, walk_length=walk_length,
        prefer_latest=prefer_latest, bool_as_num=bool_as_num, skip_text_cols=skip_text_cols,
        mmap_populate=mmap_populate, balance_labels=balance_labels,
        ablate_schema_semantics=ablate_schema_semantics, embedding_model=embedding_model, d_text=d_text,
        shuffle_seed=shuffle_seed, context_seed=context_seed, vector_db_path=vector_db_path,
        train_only_fallback=False,
        global_rank=0, local_rank=0, world_size=1, ddp=False, device=device,
    )


def run_and_report(model, tasks, pre_dir, *, ctx_size, reg_metric, out_dir, no_csv,
                   evaluator, embedding_model):
    """Run inference, write relbench submission CSVs, score via relbench's
    evaluator, print per-task + mean metrics. Returns a results dict."""
    out_dir = Path(out_dir).expanduser()
    by_metric: dict[str, list[float]] = {}
    results = {}
    print(f"\n{'task':40} {'metric':8} {'value':>9} {'n':>7}  {'align':>11}  debug")
    for task, _ctx, labels, preds_by_prefix, _nl, node_idxs in evaluator.evaluate_raw(
        [(model, "")], [ctx_size], with_node_idxs=True
    ):
        preds = preds_by_prefix[""]
        mname, mval, n, align, _ = _emit_and_score(
            out_dir, task, pre_dir, embedding_model, labels, preds, node_idxs,
            keep_csv=not no_csv,
        )
        nm, nv = metric_for(task.task_type, labels, preds, reg_metric)  # normalized-scale debug
        by_metric.setdefault(mname, []).append(mval)
        results[f"{task.db_name}/{task.table_name}"] = {"metric": mname, "value": mval, "n": n}
        print(f"{task.db_name + '/' + task.table_name:40} {mname:8} {mval:>9.4f} {n:>7}  "
              f"{align:>11}  norm[{nm}]={nv:.4f}")
    print(f"\n{'mean':40}")
    for name, vals in by_metric.items():
        print(f"  {name:10} {sum(vals) / len(vals):>9.4f}  (over {len(vals)} tasks)")
    if not no_csv:
        print(f"\nsubmission CSVs written to {out_dir}/  "
              f"(validate: python -m relbench.leaderboard {out_dir})")
    return results


def _is_better(task_type, a, b):
    return a > b if task_type == "clf" else a < b  # higher auc / lower mae


def run_ensemble(model, pre_dir, val_tasks, test_tasks, *, grid, ensemble_size, ctx_size,
                 reg_metric, out_dir, no_csv, **eval_kwargs):
    """Context-tuned + ensembled evaluation.

    Tune: for each task, pick the (local_ctx_size, bfs_width, prefer_latest) in
    ``grid`` with the best *validation* metric. Ensemble: on test, run that config with
    ``ensemble_size`` context seeds and average the per-item predictions, then
    score the averaged submission through relbench's evaluator.
    """
    from collections import defaultdict

    import numpy as np

    embedding_model = eval_kwargs["embedding_model"]

    # ---- tune on val: best context config per task ----
    best = {}  # (db, table) -> {"cfg", "value", "task_type"}
    for cfg in grid:
        lcs, bw, pl = cfg
        ev = build_evaluator(val_tasks, pre_dir, ctx_size=ctx_size, local_ctx_size=lcs,
                             bfs_width=bw, prefer_latest=pl, **eval_kwargs)
        for task, _c, labels, preds_by_prefix, _nl in ev.evaluate_raw([(model, "")], [ctx_size]):
            _, v = metric_for(task.task_type, labels, preds_by_prefix[""], reg_metric)
            key = (task.db_name, task.table_name)
            if key not in best or _is_better(task.task_type, v, best[key]["value"]):
                best[key] = {"cfg": cfg, "value": v, "task_type": task.task_type}
            print(f"  tune {task.db_name}/{task.table_name} cfg={cfg}: {v:.4f}")

    # ---- ensemble on test: best config per task, averaged over context seeds ----
    groups = defaultdict(list)
    for t in test_tasks:
        b = best.get((t.db_name, t.table_name))
        if b is not None:
            groups[b["cfg"]].append(t)

    out_dir = Path(out_dir).expanduser()
    by_metric: dict[str, list[float]] = {}
    results = {}
    print(f"\n{'task':40} {'cfg':14} {'metric':8} {'value':>9} {'n':>7}  {'align':>11}")
    for cfg, tasks in groups.items():
        lcs, bw, pl = cfg
        acc = {}  # key -> [labels, sum_preds, task, node_idxs]
        for seed in range(ensemble_size):
            ev = build_evaluator(tasks, pre_dir, ctx_size=ctx_size, local_ctx_size=lcs,
                                 bfs_width=bw, prefer_latest=pl, context_seed=seed,
                                 **eval_kwargs)
            for task, _c, labels, preds_by_prefix, _nl, node_idxs in ev.evaluate_raw(
                [(model, "")], [ctx_size], with_node_idxs=True
            ):
                key = (task.db_name, task.table_name)
                p = preds_by_prefix[""].astype(np.float64)
                if key not in acc:
                    acc[key] = [labels, np.zeros_like(p), task, node_idxs]
                acc[key][1] += p
        for key, (labels, sp, task, node_idxs) in acc.items():
            preds = sp / ensemble_size
            mname, mval, n, align, _ = _emit_and_score(
                out_dir, task, pre_dir, embedding_model, labels, preds, node_idxs,
                keep_csv=not no_csv,
            )
            by_metric.setdefault(mname, []).append(mval)
            results[f"{task.db_name}/{task.table_name}"] = {"metric": mname, "value": mval,
                                                            "cfg": cfg, "n": n}
            print(f"{task.db_name + '/' + task.table_name:40} {str(cfg):14} {mname:8} "
                  f"{mval:>9.4f} {n:>7}  {align:>11}")
    print(f"\n{'mean (ensembled)':40}")
    for name, vals in by_metric.items():
        print(f"  {name:10} {sum(vals) / len(vals):>9.4f}  (over {len(vals)} tasks)")
    if not no_csv:
        print(f"\nsubmission CSVs written to {out_dir}/  "
              f"(validate: python -m relbench.leaderboard {out_dir})")
    return results


wandb = lazy.load("wandb")


def fmt_duration(secs):
    m, s = divmod(int(secs), 60)
    return f"{m}m{s:02d}s"


class Evaluator:
    """Standard per-task eval over a fixed task list.

    Build once with sampler/loader knobs; call ``evaluate`` (or
    ``evaluate_raw``) one or more times. Re-using an instance across
    eval points reuses prefetch state and avoids loader rebuild —
    important for the in-loop training eval.

    Synthetic-DB tasks (``"synthetic" in db_name``) are dropped at
    construction; they were never evaluated by the inline path either.
    """

    def __init__(
        self,
        *,
        tasks,
        pre_dir,
        eval_bs,
        ctx_sizes,
        items_per_task,
        num_workers,
        prefetch_factor,
        persistent_workers,
        local_ctx_size,
        bfs_width,
        num_walks,
        walk_length,
        prefer_latest,
        bool_as_num,
        skip_text_cols,
        mmap_populate,
        balance_labels,
        ablate_schema_semantics,
        embedding_model,
        d_text,
        shuffle_seed,
        context_seed,
        vector_db_path,
        train_only_fallback,
        global_rank,
        local_rank,
        world_size,
        ddp,
        device,
    ):
        self.tasks = [t for t in tasks if "synthetic" not in t.db_name]
        self.eval_splits = sorted(set(t.split for t in self.tasks if t.split))
        self.ctx_sizes = ctx_sizes
        self.eval_bs = eval_bs
        self.items_per_task = items_per_task
        self.bool_as_num = bool_as_num
        self.global_rank = global_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.ddp = ddp
        self.device = device

        max_eval_ctx_size = max(ctx_sizes)

        self.eval_loaders = {}
        self.eval_loader_iters = {}

        init_pbar = tqdm(
            total=len(self.tasks),
            desc="load eval data",
            disable=local_rank != 0,
            leave=False,
        )
        init_tic = time.time()
        prefetch_time = 0.0

        for eval_task in self.tasks:
            rustler_dataset = RustlerDataset(
                tasks=[eval_task],
                pre_dir=pre_dir,
                global_rank=global_rank,
                local_rank=local_rank,
                world_size=world_size,
                local_ctx_sizes=[local_ctx_size],
                bfs_widths=[bfs_width],
                num_walks=num_walks,
                walk_length=walk_length,
                prefer_latest=[prefer_latest],
                mask_prob_max=0.0,
                embedding_model=embedding_model,
                d_text=d_text,
                shuffle_seed=shuffle_seed,
                context_seed=context_seed,
                items_per_task=items_per_task,
                quiet=True,
                bool_as_num=bool_as_num,
                ignore_data_errors=False,
                skip_text_cols=skip_text_cols,
                mmap_populate=mmap_populate,
                balance_labels=[balance_labels],
                timeout_per_item=3600.0,
                ablate_schema_semantics=ablate_schema_semantics,
                vector_db_path=vector_db_path,
                train_only_fallback=train_only_fallback,
            )
            eval_dataset = EvalDataset(
                rustler_dataset=rustler_dataset,
                eval_bs=eval_bs,
                eval_ctx_size=max_eval_ctx_size,
            )
            self.eval_loaders[eval_task] = DataLoader(
                eval_dataset,
                batch_size=None,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
                persistent_workers=persistent_workers,
                pin_memory=True,
                # in_order=True guarantees sampler-order yields; with False the
            # row order is worker-completion order — a timing race that breaks
            # cross-seed prediction averaging (context ensembling) on rows
            # written by eval_grid.
            in_order=True,
            )
            _prefetch_tic = time.time()
            self.eval_loader_iters[eval_task] = iter(self.eval_loaders[eval_task])
            prefetch_time += time.time() - _prefetch_tic
            init_pbar.update(1)
        init_pbar.close()
        if local_rank == 0:
            print(
                f"\neval data loaded in"
                f" \033[1m{fmt_duration(time.time() - init_tic)}\033[0m",
                flush=True,
            )
            print(
                f"  prefetch init took \033[1m{fmt_duration(prefetch_time)}\033[0m",
                flush=True,
            )

    def evaluate_raw(self, nets_with_prefix, eval_ctx_sizes_to_use,
                     with_node_idxs=False):
        """Per-task pipeline primitive.

        Drives per-batch forward + DDP gather + ``batch_mask`` filtering
        for every task. Yields one tuple per ``(task, ctx_size)`` on
        rank 0::

            (task, ctx_size, labels_np, preds_by_prefix_np, num_labels_np)

        - ``labels_np``: ``(n_real,)`` per-row labels.
        - ``preds_by_prefix_np``: dict ``prefix → (n_real,) preds``,
          one entry per ``(net, prefix)`` in ``nets_with_prefix``.
        - ``num_labels_np``: ``(n_real,) int64`` per-row count of
          in-context training labels for that row's target column at
          ``ctx_size`` (the ``mean_labels`` source data).

        ``n_real`` is the number of real (non-phantom) rows across all
        ranks for that task — already filtered by ``batch_mask``.

        With ``with_node_idxs=True`` a sixth element ``node_idxs_np`` is
        appended to the yielded tuple: the ``(n_real,) int64`` global
        rustler node index of each row's *seed* (target) node. Because
        rustler assigns a task row the node index ``node_idx_offset + r``
        (``r`` the 0-based row index in the relbench task-table parquet),
        ``node_idx - node_idx_offset`` recovers the exact parquet row, which
        is how :mod:`rt.eval` keys predictions back to the relbench
        ``(entity_col, time_col)`` for a leaderboard submission (eval row
        order is *not* the parquet row order, so a positional join is wrong).

        Other ranks drive every collective but yield nothing.
        """
        device = self.device
        ddp = self.ddp
        world_size = self.world_size
        global_rank = self.global_rank
        local_rank = self.local_rank

        for net, _ in nets_with_prefix:
            net.eval()

        with torch.inference_mode():
            for eval_task, eval_loader_iter in self.eval_loader_iters.items():
                eval_loader = self.eval_loaders[eval_task]

                # The number of eval batches per task MUST be identical on every
                # rank, or NCCL deadlocks (ranks issue a different number of
                # collective calls). ``len(eval_loader.dataset)`` is
                # ``ceil(num_items / (eval_bs * world_size))`` -- uniform across
                # ranks (``num_items`` is the task's total item count, not a
                # per-rank count), and the rustler sampler fills any overshoot
                # slots as phantoms (batch_mask[i]=False). ``items_per_task``
                # only caps how many batches we bother running; the cap is the
                # same integer on every rank, so it never desyncs the schedule.
                n_batches = len(eval_loader.dataset)
                if self.items_per_task is not None:
                    n_batches = min(
                        n_batches,
                        max(1, self.items_per_task // self.eval_bs // world_size),
                    )

                preds_per_prefix_per_ctx = {
                    prefix: {ctx: [] for ctx in eval_ctx_sizes_to_use}
                    for _, prefix in nets_with_prefix
                }
                num_labels_per_ctx = {ctx: [] for ctx in eval_ctx_sizes_to_use}
                labels = []
                batch_masks = []
                node_idxs_acc = []
                pbar = tqdm(
                    total=n_batches,
                    desc=f"{eval_task.db_name}/{eval_task.table_name}/{eval_task.split}",
                    disable=local_rank != 0,
                    leave=False,
                )
                # Drive the loop by the fixed, cross-rank-uniform batch count.
                # Every rank processes exactly ``n_batches`` batches (each of
                # ``eval_bs`` rows, phantom-padded as needed), so every rank
                # contributes exactly ``n_batches * eval_bs`` rows to every
                # collective below -- no StopIteration / local-count breaks.
                for _ in range(n_batches):
                    batch = next(eval_loader_iter)

                    batch_mask = batch.pop("batch_mask")

                    # Per-row in-context training-label count for the
                    # target column, for each requested ctx_size. Gathered
                    # and masked alongside labels/preds so the eventual
                    # mean_labels stat is uniform over real items.
                    for eval_ctx_size in eval_ctx_sizes_to_use:
                        tb = {k: v[:, :eval_ctx_size] for k, v in batch.items()}
                        tb_is_targets = tb["is_targets"]
                        tb_target_col = torch.full(
                            (tb_is_targets.shape[0], 1),
                            -1,
                            dtype=tb["col_name_idxs"].dtype,
                        )
                        tb_target_node = tb_target_col.clone()
                        tb_bidxs, tb_sidxs = tb_is_targets.nonzero(as_tuple=True)
                        tb_target_col[tb_bidxs, 0] = tb["col_name_idxs"][
                            tb_bidxs, tb_sidxs
                        ]
                        tb_target_node[tb_bidxs, 0] = tb["node_idxs"][
                            tb_bidxs, tb_sidxs
                        ]
                        is_label_cell = (
                            tb["is_task_nodes"]
                            & ~tb["is_padding"]
                            & (tb["col_name_idxs"] == tb_target_col)
                            & (tb["node_idxs"] != tb_target_node)
                        )
                        num_labels_per_ctx[eval_ctx_size].append(
                            is_label_cell.sum(dim=1).to(torch.int64)
                        )

                    for net, prefix in nets_with_prefix:
                        preds_by_ctx = net.predict(
                            batch,
                            eval_ctx_sizes_to_use,
                            device,
                            eval_task,
                            bool_as_num=self.bool_as_num,
                        )
                        for ctx_size, yhat in preds_by_ctx.items():
                            assert yhat.size(0) == batch_mask.size(0)
                            preds_per_prefix_per_ctx[prefix][ctx_size].append(yhat)

                    val_key = (
                        "boolean_values"
                        if eval_task.task_type == "clf" and not self.bool_as_num
                        else "number_values"
                    )
                    y = (
                        batch[val_key].squeeze(-1)
                        * batch["is_targets"].to(batch[val_key].dtype)
                    ).sum(dim=1)
                    assert y.size(0) == batch_mask.size(0)
                    labels.append(y)
                    batch_masks.append(batch_mask)
                    if with_node_idxs:
                        # Seed (target) node's global rustler index per row. Exactly
                        # one target cell per real row, so the masked sum picks it
                        # out; phantom rows have no target → 0, dropped by batch_mask.
                        nidx = (
                            batch["node_idxs"].to(torch.int64)
                            * batch["is_targets"].to(torch.int64)
                        ).sum(dim=1)
                        assert nidx.size(0) == batch_mask.size(0)
                        node_idxs_acc.append(nidx)
                    pbar.update(1)

                pbar.close()

                # prefetch next pass while we run gather + metric compute.
                self.eval_loader_iters[eval_task] = iter(eval_loader)

                # Every rank ran exactly ``n_batches`` batches of ``eval_bs``
                # rows, so ``labels_cat`` has the same length on every rank and
                # the all_gathers are inherently in lockstep -- no cross-rank
                # MIN reduce or truncation needed. Phantom rows are filtered out
                # via ``masks_gathered`` on rank 0 after the gather.
                labels_cat = torch.cat(labels, dim=0).to(device)
                masks_cat = torch.cat(batch_masks, dim=0).to(device)
                if ddp:
                    labels_gathered = torch.empty(
                        labels_cat.size(0) * world_size,
                        dtype=labels_cat.dtype,
                        device=device,
                    )
                    masks_gathered = torch.empty(
                        masks_cat.size(0) * world_size,
                        dtype=masks_cat.dtype,
                        device=device,
                    )
                    dist.all_gather_into_tensor(
                        labels_gathered, labels_cat.contiguous()
                    )
                    dist.all_gather_into_tensor(masks_gathered, masks_cat.contiguous())
                else:
                    labels_gathered = labels_cat
                    masks_gathered = masks_cat

                if global_rank == 0:
                    labels_np = labels_gathered[masks_gathered].float().cpu().numpy()

                node_idxs_np = None
                if with_node_idxs:
                    nidx_cat = torch.cat(node_idxs_acc, dim=0).to(device)
                    if ddp:
                        nidx_gathered = torch.empty(
                            nidx_cat.size(0) * world_size,
                            dtype=nidx_cat.dtype,
                            device=device,
                        )
                        dist.all_gather_into_tensor(
                            nidx_gathered, nidx_cat.contiguous()
                        )
                    else:
                        nidx_gathered = nidx_cat
                    if global_rank == 0:
                        node_idxs_np = nidx_gathered[masks_gathered].cpu().numpy()

                for eval_ctx_size in eval_ctx_sizes_to_use:
                    nlabels_cat = torch.cat(
                        num_labels_per_ctx[eval_ctx_size], dim=0
                    ).to(device)
                    if ddp:
                        nlabels_gathered = torch.empty(
                            nlabels_cat.size(0) * world_size,
                            dtype=nlabels_cat.dtype,
                            device=device,
                        )
                        dist.all_gather_into_tensor(
                            nlabels_gathered, nlabels_cat.contiguous()
                        )
                    else:
                        nlabels_gathered = nlabels_cat

                    preds_by_prefix_np = {}
                    for _, prefix in nets_with_prefix:
                        preds = torch.cat(
                            preds_per_prefix_per_ctx[prefix][eval_ctx_size], dim=0
                        ).to(device)
                        if ddp:
                            preds_gathered = torch.empty(
                                preds.size(0) * world_size,
                                dtype=preds.dtype,
                                device=device,
                            )
                            dist.all_gather_into_tensor(
                                preds_gathered, preds.contiguous()
                            )
                            preds = preds_gathered
                        if global_rank == 0:
                            preds_by_prefix_np[prefix] = (
                                preds[masks_gathered].float().cpu().numpy()
                            )

                    if global_rank == 0:
                        num_labels_np = nlabels_gathered[masks_gathered].cpu().numpy()
                        out = (
                            eval_task,
                            eval_ctx_size,
                            labels_np,
                            preds_by_prefix_np,
                            num_labels_np,
                        )
                        if with_node_idxs:
                            out = out + (node_idxs_np,)
                        yield out

    def evaluate(self, nets_with_prefix, eval_ctx_sizes_to_use, steps, reg_metric):
        """Full main.py-style pass: per-task metrics, per-split avg
        aggregation, stdout printing, wandb logging.

        ``nets_with_prefix``: list of ``(net, prefix_str)``. Prefixes
        feed into wandb keys + console labels (e.g. ``""`` for the
        live net, ``"swa_"`` for the SWA snapshot).

        ``reg_metric``: ``"mae"`` (mean absolute error) or ``"r2"``
        (coefficient of determination). Selects the per-task metric
        computed for ``task_type == "reg"`` tasks; both the wandb key
        and the ``all_metrics`` aggregate key are named
        ``avg_{reg_metric}``.

        Returns the empty-prefix net's metrics dict, or the first
        net's if no empty-prefix entry exists.
        """
        assert reg_metric in ("mae", "r2"), (
            f"reg_metric must be 'mae' or 'r2', got {reg_metric!r}"
        )
        eval_tic = time.time()
        local_rank = self.local_rank
        global_rank = self.global_rank

        if local_rank == 0:
            tqdm.write(f"[step {steps}]")

        avg_reg_key = f"avg_{reg_metric}"

        all_metrics = {}
        all_reg_scores = {}
        all_auc_scores = {}
        for _, prefix in nets_with_prefix:
            all_metrics[prefix] = {
                split: {ctx: {} for ctx in eval_ctx_sizes_to_use}
                for split in self.eval_splits
            }
            all_reg_scores[prefix] = {
                (x, y): [] for x in eval_ctx_sizes_to_use for y in self.eval_splits
            }
            all_auc_scores[prefix] = {
                (x, y): [] for x in eval_ctx_sizes_to_use for y in self.eval_splits
            }
        all_mean_labels_reg = {
            (x, y): [] for x in eval_ctx_sizes_to_use for y in self.eval_splits
        }
        all_mean_labels_clf = {
            (x, y): [] for x in eval_ctx_sizes_to_use for y in self.eval_splits
        }

        outer_pbar = tqdm(
            total=len(self.eval_loaders),
            desc=f"eval@{steps}",
            disable=local_rank != 0,
            leave=False,
        )

        last_task = None
        for (
            eval_task,
            eval_ctx_size,
            labels_np,
            preds_by_prefix_np,
            num_labels_np,
        ) in self.evaluate_raw(nets_with_prefix, eval_ctx_sizes_to_use):
            if last_task is not None and eval_task is not last_task:
                outer_pbar.update(1)
            last_task = eval_task

            # Uniform per-real-item average. Length matches labels_np.
            task_mean_labels = float(num_labels_np.mean())
            if eval_task.task_type == "reg":
                all_mean_labels_reg[(eval_ctx_size, eval_task.split)].append(
                    task_mean_labels
                )
            elif eval_task.task_type == "clf":
                all_mean_labels_clf[(eval_ctx_size, eval_task.split)].append(
                    task_mean_labels
                )

            for prefix, preds_np in preds_by_prefix_np.items():
                if eval_task.task_type == "reg":
                    metric_name = reg_metric
                    _, metric = metric_for("reg", labels_np, preds_np, reg_metric)
                    all_reg_scores[prefix][(eval_ctx_size, eval_task.split)].append(
                        metric
                    )
                    metric_str = f"{metric:<6.4f}"
                elif eval_task.task_type == "clf":
                    metric_name = "auc"
                    try:
                        _, metric = metric_for("clf", labels_np, preds_np)
                    except Exception as e:
                        labels_int = [int(x > 0) for x in labels_np]
                        n_classes = len(set(labels_int))
                        n_nan_labels = int(np.isnan(labels_np).sum())
                        n_nan_preds = int(np.isnan(preds_np).sum())
                        tqdm.write(
                            f"\033[31mroc_auc_score failed for "
                            f"{eval_task.db_name}/{eval_task.table_name}/"
                            f"{eval_task.split} ctx={eval_ctx_size}: "
                            f"{type(e).__name__}: {e} | "
                            f"n={len(labels_int)} n_classes={n_classes} "
                            f"n_nan_labels={n_nan_labels} "
                            f"n_nan_preds={n_nan_preds} "
                            f"→ falling back to AUC=0\033[0m"
                        )
                        metric = 0.0
                    all_auc_scores[prefix][(eval_ctx_size, eval_task.split)].append(
                        metric
                    )
                    metric_str = f"{metric * 100:<6.1f}"

                short_db = eval_task.db_name.split("/")[-1].split("-")[1]
                tqdm.write(
                    f"  {f'{prefix}{short_db}/{eval_task.table_name}/{eval_task.split}':<30}"
                    f"ctx: {eval_ctx_size:<5}   "
                    f"{metric_name}: \033[1m{metric_str}\033[0m  "
                    f"mean_labels: \033[1m{task_mean_labels:<5.1f}\033[0m"
                )
                all_metrics[prefix][eval_task.split][eval_ctx_size][
                    (eval_task.db_name, eval_task.table_name)
                ] = metric
                all_metrics[prefix][eval_task.split][eval_ctx_size][
                    (eval_task.db_name, eval_task.table_name, "mean_labels")
                ] = task_mean_labels

        if last_task is not None:
            outer_pbar.update(1)
        outer_pbar.close()

        if global_rank == 0:
            for _, prefix in nets_with_prefix:
                for split in self.eval_splits:
                    for eval_ctx_size in eval_ctx_sizes_to_use:
                        def _avg(xs):
                            # Single-task-type task sets (per-task fine-tuning)
                            # have no scores for the other type.
                            return sum(xs) / len(xs) if xs else float("nan")

                        avg_reg = _avg(all_reg_scores[prefix][(eval_ctx_size, split)])
                        avg_auc = _avg(all_auc_scores[prefix][(eval_ctx_size, split)])
                        wandb.log(
                            {
                                f"{prefix}{avg_reg_key}/{split}/{eval_ctx_size}": avg_reg,
                                f"{prefix}avg_auc/{split}/{eval_ctx_size}": avg_auc,
                            },
                            step=steps,
                        )
                        avg_mean_labels_reg = _avg(
                            all_mean_labels_reg[(eval_ctx_size, split)]
                        )
                        avg_mean_labels_clf = _avg(
                            all_mean_labels_clf[(eval_ctx_size, split)]
                        )
                        all_metrics[prefix][split][eval_ctx_size][avg_reg_key] = avg_reg
                        all_metrics[prefix][split][eval_ctx_size]["avg_auc"] = avg_auc
                        all_metrics[prefix][split][eval_ctx_size][
                            "avg_mean_labels_reg"
                        ] = avg_mean_labels_reg
                        all_metrics[prefix][split][eval_ctx_size][
                            "avg_mean_labels_clf"
                        ] = avg_mean_labels_clf
                        tqdm.write(
                            f"  {f'{prefix}avg/{split}':<30}"
                            f"ctx: {eval_ctx_size:<7}"
                            f"{reg_metric}: \033[1m{avg_reg:<6.4f}\033[0m  "
                            f"auc: \033[1m{avg_auc * 100:<5.1f}\033[0m  "
                            f"mean_labels_reg: \033[1m{avg_mean_labels_reg:<5.1f}\033[0m  "
                            f"mean_labels_clf: \033[1m{avg_mean_labels_clf:<5.1f}\033[0m"
                        )

        if global_rank == 0:
            tasks_by_split: dict[str, list] = {s: [] for s in self.eval_splits}
            for t in self.tasks:
                tasks_by_split[t.split].append(t)
            for _, prefix in nets_with_prefix:
                for split in self.eval_splits:
                    for eval_ctx_size in eval_ctx_sizes_to_use:
                        payload = {
                            "ctx_size": eval_ctx_size,
                            f"{prefix}ctx_scaling/steps={steps}/{split}/{avg_reg_key}": all_metrics[
                                prefix
                            ][split][eval_ctx_size][avg_reg_key],
                            f"{prefix}ctx_scaling/steps={steps}/{split}/avg_auc": all_metrics[
                                prefix
                            ][split][eval_ctx_size]["avg_auc"],
                            f"{prefix}ctx_scaling/steps={steps}/{split}/avg_mean_labels_reg": all_metrics[
                                prefix
                            ][split][eval_ctx_size]["avg_mean_labels_reg"],
                            f"{prefix}ctx_scaling/steps={steps}/{split}/avg_mean_labels_clf": all_metrics[
                                prefix
                            ][split][eval_ctx_size]["avg_mean_labels_clf"],
                        }
                        for t in tasks_by_split[split]:
                            metric_name = reg_metric if t.task_type == "reg" else "auc"
                            base = (
                                f"per_task/{prefix}ctx_scaling/steps={steps}/"
                                f"{t.db_name}/{t.table_name}/{split}"
                            )
                            payload[f"{base}/{metric_name}"] = all_metrics[prefix][
                                split
                            ][eval_ctx_size][(t.db_name, t.table_name)]
                            payload[f"{base}/mean_labels"] = all_metrics[prefix][split][
                                eval_ctx_size
                            ][(t.db_name, t.table_name, "mean_labels")]
                        wandb.log(payload)

        if local_rank == 0:
            tqdm.write(
                f"  eval done in \033[1m{fmt_duration(time.time() - eval_tic)}\033[0m"
            )
        return all_metrics
