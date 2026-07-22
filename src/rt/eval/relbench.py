"""RelBench submission scoring: denormalize / sigmoid predictions, key rows
back to the relbench parquet, write submission CSVs, score with relbench's own
evaluator."""

from __future__ import annotations

import json
import tempfile
from functools import cache
from pathlib import Path

from rt.data import read_meta, resolve_pre_dir

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
