"""RelBench submission scoring: denormalize / sigmoid predictions, key rows
back to the relbench parquet, write submission CSVs, score with relbench's own
evaluator."""

import json
import tempfile
from functools import cache
from pathlib import Path

import numpy as np
import relbench
import sklearn.metrics as M
from relbench.submit import evaluate_task

from rt.data import read_meta, resolve_pre_dir


@cache
def _seed_offset(pre_dir: str, db: str, table: str, split: str, embedder: str) -> int:
    """Global rustler ``node_idx`` of the first row of ``table``'s ``split``
    (so ``node_idx - offset`` is the relbench parquet row index)."""
    local = resolve_pre_dir(pre_dir)
    ti = json.loads((Path(local) / db / "table_info.json").read_text())
    split_cap = {"train": "Train", "val": "Val", "test": "Test"}.get(
        split, split.capitalize()
    )
    key = f"{table}:Db" if f"{table}:Db" in ti else f"{table}:{split_cap}"
    return int(ti[key]["node_idx_offset"])


@cache
def _load_relbench_task(source: str, table: str):
    return relbench.load_dataset(source).load_task(table)


def _train_stats(rtask) -> tuple[float, float]:
    """Train-split target ``(mean, std)`` -- the exact inverse of rustler's
    ``(val - mean) / std`` normalization (``std`` ddof=1, 0 -> 1.0)."""
    df = rtask.get_table("train").df
    col = rtask.target_col
    mean = float(df[col].mean())
    std = float(df[col].std(ddof=1))
    return mean, (std if std != 0.0 else 1.0)


def _emit_and_score(
    csv_out_dir: Path | None,
    task,
    pre_dir: str,
    embedder: str,
    labels,
    preds,
    node_idxs,
):
    """Denormalize/sigmoid ``preds``, write a relbench prediction-table CSV keyed
    by ``(entity_col, time_col)``, and score it with relbench's evaluator.
    ``csv_out_dir=None`` scores via a temp file and keeps no CSV.

    Returns ``(metric_name, metric_value, n, align_str, csv_path | None)``.
    ``align_str`` is a built-in alignment guard: denormalized rt labels are
    compared row-for-row to the relbench ground truth (max abs diff for reg,
    class agreement for clf) -- both should be ~perfect when the node-index
    join is correct.
    """

    meta = read_meta(pre_dir, task.db_name)
    source = meta.get("source")
    if not source:
        raise RuntimeError(
            f"{task.db_name}/meta.json has no 'source'; cannot locate the relbench task"
        )
    rtask = _load_relbench_task(source, task.table_name)
    offset = _seed_offset(pre_dir, task.db_name, task.table_name, task.split, embedder)

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
        align = (
            f"|dy|<={float(np.max(np.abs(labels * std + mean - gt_vals))):.1e}"
            if rowidx.size
            else "n/a"
        )
    else:  # clf -> probability in [0, 1]; AUROC is invariant to the sigmoid.
        out_preds = 1.0 / (1.0 + np.exp(-preds))
        agree = (
            float(np.mean((labels > 0).astype(int) == (gt_vals > 0).astype(int)))
            if rowidx.size
            else float("nan")
        )
        align = f"cls={agree:.3f}"

    sub = masked.iloc[rowidx][[rtask.entity_col, rtask.time_col]].copy()
    sub[rtask.target_col] = out_preds

    if csv_out_dir is not None:
        csv_out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = csv_out_dir / f"{task.db_name}__{task.table_name}.csv"
        sub.to_csv(csv_path, index=False)
        score_path, ret_path = csv_path, csv_path
    else:
        tf = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
        sub.to_csv(tf.name, index=False)
        tf.close()
        score_path, ret_path = Path(tf.name), None

    if rowidx.size < n_test:
        # Subsampled run (eval.items_per_task < test size): relbench's evaluator
        # refuses CSVs that do not cover the full test set, so score the
        # subsample here with the same metric definitions it uses -- AUROC for
        # clf; NMAE = MAE / train-split target std (ddof=1) for reg.

        if ret_path is None:
            Path(score_path).unlink(missing_ok=True)
        if task.task_type == "reg":
            mean, std = _train_stats(rtask)
            return (
                "nmae~",
                float(M.mean_absolute_error(gt_vals, out_preds) / std),
                int(rowidx.size),
                align,
                ret_path,
            )
        return (
            "roc_auc~",
            float(M.roc_auc_score((gt_vals > 0).astype(int), out_preds)),
            int(rowidx.size),
            align,
            ret_path,
        )

    metrics = evaluate_task(
        f"{task.db_name}/{task.table_name}", str(score_path), dataset=source
    )
    if ret_path is None:
        Path(score_path).unlink(missing_ok=True)
    mname, mval = next(iter(metrics.items()))
    return mname, float(mval), int(rowidx.size), align, ret_path
