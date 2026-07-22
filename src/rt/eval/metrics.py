"""rt-internal metric on the normalized scale (val tuning + debug numbers)."""

from __future__ import annotations

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
