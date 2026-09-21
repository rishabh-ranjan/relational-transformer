import json
from pathlib import Path


def task_moments(args):
    import ml_dtypes  # noqa: F401
    import numpy as np

    path, n, d, chunk = args
    x = np.memmap(path, dtype=ml_dtypes.bfloat16, mode="r", shape=(n, d))
    s = np.zeros(d, dtype=np.float64)
    ss = np.zeros(d, dtype=np.float64)
    for lo in range(0, n, chunk):
        v = np.asarray(x[lo : lo + chunk]).astype(np.float64)
        s += v.sum(0)
        ss += (v * v).sum(0)
    return s, ss


def main(
    *,
    features_root: str,
    out_npz: str,
    d_feat: int,
    min_rows: int,
    chunk: int,
    workers: int,
) -> None:
    import time
    from concurrent.futures import ProcessPoolExecutor

    import numpy as np

    # The adapter's input is badly conditioned: rt-j's norm_out is an RMSNorm,
    # which fixes the norm of each *row* (sd/mean 0.0049 measured) and says
    # nothing about a column across rows. Four massive-activation dims (119,
    # 206, 303, 399) carry ~90% of the squared row norm, dim 303 alone has mean
    # -13.07, and per-dimension sigma spans 79x. dL/dW is an outer product with
    # that input, so those dims own the gradient.
    #
    # This measures the mean and sigma of each dimension under the *training*
    # marginal -- task t drawn with P prop-to min(rows, 100_000), then a row
    # uniformly within it -- so that a frozen (x - mu) / sigma in front of the
    # adapter standardises exactly the distribution the optimiser sees.
    #
    # It is free at the function level for the linear arm: TabPFN z-norms every
    # column against the context, and z-norm is invariant to a positive affine
    # map per column, so identity-after-standardise is the same predictor as
    # identity. Only the geometry of the parameter space changes.
    root = Path(features_root).expanduser()
    jobs, keys, weights = [], [], []
    skipped = {"rows": 0, "degenerate": 0}
    for mp in sorted(root.glob("*/*_meta.json")):
        m = json.loads(mp.read_text())
        n, d = m["shape"]
        assert d == d_feat, f"{mp} has d={d}"
        if n < min_rows:
            skipped["rows"] += 1
            continue
        if (m["n_distinct_labels"] or 0) < 2:
            skipped["degenerate"] += 1
            continue
        jobs.append(
            (str(mp.with_name(mp.name.replace("_meta.json", "_target.bin"))), n, d, chunk)
        )
        keys.append(f"{mp.parent.name}/{m['task']}")
        weights.append(float(min(m["total_nodes"], 100_000)))
    n_rows = np.array([j[1] for j in jobs], dtype=np.int64)
    print(
        f"{len(jobs)} tasks, {n_rows.sum():,} rows, skipped {skipped}", flush=True
    )

    t0 = time.time()
    sums = np.zeros((len(jobs), d_feat))
    sqs = np.zeros((len(jobs), d_feat))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (s, ss) in enumerate(ex.map(task_moments, jobs, chunksize=4)):
            sums[i], sqs[i] = s, ss
            if i % 500 == 0:
                print(f"  {i}/{len(jobs)} ({(time.time() - t0) / 60:.1f} min)", flush=True)

    task_mean = sums / n_rows[:, None]
    task_var = np.maximum(sqs / n_rows[:, None] - task_mean**2, 0.0)

    # The training marginal, not the row-count marginal: a step draws the task
    # first, so a 500-row task and a 8192-row task contribute in proportion to
    # their pretraining weight, not their dump size.
    p = np.array(weights) / np.sum(weights)
    mean = p @ task_mean
    var = p @ (task_var + task_mean**2) - mean**2
    std = np.sqrt(np.maximum(var, 0.0))

    # A dimension with no variance under the training marginal carries no
    # signal, and dividing by its sigma would turn bf16 rounding into a unit
    # normal. Leave it alone; TabPFN's RemoveConstantFeaturesStep drops it.
    floor = 1e-4
    dead = std <= floor
    scale = np.where(dead, 1.0, std)
    print(
        f"mean |mu| max {np.abs(mean).max():.4f} at dim {int(np.abs(mean).argmax())}  "
        f"sigma min {std.min():.5f} p50 {np.median(std):.4f} max {std.max():.4f}  "
        f"spread {std.max() / std[~dead].min():.1f}x  dead {int(dead.sum())}",
        flush=True,
    )
    top = np.argsort(-np.abs(mean))[:8]
    for j in top:
        print(f"  dim {j:>3} mu {mean[j]:>9.4f} sigma {std[j]:.5f}", flush=True)

    o = Path(out_npz).expanduser()
    o.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        o,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        scale=scale.astype(np.float32),
        dead=dead,
        task_keys=np.array(keys),
        task_weight=np.array(weights),
        task_rows=n_rows,
        task_mean=task_mean.astype(np.float32),
        task_std=np.sqrt(task_var).astype(np.float32),
        features_root=str(root),
        min_rows=min_rows,
        floor=floor,
    )
    print(f"wrote {o} in {(time.time() - t0) / 60:.1f} min", flush=True)
