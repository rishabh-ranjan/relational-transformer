"""Score a published RT checkpoint on a task's test split, with no fine-tuning.

A shortcut, and only a shortcut: it drives `RTModel.predict` -- the same export,
the same 8-seed context ensemble, the same denormalization -- against the
published warm start instead of a fine-tuned checkpoint, and scores it with
`task.evaluate`. What it skips is the selection arm, which is what makes it fast
and also what makes it *not* the `rt` protocol: a real `rt` number is whatever
validation chose, which may or may not be the warm start.

Use it to read a checkpoint's zero-shot number on a task. Do not put its output
in a results table beside protocol runs.
"""

from pathlib import Path


def _patch_cutoff(cutoff_offset: int) -> None:
    """Shift the context cutoff by `cutoff_offset` seconds.

    `context_cutoff` subtracts a second from the split's earliest timestamp
    because rustler's bound is inclusive (`past_bound` is `ts > bound`), so a
    cutoff landing exactly on the split's first cohort would leave those rows
    quotable by every later seed. `cutoff_offset=0` removes that subtraction,
    which is what this measures.
    """
    from relarena.models.rt import config as cfg

    original = cfg.context_cutoff

    def patched(eval_table):
        value = original(eval_table)
        return None if value is None else value + 1 + cutoff_offset

    cfg.context_cutoff = patched


def main(
    *,
    dataset: str,
    task: str,
    split: str,
    mask_labels: bool,
    cutoff_offset: int,
    cache_dir: str,
    out_dir: str,
    run_id: str,
) -> None:
    """Score the warm start on `dataset/task`'s `split`; write a CSV.

    `split` is "test" or "val". Scoring **val** is the diagnostic that separates
    a bad model from a broken pipeline: `rt.train`'s own in-loop evaluator
    reports a number for the same checkpoint on the same rows, so a val score
    that agrees with it says the export, the context build, the node-index join
    and the denormalization are all right, and a test score is then the model's.
    A val score near chance says the fault is ours.

    Context quoting is left entirely to `db_cutoff`, as the benchmark leaves it
    labelled rows of the split being scored. `rt.train`'s in-loop eval quotes
    them (it passes `False`), so reproducing its number needs `False` too; a
    benchmark prediction needs `True`.

    `mask_labels` drops the target column from the table handed to `predict`,
    so the export writes the same constant placeholder it writes for test. It
    is the last difference between the two splits: RelArena's `InnerSplit` sets
    `eval_table=eval_target=val_table`, so a val prediction is handed a table
    that still carries its own answers, while the test table is masked. Whether
    that matters is exactly what this flag measures.
    """
    import pandas as pd

    from relarena.cache import resolve_cache_config
    from relarena.dataset import RelBenchDatasetTask, concat_tables
    from relarena.metrics import primary_metric
    from relarena.models.rt import config as cfg
    from relarena.models.rt.export import target_stats
    from relarena.models.rt.model import RTModel

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    source = RelBenchDatasetTask(dataset, task, download=True)
    if split == "test":
        # The reporting phase: fit on train+val, score the masked test table
        # against the test-censored database.
        chosen = source.outer_split()
        train_table = concat_tables(chosen.train_table, chosen.val_table)
        eval_table, target_table = chosen.eval_table, None
    elif split == "val":
        # The selection phase: train split, val-censored database, and val
        # labels to score against -- the same rows rt.train evaluates in-loop.
        chosen = source.inner_split()
        train_table = chosen.train_table
        eval_table, target_table = chosen.eval_table, chosen.eval_target
        if mask_labels:
            from relbench.base import Table

            eval_table = Table(
                df=eval_table.df.drop(columns=[source.task.target_col]),
                fkey_col_to_pkey_table=eval_table.fkey_col_to_pkey_table,
                pkey_col=eval_table.pkey_col,
                time_col=eval_table.time_col,
            )
    else:
        raise ValueError(f"split must be 'test' or 'val'; got {split!r}")
    train_union = train_table

    # `fill`, not `raise`: this entry point warms what it needs itself.
    cache = resolve_cache_config(cache_dir, on_miss="fill")
    phase = "outer" if split == "test" else "inner"
    model = RTModel({}, cache=cache, run_identity=source.run_identity(phase))
    # Stand in for `fit`. Every one of these is what the reporting arm would
    # have set; the only difference is the checkpoint, which is the published
    # one rather than one this run produced.
    model._phase = cfg.PHASE_OUTER
    model._task_type = source.task.task_type
    model._train_table = train_union
    model._target_stats = target_stats(train_union, source.task)
    model._checkpoint = cfg.warm_start(source.task.task_type)

    print(
        f"+ zero-shot {model._checkpoint} on {dataset}/{task} {split} "
        f"(mask_labels={mask_labels}, "
        f"cutoff_offset={cutoff_offset})",
        flush=True,
    )
    _patch_cutoff(cutoff_offset)
    pred = model.predict(source.task, chosen.db_state, eval_table)

    metric = primary_metric(source.task)
    metrics = list(source.task.metrics)
    if metric.__name__ not in {m.__name__ for m in metrics}:
        metrics.append(metric)
    # `target_table=None` on test: RelBench loads the held-out labels itself.
    scores = source.task.evaluate(pred, target_table, metrics=metrics)

    frame = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "task": task,
                "split": split,
                "mask_labels": mask_labels,
                "cutoff_offset": cutoff_offset,
                **scores,
            }
        ]
    )
    path = out / f"zero-shot-{split}-{run_id}.csv"
    frame.to_csv(path, index=False)
    print(f"+ wrote {path}", flush=True)
    print(frame.to_string(index=False), flush=True)
