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


def main(
    *,
    dataset: str,
    task: str,
    cache_dir: str,
    out_dir: str,
    run_id: str,
) -> None:
    """Score the warm start on `dataset/task`'s test split; write a CSV."""
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
    outer = source.outer_split()
    train_union = concat_tables(outer.train_table, outer.val_table)

    # `fill`, not `raise`: this entry point warms what it needs itself.
    cache = resolve_cache_config(cache_dir, on_miss="fill")
    model = RTModel({}, cache=cache, run_identity=source.run_identity("outer"))
    # Stand in for `fit`. Every one of these is what the reporting arm would
    # have set; the only difference is the checkpoint, which is the published
    # one rather than one this run produced.
    model._phase = cfg.PHASE_OUTER
    model._task_type = source.task.task_type
    model._train_table = train_union
    model._target_stats = target_stats(train_union, source.task)
    model._checkpoint = cfg.warm_start(source.task.task_type)

    print(f"+ zero-shot {model._checkpoint} on {dataset}/{task}", flush=True)
    pred = model.predict(source.task, outer.db_state, outer.eval_table)

    metric = primary_metric(source.task)
    metrics = list(source.task.metrics)
    if metric.__name__ not in {m.__name__ for m in metrics}:
        metrics.append(metric)
    # `target_table=None`: RelBench loads the held-out test labels itself.
    scores = source.task.evaluate(pred, None, metrics=metrics)

    frame = pd.DataFrame([{"dataset": dataset, "task": task, **scores}])
    path = out / f"zero-shot-{run_id}.csv"
    frame.to_csv(path, index=False)
    print(f"+ wrote {path}", flush=True)
    print(frame.to_string(index=False), flush=True)
