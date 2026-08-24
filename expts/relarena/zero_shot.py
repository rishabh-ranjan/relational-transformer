from pathlib import Path


def _patch_cutoff(cutoff_offset: int) -> None:
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
    import pandas as pd
    from relarena.cache import resolve_cache_config
    from relarena.dataset import RelBenchDatasetTask, concat_tables
    from relarena.metrics import primary_metric
    from relarena.models.rt import config as cfg
    from relarena.models.rt.export import target_stats
    from relarena.models.rt.model import RTModel

    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = str(Path(cache_dir).expanduser())
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    source = RelBenchDatasetTask(dataset, task, download=True)
    if split == "test":
        chosen = source.outer_split()
        train_table = concat_tables(chosen.train_table, chosen.val_table)
        eval_table, target_table = chosen.eval_table, None
    elif split == "val":
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

    cache = resolve_cache_config(cache_dir, on_miss="fill")
    phase = "outer" if split == "test" else "inner"
    model = RTModel({}, cache=cache, run_identity=source.run_identity(phase))
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
