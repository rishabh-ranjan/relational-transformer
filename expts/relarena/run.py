from pathlib import Path


def main(
    *,
    dataset: str,
    task: str,
    model: str,
    seed: int,
    n_trials: int,
    cache_dir: str,
    out_dir: str,
    run_id: str,
) -> None:
    import logging

    import pandas as pd

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    import relarena.models  # noqa: F401  -- registers the built-in models
    from relarena.registry import registry
    from relarena.results import summary_to_dataframe
    from relarena.runner import run_experiment

    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = str(Path(cache_dir).expanduser())
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    if model.startswith("rt"):
        from relarena.models.rt.warm_cache import precompute_dataset_task

        print(f"+ warming rt tensor cache for {dataset}/{task}", flush=True)
        precompute_dataset_task(dataset, task, cache_dir=cache_dir)

    print(f"+ running {model} on {dataset}/{task} (seed {seed})", flush=True)
    summary = run_experiment(
        registry.get(model),
        dataset,
        task,
        seed=seed,
        n_trials=n_trials,
        cache_dir=cache_dir,
    )

    frame = summary_to_dataframe(summary)
    path = out / f"{run_id}.csv"
    frame.to_csv(path, index=False)

    best = summary.tuned or summary.default
    print(f"+ wrote {path}", flush=True)
    if best is None or best.test_score is None:
        raise RuntimeError(
            f"{model} on {dataset}/{task} produced no test score; see the refit "
            "traceback above. The result frame was still written to "
            f"{path} for inspection."
        )
    if best is not None:
        print(
            f"+ {summary.metric_name}: val={best.val_score} test={best.test_score}",
            flush=True,
        )
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(frame, flush=True)
