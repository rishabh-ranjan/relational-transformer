"""One RelArena experiment, as a roach target. See [submit.py](submit.py).

Runs `(model, dataset, task, seed)` end to end inside the job: warm RT's tensor
cache, run the harness, write the result frame. Nothing is read from the
environment -- every knob is an argument, so the same call is the same job.

The warm and the run are one job on purpose. `run_experiment` resolves its cache
with `on_miss="raise"` (a configured benchmark must not quietly re-embed a
database per trial), so the artifacts have to exist before it starts; and the
cache lives on node-local disk, so the warmer has to be on the node that will
read it.
"""

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
    """Warm the caches, run one experiment, write `<out_dir>/<run_id>.csv`."""
    import pandas as pd

    import relarena.models  # noqa: F401  -- registers the built-in models
    from relarena.registry import registry
    from relarena.results import summary_to_dataframe
    from relarena.runner import run_experiment

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # Every rt-family model, not just "rt": the name check was exact, so
    # `rt-norefit` skipped warming entirely and every one of its jobs died on
    # the first cache miss (run_experiment reads with on_miss="raise").
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
    if best is not None:
        print(
            f"+ {summary.metric_name}: val={best.val_score} test={best.test_score}",
            flush=True,
        )
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(frame, flush=True)
