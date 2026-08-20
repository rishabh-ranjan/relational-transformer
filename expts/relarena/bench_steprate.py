"""Why is training ~2x the fitted s/step? Isolate the candidates.

The fitted 0.94 s/step came from one task (rel-f1/driver-position) under a
configuration that has since changed. Three things could explain the gap, and
they call for different fixes:

  H1  num_walks=10_000 is expensive, and its cost scales with the graph, so it
      is cheap on rel-f1 and dear on rel-avito. Fix: fewer walks.
  H2  the observed rate is wall-clock per step and therefore includes the
      in-loop eval, which went from ensemble 1 to ensemble 4. Then nothing is
      slower; the estimator is measuring something else. Fix: none needed.
  H3  s/step genuinely scales with database size and a constant fitted on the
      smallest dataset in the sweep was never going to hold. Fix: reproject.

Every config runs the same step count from the same warm cache, so total wall
time is directly comparable -- startup and preprocessing are common to all.
"""

from __future__ import annotations

import time


def main(dataset: str, task: str, cache_dir: str, steps: int = 300) -> None:
    from relarena.cache import resolve_cache_config
    from relarena.dataset import RelBenchDatasetTask
    from relarena.models.rt import config as cfg
    from relarena.models.rt.export import preprocessed_dir, TASK_DIR
    from relarena.models.rt.model import DB_NAME
    from relarena.models.rt.warm_cache import precompute_dataset_task

    print(f"+ warming {dataset}/{task}", flush=True)
    precompute_dataset_task(dataset, task, cache_dir=cache_dir)
    src = RelBenchDatasetTask(dataset, task, download=True)
    inner = src.inner_split()
    pre = preprocessed_dir(
        inner.db_state, src.task,
        {"train": inner.train_table, "val": inner.eval_table},
        cache=resolve_cache_config(cache_dir, on_miss="fill"),
        identity=src.run_identity("inner"), db_name=DB_NAME,
    )

    from rt.train import main as train_main

    def run(label: str, **over) -> float:
        args = cfg.train_args(
            phase=cfg.PHASE_INNER, task_type=src.task.task_type, pre_dir=str(pre),
            db_name=DB_NAME, task_name=TASK_DIR, train_split="train",
            eval_split="val", db_cutoff=cfg.context_cutoff(src.task, "val"),
            total_steps=steps, out_root=f"/tmp/ranjanr/bench-{label}",
            run_id=label, seed=0,
        )
        args["early_stop_after_steps"] = None   # never stop; every config runs `steps`
        args.update(over)
        tic = time.perf_counter()
        train_main(**args)
        dt = time.perf_counter() - tic
        print(f"RATE {label:22s} {dt:7.1f}s for {steps} steps -> {dt/steps:5.3f} s/step",
              flush=True)
        return dt

    print(f"+ {dataset}/{task}: {steps} steps per config", flush=True)
    base = run("walks10k-eval", )
    noev = run("walks10k-noeval", eval_freq=None, eval_splits=[])
    w1k = run("walks1k-noeval", eval_freq=None, eval_splits=[], num_walks=1_000)
    w0 = run("walks0-noeval", eval_freq=None, eval_splits=[], num_walks=0)
    print("\n=== attribution")
    print(f"  in-loop eval share : {(base-noev)/base*100:5.1f}%  ({base-noev:.0f}s of {base:.0f}s)")
    print(f"  walks 10k -> 1k    : {(noev-w1k)/noev*100:5.1f}% faster ({noev:.0f}s -> {w1k:.0f}s)")
    print(f"  walks 10k -> 0     : {(noev-w0)/noev*100:5.1f}% faster ({noev:.0f}s -> {w0:.0f}s)")
    print(f"  pure train s/step  : {noev/steps:.3f} (walks 10k), {w1k/steps:.3f} (1k), {w0/steps:.3f} (0)")
