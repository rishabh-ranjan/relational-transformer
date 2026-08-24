from __future__ import annotations

import time


def main(dataset: str, task: str, cache_dir: str, seeds: int = 4) -> None:
    import numpy as np
    import torch

    from relarena.cache import resolve_cache_config
    from relarena.dataset import RelBenchDatasetTask, concat_tables
    from relarena.models.rt import config as cfg
    from relarena.models.rt.export import preprocessed_dir
    from relarena.models.rt.model import DB_NAME, _seed_offset
    from relarena.models.rt.warm_cache import precompute_dataset_task

    print(f"+ warming {dataset}/{task}", flush=True)
    precompute_dataset_task(dataset, task, cache_dir=cache_dir)

    source = RelBenchDatasetTask(dataset, task, download=True)
    outer = source.outer_split()
    union = concat_tables(outer.train_table, outer.val_table)
    cache = resolve_cache_config(cache_dir, on_miss="fill")
    pre_dir = preprocessed_dir(
        outer.db_state,
        source.task,
        {"train": union, "test": outer.eval_table},
        cache=cache,
        identity=source.run_identity("outer"),
        db_name=DB_NAME,
    )

    from rt import RelationalTransformer
    from rt.data import get_tasks
    from rt.eval import build_evaluator

    from relarena.models.rt.export import TASK_DIR

    table = outer.eval_table
    _offset = _seed_offset(pre_dir, "test")
    cutoff = cfg.context_cutoff(source.task, "test")
    context = (1024, 1024, 256, False)
    rt_tasks = get_tasks(str(pre_dir), [(DB_NAME, TASK_DIR)], ("test",))
    member_seeds = cfg.ensemble_context_seeds(seeds)

    def run(compile_: bool) -> float:
        net = RelationalTransformer.from_pretrained(
            cfg.warm_start(source.task.task_type), device="cuda", compile=compile_
        ).to(torch.bfloat16)
        args = cfg.eval_args(
            device="cuda",
            context_seed=member_seeds[0],
            num_rows=len(table.df),
            db_cutoff=cutoff,
            context=context,
        )
        ev = build_evaluator(rt_tasks, str(pre_dir), **args)
        for _ in ev.evaluate_raw(
            [(net, "")], args["ctx_size_list"], with_node_idxs=True
        ):
            pass
        torch.cuda.synchronize()

        tic = time.perf_counter()
        n = 0
        for s in member_seeds:
            args = cfg.eval_args(
                device="cuda",
                context_seed=s,
                num_rows=len(table.df),
                db_cutoff=cutoff,
                context=context,
            )
            ev = build_evaluator(rt_tasks, str(pre_dir), **args)
            for res in ev.evaluate_raw(
                [(net, "")], args["ctx_size_list"], with_node_idxs=True
            ):
                n += len(np.asarray(res[5]))
        torch.cuda.synchronize()
        dt = time.perf_counter() - tic
        print(
            f"  compile={compile_!s:5s}  {dt:7.1f}s over {seeds} seeds, {n} row-scores"
            f"  ({n / dt:.0f} rows/s)",
            flush=True,
        )
        return dt

    print(f"+ {dataset}/{task}: {len(table.df)} test rows, ctx={context}", flush=True)
    eager = run(False)
    comp = run(True)
    print(
        f"\nRESULT compile speedup: {eager / comp:.2f}x  "
        f"(eager {eager:.1f}s -> compiled {comp:.1f}s)",
        flush=True,
    )
