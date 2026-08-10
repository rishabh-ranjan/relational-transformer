"""Stochastic-sampler arm against its baseline, per task, from wandb.

    pixi run python expts/fine_tune/scan_stoc.py

Two rows per task: the run whose contexts are drawn at random
(`num_walks=10_000`, a grid of local context sizes and BFS widths, and token
masking), and the run of the same task built from the one fixed context
(`num_walks=0`). Both sides are random init -- the RT-P arm samples
stochastically too, and warm-starting is a different comparison.

Selection is by validation: the step where the val metric is best, and the test
number at that same step. Never the best test number, which is a number no
procedure could have picked. Both curves are cut to the shorter run's
`total_steps` first, so a run that was allowed to train longer cannot win on
having had more steps to pick a best from.

Comparability is *effective*, not literal: a config key that cannot change the
numbers being read is not a confound. What remains is printed under each table,
and anything there makes the two rows a pair of numbers rather than a
comparison.
"""

import functools

import wandb

# The sampler itself: the whole set is the difference under test, so a
# difference in any of them is not a confound.
UNDER_TEST = {
    "num_walks",
    "walk_length",
    "local_ctx_size_list",
    "bfs_width_list",
    "prefer_latest_list",
    "mask_prob_max",
    "eval_lcs_bw_pl_grid",
    "eval_num_walks",
    "eval_walk_length",
}

# Keys that cannot match across two runs of the same task and say nothing about
# what the comparison is measuring: the run's identity and the slot it landed on.
IGNORED = {
    "run_id",
    "run_name",
    "num_workers",
    "tokens_per_gpu",
    # Both curves are cut to the shorter run's horizon below, so the horizon
    # itself is not a difference between what the two rows report.
    "total_steps",
    # It reaches only the `swa/` table, which is read knowing that; the live
    # table cannot see it at all.
    "swa_momentum",
}


# A key a run predates is not a run that chose differently: it ran whatever the
# code did before the key existed. Absence is read as that value.
BEFORE_THE_KEY = {"loss_fn": "huber"}


@functools.cache
def split_rows() -> dict[str, tuple[int, int]]:
    """`(num_rows_val, num_rows_test)` per `{db}/{task}`, from RelBench's stats.

    What decides whether an `eval_items_per_task` difference is real: a cap
    above the split reads the whole split, so two runs capping differently
    above it evaluate on exactly the same rows.
    """
    import pandas as pd
    from huggingface_hub import hf_hub_download

    stats = pd.read_parquet(
        hf_hub_download(
            "stanford-star/relbench", "STATS/tasks.parquet", repo_type="dataset"
        )
    )
    return {
        f"{r.database}/{r.task}": (int(r.num_rows_val), int(r.num_rows_test))
        for r in stats.itertuples()
    }


def confounds_of(run, ref, task) -> set[str]:
    """Config keys that differ between two runs and could move the numbers.

    `UNDER_TEST` is the sampler, the difference under test. `IGNORED` is the set that
    cannot reach these numbers at all, and `BEFORE_THE_KEY` fills in what a run
    older than a key actually ran. `eval_items_per_task` is the one key
    that depends on the task: it caps how much of a split is evaluated, so two
    different caps that both sit above the split are the same eval.
    """

    def value(cfg, k):
        return cfg.get(k, BEFORE_THE_KEY.get(k))

    diff = {
        k
        for k in set(run.config) | set(ref.config)
        if k not in IGNORED
        and k not in UNDER_TEST
        and value(run.config, k) != value(ref.config, k)
    }
    if "eval_items_per_task" in diff:
        caps = (run.config["eval_items_per_task"], ref.config["eval_items_per_task"])
        if min(caps) >= max(split_rows()["/".join(task)]):
            diff.discard("eval_items_per_task")
    return diff


def best_by_val(run, val_key, test_key, higher_is_better, max_step):
    """`(step, val, test)` at the step where the val curve is best, up to
    `max_step`.

    Reads the whole history rather than the summary: the summary holds the last
    step's value, and the run is not stopped at its best one.
    """
    rows = [
        r
        for r in run.scan_history(keys=["step", val_key, test_key])
        if r.get(val_key) is not None and r["step"] <= max_step
    ]
    if not rows:
        return None
    pick = max if higher_is_better else min
    row = pick(rows, key=lambda r: r[val_key])
    return int(row["step"]), row[val_key], row[test_key]


def main() -> None:
    api = wandb.Api()
    runs = list(api.runs("rtv2/2026-08-07-fine_tune"))

    stoc, base = {}, {}
    for run in runs:
        db_task = run.config.get("db_task_list")
        if not db_task or len(db_task) != 1:
            continue
        # A cancelled attempt stopped wherever it stopped; its curve is a
        # prefix of a run that was never asked to finish.
        if run.state not in ("finished", "running"):
            continue
        # The RT-P arm samples stochastically as well: it would land on the
        # stoc side and turn this into a comparison of two differences.
        if run.config.get("load_ckpt_path") is not None:
            continue
        task = tuple(db_task[0])
        if run.config.get("num_walks"):
            if task not in stoc or run.created_at > stoc[task].created_at:
                stoc[task] = run
        else:
            base.setdefault(task, []).append(run)

    for task in sorted(stoc):
        db, name = task
        # The comparison run is the *comparable* one, not simply the newest:
        # the project also holds arms that vary the sampler or the init, and
        # the newest run of a task is as likely to be one of those. Fewest
        # confounds first, then the longest horizon (the comparison is cut to
        # it), newest to break a tie.
        candidates = base.get(task, [])
        if not candidates:
            print(f"{db}/{name}: no fixed-context run to compare against\n")
            continue
        fewest = min(len(confounds_of(stoc[task], r, task)) for r in candidates)
        ref = max(
            (r for r in candidates if len(confounds_of(stoc[task], r, task)) == fewest),
            key=lambda r: (r.config["total_steps"], r.created_at),
        )

        metric_and_dir = [
            (m, h)
            for m, h in (("auroc", True), ("nmae", False))
            if f"{m}/val/{db}/{name}" in stoc[task].summary
        ]
        if not metric_and_dir:
            print(f"{db}/{name}: the stochastic run has logged no val metric yet\n")
            continue
        metric, higher = metric_and_dir[0]
        horizon = min(stoc[task].config["total_steps"], ref.config["total_steps"])

        print(f"{db}/{name}  ({metric}, best val step, first {horizon} steps)")
        # The live curve and its SWA twin are separate selections: the SWA
        # weights peak at a different step, so each table picks its own.
        for kind, prefix in (("live", ""), ("swa", "swa/")):
            val_key = f"{prefix}{metric}/val/{db}/{name}"
            test_key = f"{prefix}{metric}/test/{db}/{name}"
            print(f"  {kind:22s} {'step':>7s} {'val':>8s} {'test':>8s}")
            for label, run in (
                ("num_walks=10_000", stoc[task]),
                ("num_walks=0", ref),
            ):
                got = best_by_val(run, val_key, test_key, higher, horizon)
                if got is None:
                    print(f"    {label:20s} {'-- no history --':>25s}")
                    continue
                step, v, t = got
                print(f"    {label:20s} {step:7d} {v:8.2f} {t:8.2f}")

        print(f"  compared against: {ref.name}  ({ref.created_at})")
        confounds = sorted(confounds_of(stoc[task], ref, task))
        print(
            "  effectively differs only in the sampler"
            if not confounds
            else f"  CONFOUNDED by: {confounds}"
        )
        print()


if __name__ == "__main__":
    main()
