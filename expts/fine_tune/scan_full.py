"""Full-attention arm against its baseline, per task, from wandb.

    pixi run python expts/fine_tune/scan_full.py

Two rows per task: the `skip_full_attn=False` run, and the latest run of the
same task from before the flag existed -- whose net is three attentions per
block, which is what `skip_full_attn=True` builds, so it is the comparison
even though its config has no such key.

Selection is by validation: the step where the val metric is best, and the test
number at that same step. Never the best test number, which is a number no
procedure could have picked.

The config diff is printed under each table. Anything beyond `skip_full_attn`
there is a confound in the comparison, not a footnote -- read it before reading
the numbers.
"""

import wandb

# Keys that cannot match across two runs of the same task and say nothing about
# what the comparison is measuring: the run's identity and the slot it landed on.
IGNORED = {
    "run_id",
    "run_name",
    "num_workers",
    "tokens_per_gpu",
}


def confounds_of(run, ref) -> set[str]:
    """Config keys that differ between two runs other than `skip_full_attn`.

    Every one of them is a second thing the two runs do not share, which is the
    difference between a comparison and a pair of numbers.
    """
    return {
        k
        for k in set(run.config) | set(ref.config)
        if k not in IGNORED
        and k != "skip_full_attn"
        and run.config.get(k) != ref.config.get(k)
    }


def best_by_val(run, val_key, test_key, higher_is_better):
    """`(step, val, test)` at the step where the val curve is best.

    Reads the whole history rather than the summary: the summary holds the last
    step's value, and the run is not stopped at its best one.
    """
    rows = [
        r
        for r in run.scan_history(keys=["step", val_key, test_key])
        if r.get(val_key) is not None
    ]
    if not rows:
        return None
    pick = max if higher_is_better else min
    row = pick(rows, key=lambda r: r[val_key])
    return int(row["step"]), row[val_key], row[test_key]


def main() -> None:
    api = wandb.Api()
    runs = list(api.runs("rtv2/2026-08-07-fine_tune"))

    # A requeued attempt is its own wandb run sharing the group; the newest
    # attempt of a group is the one that ran to the end.
    full, base = {}, {}
    for run in runs:
        db_task = run.config.get("db_task_list")
        if not db_task or len(db_task) != 1:
            continue
        # A cancelled attempt stopped wherever it stopped; its curve is a
        # prefix of a run that was never asked to finish.
        if run.state not in ("finished", "running"):
            continue
        task = tuple(db_task[0])
        # The flag is absent on every run that predates it, and absence means
        # the three-attention net -- exactly what True builds.
        if run.config.get("skip_full_attn") is False:
            if task not in full or run.created_at > full[task].created_at:
                full[task] = run
        else:
            base.setdefault(task, []).append(run)

    for task in sorted(full):
        db, name = task
        # The comparison run is the *comparable* one, not simply the newest:
        # the project also holds arms that vary the sampler, and the newest run
        # of a task is as likely to be one of those. Fewest confounds first,
        # newest among those.
        candidates = base.get(task, [])
        if not candidates:
            print(f"{db}/{name}: no three-attention run to compare against\n")
            continue
        fewest = min(len(confounds_of(full[task], r)) for r in candidates)
        ref = max(
            (r for r in candidates if len(confounds_of(full[task], r)) == fewest),
            key=lambda r: r.created_at,
        )

        metric_and_dir = [
            (m, h)
            for m, h in (("auroc", True), ("nmae", False))
            if f"{m}/val/{db}/{name}" in full[task].summary
        ]
        if not metric_and_dir:
            print(f"{db}/{name}: the full-attention run has logged no val metric yet\n")
            continue
        metric, higher = metric_and_dir[0]
        val_key, test_key = f"{metric}/val/{db}/{name}", f"{metric}/test/{db}/{name}"

        print(f"{db}/{name}  ({metric}, best val step)")
        print(f"  {'arm':22s} {'step':>7s} {'val':>8s} {'test':>8s}")
        for label, run in (
            ("skip_full_attn=False", full[task]),
            ("skip_full_attn=True", ref),
        ):
            got = best_by_val(run, val_key, test_key, higher)
            if got is None:
                print(f"  {label:22s} {'-- no history --':>25s}")
                continue
            step, v, t = got
            print(f"  {label:22s} {step:7d} {v:8.2f} {t:8.2f}")

        print(f"  compared against: {ref.name}  ({ref.created_at})")
        confounds = sorted(confounds_of(full[task], ref))
        print(
            "  clean comparison" if not confounds else f"  CONFOUNDED by: {confounds}"
        )
        print()


if __name__ == "__main__":
    main()
