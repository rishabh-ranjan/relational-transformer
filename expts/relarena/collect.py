"""Collect the sweep's result frames into one table. Run it anywhere.

    pixi run python expts/relarena/collect.py

Each job writes `<out_dir>/<run_id>.csv` -- one relarena result frame, already
in the shared schema. This concatenates them, keeps the row each experiment
reports (the tuned one, or the default when a model is parameter-free), and
prints it beside the published baselines from expts/fine_tune/results.csv so a
number can be read against what else scores on that task.

Not part of any job: a job writes its own frame and nothing else, so this reads
the directory afterwards and is safe to run while the sweep is still going.
"""

from pathlib import Path

SHARE = Path("/dfs/user/ranjanr/share/relarena")
BASELINES = Path(__file__).parent.parent / "fine_tune" / "results.csv"


def main() -> None:
    import pandas as pd

    frames = []
    for f in sorted((SHARE / "results").glob("*.csv")):
        if f.name.startswith("zero-shot"):
            continue  # not protocol runs; see zero_shot.py
        try:
            frames.append(pd.read_csv(f))
        except Exception as exc:  # a job still writing, or one that died mid-write
            print(f"  ! skipping {f.name}: {exc}")
    if not frames:
        print("no protocol results yet")
        return
    d = pd.concat(frames, ignore_index=True)

    # The reported row per experiment: relarena marks it `selected`.
    rep = d[d.get("selected", False) == True] if "selected" in d else d  # noqa: E712
    if rep.empty:
        rep = d
    cols = [c for c in ("dataset", "task", "task_type", "metric", "val_score",
                        "test_score", "fit_time_tuning", "fit_time_refit",
                        "predict_time_refit") if c in rep]
    rep = rep[cols].sort_values(["dataset", "task"])
    print(f"\n=== rt: {len(rep)} of 21 tasks ===")
    print(rep.to_string(index=False))

    if BASELINES.exists():
        base = pd.read_csv(BASELINES)
        base = base[base.get("selected", False) == True]  # noqa: E712
        piv = base.pivot_table(
            index=["dataset", "task"], columns="model", values="test_score"
        )
        merged = rep.set_index(["dataset", "task"]).join(piv, how="left")
        print("\n=== rt test_score against the published baselines ===")
        print(merged.to_string())


if __name__ == "__main__":
    main()
