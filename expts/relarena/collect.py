from pathlib import Path

SHARE = Path("~/scratch/share/relarena").expanduser()
BASELINES = Path(__file__).parent.parent / "fine_tune" / "results.csv"


def main() -> None:
    import pandas as pd

    frames = []
    for f in sorted((SHARE / "results").glob("*.csv")):
        if f.name.startswith("zero-shot"):
            continue
        try:
            frames.append(pd.read_csv(f))
        except Exception as exc:
            print(f"  ! skipping {f.name}: {exc}")
    if not frames:
        print("no protocol results yet")
        return
    d = pd.concat(frames, ignore_index=True)

    rep = d[d.get("selected", False) == True] if "selected" in d else d  # noqa: E712
    if rep.empty:
        rep = d
    rep = rep.drop(columns=[c for c in rep if c.startswith("val_")], errors="ignore")
    cols = [
        c
        for c in (
            "dataset",
            "task",
            "task_type",
            "metric",
            "test_score",
            "fit_time_tuning",
            "fit_time_refit",
            "predict_time_refit",
        )
        if c in rep
    ]
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
