import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from relbench.submit import evaluate_submission  # noqa: E402
from relbench.submit import main as package  # noqa: E402

from expts.fine_tune.run import stage_dir  # noqa: E402
from expts.fine_tune.submit import (  # noqa: E402
    ENTITY,
    MODELS,
    OUT_ROOT,
    PROJECT,
    TASKS,
    paper,
    stds,
)

LEADERBOARD = Path(OUT_ROOT).expanduser() / "leaderboard" / PROJECT


def gather(model: str) -> tuple[Path, list[str]]:
    preds = LEADERBOARD / model
    preds.mkdir(parents=True, exist_ok=True)
    missing = []
    for db, task in TASKS:
        src = (
            stage_dir(OUT_ROOT, ENTITY, PROJECT, f"{model}-{db}-{task}-test")
            / "eval_out"
            / f"{db}__{task}.csv"
        )
        if src.exists():
            shutil.copyfile(src, preds / src.name)
        else:
            missing.append(f"{db}/{task}")
    return preds, missing


def compare(model: str, result: dict) -> None:
    ref = paper()
    print(f"\n{model}: ours vs the RelArena-alpha paper's rt-plurel (test)")
    print(f"  {'task':28s} {'metric':8s} {'paper':>8s} {'ours':>8s}")
    for db, task in TASKS:
        key = f"{db}/{task}"
        entry = result["tasks"].get(key)
        row = ref[key]
        if entry is None or entry["metric"] is None:
            ours = "-"
        elif entry["metric_name"] == "roc_auc":
            ours = f"{entry['metric']:.4f}"
        else:
            ours = f"{entry['metric'] * stds()[key]:.4g}"
        theirs = row["rt-plurel"]
        print(f"  {key:28s} {row['metric']:8s} {theirs:>8s} {ours:>8s}")


def main() -> None:
    for model, _ in MODELS:
        preds, missing = gather(model)
        print(
            f"== {model}: {len(TASKS) - len(missing)}/{len(TASKS)} prediction tables in {preds}"
        )
        for m in missing:
            print(f"   missing {m}")
        if len(missing) == len(TASKS):
            continue
        result = evaluate_submission(preds, verbose=False)
        compare(model, result)
        if result["validated"]:
            package([str(preds), "--out", str(LEADERBOARD / f"{model}.zip")])


if __name__ == "__main__":
    main()
