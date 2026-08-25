import ast
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from relbench.submit import evaluate_submission  # noqa: E402
from relbench.submit import main as package  # noqa: E402

from expts.icl.submit import OUT_ROOT, PRE_DIR, PROJECT, TASKS, stage_dir  # noqa: E402

HERE = Path(__file__).parent
LEADERBOARD = Path(OUT_ROOT).expanduser() / "leaderboard" / PROJECT
N_CFGS = 120
N_TOP = 4
N_SEEDS = 4


def collect_tuning() -> dict:
    out = {}
    for db, task in TASKS:
        path = stage_dir(f"tune-{db}-{task}") / "tuning.json"
        if not path.exists():
            print(f"  {db}/{task:20s} tuning not finished")
            continue
        rec = json.loads(path.read_text())[f"{db}/{task}"]
        scores = rec["val_scores"]
        assert len(scores) == N_CFGS, (
            f"{db}/{task}: {len(scores)} configs scored, expected {N_CFGS}"
        )
        reverse = rec["task_type"] == "clf"
        ranked = sorted(
            scores.items(), key=lambda kv: (-kv[1] if reverse else kv[1], kv[0])
        )
        out[f"{db}/{task}"] = {
            "task_type": rec["task_type"],
            "val_ensemble_size": rec["val_ensemble_size"],
            "best_cfg": rec["best_cfg"],
            "best_value": rec["best_value"],
            "top_cfgs": [list(ast.literal_eval(c)) for c, _ in ranked[:N_TOP]],
            "top_values": [v for _, v in ranked[:N_TOP]],
            "val_scores": scores,
        }
        print(
            f"  {db}/{task:20s} "
            + "  ".join(f"{c.replace(' ', '')} {v:.4f}" for c, v in ranked[:N_TOP])
        )
    if not out:
        return out
    dest = HERE / "tuned_configs.json"
    dest.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"wrote {dest} ({len(out)}/{len(TASKS)} tasks); commit it before submitting")
    return out


def reduce(tuned: dict) -> None:
    from rt.data import get_tasks
    from rt.eval.relbench import _emit_and_score

    preds = LEADERBOARD / "preds"
    results = {}
    missing = []
    for db, task in TASKS:
        key = f"{db}/{task}"
        units = [stage_dir(f"ens-{db}-{task}-cfg{rank}") for rank in range(N_TOP)]
        if key not in tuned or not all((u / "result.json").exists() for u in units):
            missing.append(key)
            continue
        total = labels = nodes = None
        curves = []
        for unit in units:
            st = np.load(unit / "state.npz")
            assert int(st["seeds"]) == N_SEEDS, (
                f"{unit}: {int(st['seeds'])}/{N_SEEDS} seeds"
            )
            if total is None:
                total = st["sum_preds"].astype(np.float64)
                labels, nodes = st["labels"], st["node_idxs"]
            else:
                assert np.array_equal(nodes, st["node_idxs"])
                assert np.array_equal(labels, st["labels"])
                total = total + st["sum_preds"].astype(np.float64)
            curves.append(json.loads((unit / "result.json").read_text())["curve"])
        (rt_task,) = get_tasks(PRE_DIR, [(db, task)], ("test",))
        mname, mval, n, align, _ = _emit_and_score(
            preds,
            rt_task,
            PRE_DIR,
            "all-MiniLM-L12-v2",
            labels,
            total / (N_TOP * N_SEEDS),
            nodes,
        )
        results[key] = {
            "task_type": rt_task.task_type,
            "metric": mname,
            "value": mval,
            "n": n,
            "align": align,
            "top_cfgs": tuned[key]["top_cfgs"],
            "top_values": tuned[key]["top_values"],
            "per_cfg_test_curve": curves,
        }
        print(f"  {key:28s} {mname}={mval:.4f} n={n} {align}", flush=True)
    for key in missing:
        print(f"  {key:28s} not all {N_TOP} units done")
    if not results:
        return
    (LEADERBOARD / "results.json").write_text(
        json.dumps(results, indent=1, sort_keys=True) + "\n"
    )
    result = evaluate_submission(preds, verbose=True)
    compare(result)
    if result["validated"]:
        package([str(preds), "--out", str(LEADERBOARD / "rt-plurel-icl.zip")])


def compare(result: dict) -> None:
    with open(HERE / "reference.csv", newline="") as f:
        ref = {row["task"]: row for row in csv.DictReader(f)}
    print(
        "\nours vs RT-J in-context (paper, top-4 x 4) and RT-PluRel fine-tuned (RelArena)"
    )
    print(
        f"  {'task':28s} {'metric':8s} {'rt-j-icl':>9s} {'rt-plurel-ft':>13s} {'ours':>8s}"
    )
    for db, task in TASKS:
        key = f"{db}/{task}"
        entry = result["tasks"].get(key)
        ours = (
            "-"
            if entry is None or entry["metric"] is None
            else f"{entry['metric'] * 100:.2f}"
        )
        row = ref[key]
        print(
            f"  {key:28s} {row['metric']:8s} {row['rt-j-icl']:>9s} "
            f"{row['rt-plurel-ft']:>13s} {ours:>8s}"
        )


def main() -> None:
    print("== tuning")
    tuned = collect_tuning()
    print("== ensemble")
    reduce(tuned)


if __name__ == "__main__":
    main()
