"""Average the 16 full-test predictions per task into the RelBench submission.

Reads the 4 per-config state files each task's units wrote (each the sum of 4
raw context-seed predictions), averages all 16 in raw output space aligned by
node index, then writes the prediction CSVs through ``rt.eval``'s own
submission writer (which denormalizes regression to the target scale and
sigmoids classification logits) and scores them with RelBench's evaluator.

Also writes ``results.json`` beside the CSVs (the paper's tuned+ensembled
per-task table) and prints the packaging commands (``python -m
relbench.submit``) that validate the directory and produce the leaderboard
zips.

    pixi run python -m expts.repaper.submit.reduce
"""

import json
from pathlib import Path

import numpy as np

from expts.repaper.config import OUT_ROOT, PRE_DIR, SHARE, project

OUT_ROOT = Path(OUT_ROOT).expanduser() / "repaper-submit"
CSV_DIR = Path(SHARE).expanduser() / "leaderboard" / "preds"
EMBEDDER = "all-MiniLM-L12-v2"
N_CFGS = 4
N_SEEDS = 4


def main() -> None:
    from rt.data import get_tasks
    from rt.eval.relbench import _emit_and_score

    cfgs = json.loads(
        (Path(__file__).parents[1] / "tune" / "tuned_configs.json").read_text()
    )
    results = {}
    for task_key, rec in sorted(cfgs.items()):
        db, table = task_key.split("/")
        total = labels = nodes = None
        for rank in range(N_CFGS):
            st = np.load(OUT_ROOT / f"cfg{rank}" / f"{db}__{table}.state.npz")
            assert int(st["seeds"]) == N_SEEDS, (
                f"{task_key} cfg{rank}: {int(st['seeds'])}/{N_SEEDS} seeds done"
            )
            if total is None:
                total = st["sum_preds"].astype(np.float64)
                labels, nodes = st["labels"], st["node_idxs"]
            else:
                assert np.array_equal(nodes, st["node_idxs"])
                assert np.array_equal(labels, st["labels"])
                total = total + st["sum_preds"].astype(np.float64)
        mean_pred = total / (N_CFGS * N_SEEDS)

        (task,) = get_tasks(PRE_DIR, [(db, table)], ("test",))
        mname, mval, n, align, csv = _emit_and_score(
            CSV_DIR, task, PRE_DIR, EMBEDDER, labels, mean_pred, nodes
        )
        results[task_key] = {
            "task_type": task.task_type,
            "metric": mname,
            "value": mval,
            "n": n,
            "align": align,
            "top_cfgs": rec["top_cfgs"],
        }
        print(f"{task_key}: {mname}={mval:.4f} n={n} {align}", flush=True)

    by_type = {"clf": [], "reg": []}
    for r in results.values():
        by_type[r["task_type"]].append(r["value"])
    summary = {
        "mean_clf": float(np.mean(by_type["clf"])),
        "mean_reg": float(np.mean(by_type["reg"])),
        "per_task": results,
    }
    out = CSV_DIR.parent / "results.json"
    out.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")

    # The paper's tuned+ensembled table fetches these flat keys.
    import wandb

    run = wandb.init(
        entity="rtv2",
        project=project("submit"),
        name="rtj-top4x4",
        reinit="finish_previous",
    )
    wandb.log(
        {
            "mean/clf": summary["mean_clf"],
            "mean/reg": summary["mean_reg"],
            **{f"value/{k}": r["value"] for k, r in results.items()},
            **{f"task_type/{k}": r["task_type"] for k, r in results.items()},
        }
    )
    run.finish()

    print(
        f"\nmean clf: {summary['mean_clf']:.4f}  mean reg: {summary['mean_reg']:.4f}"
        f"\nwrote {out}"
        f"\n\nnow validate + package for the leaderboard:"
        f"\n  pixi run python -m relbench.submit {CSV_DIR}",
        flush=True,
    )


if __name__ == "__main__":
    main()
