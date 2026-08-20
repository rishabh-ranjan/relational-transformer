"""Collect the 21 per-task tuning.json files into tuned_configs.json.

Refuses to run on an incomplete grid. The output is committed beside this
file: it is what the enscurve tuned arm, the default-vs-tuned table, and the
leaderboard top-4 ensemble read, and regenerating it costs the whole grid.

    pixi run python -m expts.repaper_tune.collect
"""

import ast
import json
from pathlib import Path

PRE_DIR = "/dfs/user/ranjanr/share/stanford-star/relbench-preprocessed"
OUT_ROOT = "/dfs/user/ranjanr/ckpts"
PROJECT = "2026-08-19-repaper-tune"
N_TOP = 4  # configs the leaderboard ensemble keeps per task
N_CFGS = 120  # every job must have scored the whole grid


def main() -> None:
    tasks = [
        tuple(p)
        for p in json.loads(
            (Path(PRE_DIR) / "db-task-lists" / "forecast.json").read_text()
        )
    ]
    out = {}
    for db, table in tasks:
        path = (
            Path(OUT_ROOT) / "rtv2" / PROJECT / f"tune--{db}--{table}" / "tuning.json"
        )
        assert path.exists(), f"missing {path}; the grid is not finished"
        rec = json.loads(path.read_text())[f"{db}/{table}"]
        scores = rec["val_scores"]  # str((ctx, lcs, bw, pl)) -> value
        assert len(scores) == N_CFGS, (
            f"{db}/{table}: {len(scores)} configs scored, expected {N_CFGS}"
        )
        reverse = rec["task_type"] == "clf"  # higher auroc / lower mae
        ranked = sorted(
            scores.items(), key=lambda kv: (-kv[1] if reverse else kv[1], kv[0])
        )
        out[f"{db}/{table}"] = {
            "task_type": rec["task_type"],
            "val_ensemble_size": rec["val_ensemble_size"],
            "best_cfg": rec["best_cfg"],
            "best_value": rec["best_value"],
            "top_cfgs": [list(ast.literal_eval(c)) for c, _ in ranked[:N_TOP]],
            "top_values": [v for _, v in ranked[:N_TOP]],
            "val_scores": scores,
        }
    dest = Path(__file__).with_name("tuned_configs.json")
    dest.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"wrote {dest} ({len(out)} tasks)")
    for k, v in sorted(out.items()):
        print(f"  {k}: best={v['best_cfg']} ({v['best_value']:.4f})")


if __name__ == "__main__":
    main()
