import ast
import json
from pathlib import Path

from expts.repaper.config import PRE_DIR


def grid(db: str, table: str) -> Path:
    return (
        Path("~/scratch/relational-transformer/icl/rtv2/2026-08-25-icl").expanduser()
        / f"tune-rt-j-{db}-{table}"
        / "tuning.json"
    )
    # from expts.repaper.config import CKPT_ROOT, project
    # return (
    #     Path(CKPT_ROOT).expanduser()
    #     / "rtv2"
    #     / project("tune")
    #     / f"tune--{db}--{table}"
    #     / "tuning.json"
    # )


def main() -> None:
    tasks = [
        tuple(p)
        for p in json.loads(
            (Path(PRE_DIR).expanduser() / "db-task-lists" / "forecast.json").read_text()
        )
    ]
    out = {}
    for db, table in tasks:
        path = grid(db, table)
        assert path.exists(), f"missing {path}; the grid is not finished"
        rec = json.loads(path.read_text())[f"{db}/{table}"]
        scores = rec["val_scores"]
        assert len(scores) == 120, (
            f"{db}/{table}: {len(scores)} configs scored, expected 120"
        )
        assert rec["val_ensemble_size"] == 4, rec["val_ensemble_size"]
        reverse = rec["task_type"] == "clf"
        top = sorted(
            scores.items(), key=lambda kv: (-kv[1] if reverse else kv[1], kv[0])
        )[:4]
        assert list(ast.literal_eval(top[0][0])) == list(rec["best_cfg"]), (
            f"{db}/{table}: best_cfg {rec['best_cfg']} is not the top score {top[0]}"
        )
        out[f"{db}/{table}"] = {
            "task_type": rec["task_type"],
            "val_ensemble_size": rec["val_ensemble_size"],
            "best_cfg": rec["best_cfg"],
            "best_value": rec["best_value"],
            "top_cfgs": [list(ast.literal_eval(c)) for c, _ in top],
            "top_values": [v for _, v in top],
            "val_scores": scores,
            "grid": str(path),
        }
    dest = Path(__file__).with_name("tuned_configs.json")
    dest.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"wrote {dest} ({len(out)} tasks)")
    for k, v in sorted(out.items()):
        print(f"  {k}: best={v['best_cfg']} ({v['best_value']:.4f})")


if __name__ == "__main__":
    main()
