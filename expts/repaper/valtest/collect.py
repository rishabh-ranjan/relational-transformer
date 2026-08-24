import json
from pathlib import Path

import numpy as np
import wandb

from expts.repaper.config import OUT_ROOT, project

DEFAULT_DIR = Path(OUT_ROOT).expanduser() / "repaper-scaling" / "subsampled" / "rt"
TUNED_DIR = Path(OUT_ROOT).expanduser() / "repaper-valtest" / "tuned"
N_TASKS = 21


def main() -> None:
    cfgs = json.loads(
        (Path(__file__).parents[1] / "tune" / "tuned_configs.json").read_text()
    )
    assert len(cfgs) == N_TASKS
    rows = []
    out = {}
    for task_key, rec in sorted(cfgs.items()):
        db, table = task_key.split("/")
        default = json.loads((DEFAULT_DIR / f"{db}__{table}.json").read_text())
        tuned = json.loads((TUNED_DIR / f"{db}__{table}.json").read_text())
        ctx, lcs, bw, pl = rec["best_cfg"]
        d_val = default["per_ctx"]["8192"]["metric_value"]
        t_val = tuned["per_ctx"][str(ctx)]["metric_value"]
        out[task_key] = {
            "task_type": rec["task_type"],
            "tuned_cfg": rec["best_cfg"],
            "default": d_val,
            "tuned": t_val,
        }
        rows.append([db, table, rec["task_type"], ctx, lcs, bw, bool(pl), d_val, t_val])
        print(
            f"{task_key}: default={d_val:.4f} tuned={t_val:.4f} "
            f"cfg=({ctx},{lcs},{bw},{'T' if pl else 'F'})",
            flush=True,
        )

    for tt in ("clf", "reg"):
        d = float(np.mean([r["default"] for r in out.values() if r["task_type"] == tt]))
        t = float(np.mean([r["tuned"] for r in out.values() if r["task_type"] == tt]))
        out[f"mean_{tt}"] = {"default": d, "tuned": t}
        print(f"mean {tt}: default={d:.4f} tuned={t:.4f}", flush=True)

    dest = Path(__file__).with_name("results.json")
    dest.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"wrote {dest}")

    run = wandb.init(
        entity="rtv2",
        project=project("valtest"),
        name="valtest",
        reinit="finish_previous",
    )
    flat = {}
    for task_key, rec in out.items():
        if task_key.startswith("mean_"):
            continue
        flat[f"default/{task_key}"] = rec["default"]
        flat[f"tuned/{task_key}"] = rec["tuned"]
        flat[f"cfg/{task_key}"] = json.dumps(rec["tuned_cfg"])
        flat[f"task_type/{task_key}"] = rec["task_type"]
    wandb.log(
        {
            "table": wandb.Table(
                columns=[
                    "db",
                    "table",
                    "task_type",
                    "ctx",
                    "lcs",
                    "bw",
                    "pl",
                    "default",
                    "tuned",
                ],
                data=rows,
            ),
            "mean/clf/default": out["mean_clf"]["default"],
            "mean/clf/tuned": out["mean_clf"]["tuned"],
            "mean/reg/default": out["mean_reg"]["default"],
            "mean/reg/tuned": out["mean_reg"]["tuned"],
            **flat,
        }
    )
    run.finish()


if __name__ == "__main__":
    main()
