import json
from pathlib import Path

import numpy as np

from expts.repaper.config import CKPT_CLF, CKPT_REG, PRE_DIR, SHARE, project

CSV_DIR = Path(SHARE).expanduser() / "leaderboard" / "preds"
N_CFGS = 4
N_SEEDS = 4


def units(db: str, table: str, rank: int) -> list[tuple[Path, int]]:
    icl = Path("~/scratch/relational-transformer/icl/rtv2/2026-08-25-icl").expanduser()
    whole = icl / f"ens-rt-j-{db}-{table}-cfg{rank}"
    if (whole / "result.json").exists():
        return [(whole, N_SEEDS)]
    return [(icl / f"ens-rt-j-{db}-{table}-cfg{rank}-s{k}", 1) for k in range(N_SEEDS)]
    # from expts.repaper.config import OUT_ROOT
    # unit = Path(OUT_ROOT).expanduser() / "repaper-submit" / f"cfg{rank}"
    # return [(unit / f"{db}__{table}", N_SEEDS)]


def load_unit(unit: Path, seeds: int, cfg: list, ckpt: str) -> np.lib.npyio.NpzFile:
    if unit.is_dir():
        result_path, state_path = unit / "result.json", unit / "state.npz"
    else:
        result_path = unit.with_suffix(".json")
        state_path = unit.with_suffix(".state.npz")
    assert result_path.exists(), f"{unit}: not finished"
    config = json.loads(result_path.read_text())["config"]
    got = [
        config["ctx_size"],
        config["local_ctx_size"],
        config["bfs_width"],
        config["prefer_latest"],
    ]
    assert got == list(cfg), f"{unit}: config {got} != tuned {cfg}"
    assert config["n_seeds"] == seeds, f"{unit}: {config['n_seeds']} seeds != {seeds}"
    assert config.get("checkpoint", ckpt) == ckpt, f"{unit}: checkpoint {config}"
    assert config["shuffle_seed"] == 0 and config["context_seed"] == 0
    assert config["db_cutoff"] is None
    st = np.load(state_path)
    assert int(st["seeds"]) == seeds, f"{unit}: {int(st['seeds'])}/{seeds} seeds done"
    return st


def main() -> None:
    from rt.data import get_tasks
    from rt.eval.relbench import _emit_and_score

    cfgs = json.loads(
        (Path(__file__).parents[1] / "tune" / "tuned_configs.json").read_text()
    )
    assert len(cfgs) == 21, f"{len(cfgs)} tuned configs, expected 21"
    results = {}
    for task_key, rec in sorted(cfgs.items()):
        db, table = task_key.split("/")
        (task,) = get_tasks(PRE_DIR, [(db, table)], ("test",))
        ckpt = {"clf": CKPT_CLF, "reg": CKPT_REG}[task.task_type]
        total = labels = nodes = None
        for rank in range(N_CFGS):
            for unit, seeds in units(db, table, rank):
                st = load_unit(unit, seeds, rec["top_cfgs"][rank], ckpt)
                if total is None:
                    total = st["sum_preds"].astype(np.float64)
                    labels, nodes = st["labels"], st["node_idxs"]
                else:
                    assert np.array_equal(nodes, st["node_idxs"])
                    assert np.array_equal(labels, st["labels"])
                    total = total + st["sum_preds"].astype(np.float64)
        mean_pred = total / (N_CFGS * N_SEEDS)

        mname, mval, n, align, csv = _emit_and_score(
            CSV_DIR, task, PRE_DIR, "all-MiniLM-L12-v2", labels, mean_pred, nodes
        )
        results[task_key] = {
            "task_type": task.task_type,
            "metric": mname,
            "value": mval,
            "n": n,
            "align": align,
            "top_cfgs": rec["top_cfgs"],
            "top_values": rec["top_values"],
            "units": [str(u) for r in range(N_CFGS) for u, _ in units(db, table, r)],
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
