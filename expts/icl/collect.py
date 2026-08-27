import ast
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from relbench.submit import evaluate_submission  # noqa: E402
from relbench.submit import main as package  # noqa: E402

from expts.icl.submit import (  # noqa: E402
    ENTITY,
    MODELS,
    OUT_ROOT,
    PRE_DIR,
    PROJECT,
    TASKS,
    ctx_sizes,
    lcs_bw_pl_grid,
    stage_dir,
    targets_for,
)

HERE = Path(__file__).parent
LEADERBOARD = Path(OUT_ROOT).expanduser() / "leaderboard" / PROJECT
METRIC = {"clf": "auroc", "reg": "nmae"}
N_CFGS = 120
N_TOP = 4
N_SEEDS = 4


def grid_order() -> list[tuple[int, int, int, bool]]:
    return [
        (ctx, lcs, bw, pl)
        for lcs, bw, pl in lcs_bw_pl_grid()
        for ctx in ctx_sizes()
        if lcs <= ctx
    ]


def better(task_type: str, a: float, b: float) -> bool:
    return a > b if task_type == "clf" else a < b


def summary_run(name: str, step: str):
    import wandb

    LEADERBOARD.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project=PROJECT,
        entity=ENTITY,
        name=name,
        config={"run_name": name},
        dir=str(LEADERBOARD),
    )
    wandb.define_metric(step)
    wandb.define_metric("*", step_metric=step)
    return run


def target_means() -> dict[str, float]:
    by_key = defaultdict(list)
    for db, task in TASKS:
        for key, v in targets_for(db, task).items():
            metric, split, _, _, *suffix = key.split("/")
            by_key["/".join([metric, split, "mean", *suffix])].append(v)
    return {k: float(np.mean(v)) for k, v in by_key.items()}


def log_tuning(model: str, tuned: dict) -> None:
    import wandb

    run = summary_run(f"{model}/tune", "tune/idx")
    best: dict[str, float] = {}
    for idx, (ctx, lcs, bw, pl) in enumerate(grid_order(), 1):
        point = {
            "tune/idx": idx,
            "tune/ctx_size": ctx,
            "tune/local_ctx_size": lcs,
            "tune/bfs_width": bw,
            "tune/prefer_latest": int(pl),
        }
        now, top = defaultdict(list), defaultdict(list)
        for key, rec in tuned.items():
            metric = METRIC[rec["task_type"]]
            v = rec["val_scores"][str((ctx, lcs, bw, pl))] * 100.0
            if key not in best or better(rec["task_type"], v, best[key]):
                best[key] = v
            point[f"tune/{metric}/val/{key}"] = v
            point[f"tune/best/{metric}/val/{key}"] = best[key]
            now[metric].append(v)
            top[metric].append(best[key])
        for metric in now:
            point[f"tune/{metric}/val/mean"] = float(np.mean(now[metric]))
            point[f"tune/best/{metric}/val/mean"] = float(np.mean(top[metric]))
        wandb.log(point)
    run.finish()


def collect_tuning(model: str) -> dict:
    out = {}
    for db, task in TASKS:
        path = stage_dir(f"tune-{model}-{db}-{task}") / "tuning.json"
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
    if len(out) == len(TASKS):
        log_tuning(model, out)
    return out


def log_ensemble(model: str, sums: dict, results: dict) -> None:
    import wandb

    from rt.eval import metric_for

    run = summary_run(f"{model}/top4x4", "ens_size")
    for k in range(1, N_TOP + 1):
        point = {"ens_size": k * N_SEEDS}
        by_metric = defaultdict(list)
        for key, (task_type, labels, cfg_sums) in sums.items():
            metric = METRIC[task_type]
            _, v = metric_for(task_type, labels, sum(cfg_sums[:k]) / (k * N_SEEDS))
            point[f"{metric}/test/{key}"] = v * 100.0
            by_metric[metric].append(v * 100.0)
        for metric, vs in by_metric.items():
            point[f"{metric}/test/mean"] = float(np.mean(vs))
        for db, task in TASKS:
            point.update({f"target/{t}": v for t, v in targets_for(db, task).items()})
        point.update({f"target/{t}": v for t, v in target_means().items()})
        wandb.log(point)
    official = {
        f"relbench/{METRIC[r['task_type']]}/test/{key}": r["value"] * 100.0
        for key, r in results.items()
    }
    for metric in METRIC.values():
        vs = [v for k, v in official.items() if k.startswith(f"relbench/{metric}/")]
        official[f"relbench/{metric}/test/mean"] = float(np.mean(vs))
    wandb.log(official)
    run.finish()


def reduce(model: str, tuned: dict) -> None:
    from rt.data import get_tasks
    from rt.eval import metric_for
    from rt.eval.relbench import _emit_and_score

    preds = LEADERBOARD / model / "preds"
    results, sums = {}, {}
    missing = []
    for db, task in TASKS:
        key = f"{db}/{task}"
        units = []
        for rank in range(N_TOP):
            whole = stage_dir(f"ens-{model}-{db}-{task}-cfg{rank}")
            units.append(
                [whole]
                if (whole / "result.json").exists()
                else [
                    stage_dir(f"ens-{model}-{db}-{task}-cfg{rank}-s{k}")
                    for k in range(N_SEEDS)
                ]
            )
        if key not in tuned or not all(
            (p / "result.json").exists() for parts in units for p in parts
        ):
            missing.append(key)
            continue
        (rt_task,) = get_tasks(PRE_DIR, [(db, task)], ("test",))
        labels = nodes = None
        cfg_sums, curves = [], []
        for parts in units:
            seed_sums = []
            for part in parts:
                st = np.load(part / "state.npz")
                assert int(st["seeds"]) * len(parts) == N_SEEDS, (
                    f"{part}: {int(st['seeds'])} seeds in {len(parts)} parts"
                )
                if labels is None:
                    labels, nodes = st["labels"], st["node_idxs"]
                else:
                    assert np.array_equal(nodes, st["node_idxs"])
                    assert np.array_equal(labels, st["labels"])
                seed_sums.append(st["sum_preds"].astype(np.float64))
            cfg_sums.append(sum(seed_sums))
            if len(parts) == 1:
                curves.append(
                    json.loads((parts[0] / "result.json").read_text())["curve"]
                )
            else:
                curves.append(
                    {
                        str(k): metric_for(
                            rt_task.task_type, labels, sum(seed_sums[:k]) / k
                        )[1]
                        for k in range(1, N_SEEDS + 1)
                    }
                )
        mname, mval, n, align, _ = _emit_and_score(
            preds,
            rt_task,
            PRE_DIR,
            "all-MiniLM-L12-v2",
            labels,
            sum(cfg_sums) / (N_TOP * N_SEEDS),
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
        sums[key] = (rt_task.task_type, labels, cfg_sums)
        print(f"  {key:28s} {mname}={mval:.4f} n={n} {align}", flush=True)
    for key in missing:
        print(f"  {key:28s} not all {N_TOP} units done")
    if not results:
        return
    (LEADERBOARD / model / "results.json").write_text(
        json.dumps(results, indent=1, sort_keys=True) + "\n"
    )
    result = evaluate_submission(preds, verbose=True)
    compare(result)
    if len(results) == len(TASKS):
        log_ensemble(model, sums, results)
    if result["validated"]:
        package([str(preds), "--out", str(LEADERBOARD / model / f"{model}-icl.zip")])


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
    tuned = {}
    for model, _ in MODELS:
        print(f"== {model}: tuning")
        tuned[model] = collect_tuning(model)
    dest = HERE / "tuned_configs.json"
    dest.write_text(json.dumps(tuned, indent=1, sort_keys=True) + "\n")
    print(
        f"wrote {dest} ("
        + ", ".join(f"{m}: {len(t)}/{len(TASKS)}" for m, t in tuned.items())
        + " tasks); commit it before submitting"
    )
    for model, _ in MODELS:
        print(f"== {model}: ensemble")
        reduce(model, tuned[model])


if __name__ == "__main__":
    main()
