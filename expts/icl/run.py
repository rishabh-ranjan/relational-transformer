import json
import os
import time
import uuid
from pathlib import Path


def main(
    *,
    run_id: str,
    db: str,
    task: str,
    load_ckpt_path: str,
    pre_dir: str,
    ctx_size: int,
    local_ctx_size: int,
    bfs_width: int,
    prefer_latest: bool,
    n_seeds: int,
    items_per_task: int,
    num_walks: int,
    walk_length: int,
    shuffle_seed: int,
    context_seed: int,
    tokens_per_gpu: int,
    num_workers: int,
    prefetch_factor: int,
    mmap_populate: bool,
    db_cutoff: str | int | None,
    run_name: str | None,
    targets: dict[str, float],
    project: str,
    entity: str,
    out_root: str,
    wandb_disabled: bool,
) -> None:
    params = dict(locals())
    out = Path(out_root).expanduser() / entity / project / run_id
    result_path = out / "result.json"
    state_path = out / "state.npz"

    import numpy as np
    import torch
    import wandb

    from rt.data import get_tasks
    from rt.eval import build_evaluator, member_context_seed, metric_for
    from rt.eval._eval import METRIC_NAMES, setup_dist
    from rt.model import load_rt_model
    from rt.progress import log

    if result_path.exists():
        log(stage=run_id, done=str(result_path))
        return

    device, _, _, world_size, ddp = setup_dist(num_workers)
    assert world_size == 1 and not ddp, "one unit is one gpu"
    (rt_task,) = get_tasks(pre_dir, [(db, task)], ("test",))
    metric = METRIC_NAMES[rt_task.task_type]
    cfg = (ctx_size, local_ctx_size, bfs_width, prefer_latest)

    if not wandb_disabled:
        job = os.environ.get("SLURM_JOB_ID")
        attempt = (
            f"{job}.{os.environ.get('SLURM_RESTART_COUNT', '0')}"
            if job
            else f"{int(time.time())}"
        )
        out.mkdir(parents=True, exist_ok=True)
        wandb.init(
            project=project,
            entity=entity,
            name=f"{run_name}-{attempt}" if run_name else attempt,
            id=f"{run_id}-{attempt}",
            group=run_id,
            resume="never",
            config=params,
            dir=str(out),
            settings=wandb.Settings(
                console_multipart=True,
                console_chunk_max_seconds=60,
            ),
        )
        wandb.define_metric("ens_size")
        wandb.define_metric("*", step_metric="ens_size")

    model, config = load_rt_model(load_ckpt_path, device=device, compile=True)
    model = model.to(torch.bfloat16)
    log(stage=run_id, checkpoint=load_ckpt_path, context=str(cfg), seeds=n_seeds)

    curve: dict[str, float] = {}
    sum_preds = labels0 = nodes0 = None
    start = 0
    if state_path.exists():
        st = np.load(state_path)
        sum_preds, labels0, nodes0 = st["sum_preds"], st["labels"], st["node_idxs"]
        start = int(st["seeds"])
        curve = json.loads(str(st["curve"]))
        log(stage=run_id, resumed_from=str(state_path), seeds=start)

    for seed in range(start, n_seeds):
        ev = build_evaluator(
            [rt_task],
            pre_dir,
            embedder=config["embedder"],
            d_text=config["d_text"],
            device=device,
            ctx_size_list=[ctx_size],
            local_ctx_size=local_ctx_size,
            bfs_width=bfs_width,
            prefer_latest=prefer_latest,
            num_walks=num_walks,
            walk_length=walk_length,
            tokens_per_gpu=tokens_per_gpu,
            items_per_task=items_per_task,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            context_seed=member_context_seed(context_seed, seed),
            shuffle_seed=shuffle_seed,
            mmap_populate=mmap_populate,
            vector_db_path=None,
            db_cutoff=db_cutoff,
        )
        ((_t, _ctx, labels, preds_by_prefix, _nl, node_idxs),) = list(
            ev.evaluate_raw([(model, "")], [ctx_size], with_node_idxs=True)
        )
        p = preds_by_prefix[""].astype(np.float64)
        if sum_preds is None:
            sum_preds, labels0, nodes0 = np.zeros_like(p), labels, node_idxs
        assert np.array_equal(nodes0, node_idxs) and np.array_equal(labels0, labels)
        sum_preds += p
        k = seed + 1
        mname, mval = metric_for(rt_task.task_type, labels0, sum_preds / k)
        curve[str(k)] = mval
        log(
            indent=1,
            task=f"{db}/{task}",
            cfg=str(cfg).replace(" ", ""),
            ens_size=k,
            metric=mname,
            value=f"{mval:.4f}",
            n=int(labels0.shape[0]),
        )
        if not wandb_disabled:
            wandb.log(
                {
                    "ens_size": k,
                    f"{metric}/test/{db}/{task}": mval * 100.0,
                    f"{metric}/test/mean": mval * 100.0,
                    **{f"target/{t}": v for t, v in targets.items()},
                }
            )

        out.mkdir(parents=True, exist_ok=True)
        tmp = out / f".tmp.{os.getpid()}.{uuid.uuid4().hex}.npz"
        np.savez(
            tmp,
            sum_preds=sum_preds,
            labels=labels0,
            node_idxs=nodes0,
            seeds=k,
            curve=json.dumps(curve),
        )
        os.replace(tmp, state_path)
        del ev

    result = {
        "task": f"{db}/{task}",
        "task_type": rt_task.task_type,
        "metric": "roc_auc" if rt_task.task_type == "clf" else "nmae",
        "n": int(labels0.shape[0]),
        "curve": curve,
        "config": {
            "ctx_size": ctx_size,
            "local_ctx_size": local_ctx_size,
            "bfs_width": bfs_width,
            "prefer_latest": prefer_latest,
            "n_seeds": n_seeds,
            "shuffle_seed": shuffle_seed,
            "context_seed": context_seed,
            "db_cutoff": db_cutoff,
            "checkpoint": load_ckpt_path,
        },
    }
    tmp = out / f".tmp.{os.getpid()}.{uuid.uuid4().hex}.json"
    tmp.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, result_path)
    if not wandb_disabled:
        wandb.finish()
    log(stage=run_id, done=str(result_path))
