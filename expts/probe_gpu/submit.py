"""Probe: can `rt.eval`'s inference run on the pre-Ampere cards of the `il`
partition, in which dtype, and how much slower than an a100?

`probe` rebuilds what `rt.eval._eval.main` does for a fixed single-config eval
-- load the checkpoint, cast it, score one task -- once per dtype, and times
the scoring pass alone. The cast is the thing under test, so it is here rather
than taken from `rt.eval`.
"""

import time
import traceback
from pathlib import Path

from roach.slurm import Resources, submit

# The pinned RT-P fine-tune checkpoint `submit_ens_only.ckpt_for` copied for
# this task; read-only here.
CKPT = "/dfs/user/ranjanr/ckpts/rtv2/fine-tune-pinned/rel-f1__driver-dnf/swa_steps=4000.safetensors"
PRE_DIR = "/dfs/user/ranjanr/share/stanford-star/relbench-preprocessed"
DB_TASK = ("rel-f1", "driver-dnf")
ITEMS = 512


def probe(*, run_id: str, num_workers: int) -> None:
    """Score the same 512 test rows in bf16, fp16 and fp32, timing each."""
    import torch

    from rt.data import get_tasks
    from rt.eval._eval import build_evaluator, run_and_report
    from rt.model import load_rt_model
    from rt.progress import log

    p = torch.cuda.get_device_properties(0)
    log(gpu=p.name.replace(" ", "-"), capability=f"{p.major}.{p.minor}", run=run_id)
    tasks = get_tasks(PRE_DIR, [DB_TASK], ("test",))

    started = time.time()
    for dtype in (torch.bfloat16,):
        # 2**18 is what the eval jobs use (eval_bs 128 at ctx 2048); the rest
        # find the largest batch each card holds, and -- on a card where every
        # size fits -- say whether the metric moves with the batch size.
        for tokens_per_gpu in (2**18, 2**17, 2**16, 2**15):
            name = f"{dtype}".removeprefix("torch.")
            if time.time() - started > 3600:
                log(skipped=name, tokens_per_gpu=tokens_per_gpu, reason="time_budget")
                continue
            net = None
            try:
                net, config = load_rt_model(CKPT, device="cuda", compile=False)
                net = net.to(dtype)
                # The batches carry bf16 text embeddings whatever the net is,
                # so a net in another dtype needs its inputs cast with it.
                predict = net.predict

                def _predict(batch, *a, _p=predict, _d=dtype, **k):
                    batch = {
                        key: v.to(_d) if v.is_floating_point() else v
                        for key, v in batch.items()
                    }
                    return _p(batch, *a, **k)

                net.predict = _predict
                ev = build_evaluator(
                    tasks,
                    PRE_DIR,
                    embedder=config["embedder"],
                    d_text=config["d_text"],
                    device="cuda",
                    ctx_size_list=[2048],
                    local_ctx_size=2048,
                    bfs_width=128,
                    num_walks=10_000,
                    walk_length=20,
                    tokens_per_gpu=tokens_per_gpu,
                    items_per_task=ITEMS,
                    num_workers=num_workers,
                    context_seed=0,
                    prefer_latest=True,
                    shuffle_seed=0,
                    mmap_populate=True,
                    prefetch_factor=2,
                    vector_db_path=None,
                )
                tic = time.time()
                results = run_and_report(
                    net,
                    tasks,
                    PRE_DIR,
                    ctx_size=2048,
                    csv_out_dir=None,
                    evaluator=ev,
                    embedder=config["embedder"],
                )
                dt = time.time() - tic
                (r,) = results.values()
                log(
                    RESULT=name,
                    eval_bs=ev.eval_bs,
                    n=r["n"],
                    seconds=f"{dt:.1f}",
                    items_per_s=f"{r['n'] / dt:.2f}",
                    metric=r["metric"],
                    value=f"{r['value']:.6f}",
                )
                ev = None
            except Exception as e:
                log(FAILED=name, tokens_per_gpu=tokens_per_gpu, error=type(e).__name__)
                traceback.print_exc()
            finally:
                net = None
                torch.cuda.empty_cache()


def nodes() -> list[tuple[str, str, str | None, str | None, int]]:
    """(label, gpu type, constraint, nodelist, cpus) per card to probe.

    cpus is what the node can give one gpu without --exclusive; hyperturing2
    has 9 physical cores per card, so 9 there rather than the a100 nodes' 14.
    hyperion3's titanxp (sm 61) is not here: this torch has no kernels for it
    at all, so there is nothing to time.
    """
    return [
        # ("a100", "a100:1", "ampere", None, 14),
        ("rtx8000", "rtx8000:1", None, "hyperturing2", 9),
        ("2080ti", "2080ti:1", None, "turing3", 8),
    ]


def main() -> None:
    for label, gpus, constraint, nodelist, cpus in nodes():
        resources = Resources(
            partition="il",
            account="infolab",
            qos="il-lo",
            time="02:00:00",
            gpus=gpus,
            cpus_per_task=cpus,
            ntasks=None,
            exclusive=False,
            mem=None,
            mem_per_gpu=None,
            constraint=constraint,
            nodelist=nodelist,
        )
        print(f"  {label:10s} {resources.gpus} {resources.qos}")
        submit(
            "expts.probe_gpu.submit:probe",
            args=dict(num_workers=cpus),
            resources=resources,
            name=f"probe-gpu-{label}",
            repo_root=str(Path(__file__).resolve().parents[2]),
            log_root="/dfs/user/ranjanr/slurm-logs/rishabh-ranjan/relational-transformer/expts/probe-gpu",
            clone_root="/lfs/local/0/roach_clones",
            secrets_dir="/dfs/user/ranjanr/.secrets",
            run_id=None,
        )


if __name__ == "__main__":
    main()
