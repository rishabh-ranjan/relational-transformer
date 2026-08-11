"""Probe: can `rt.eval:main` run on the pre-Ampere cards of the `il` partition,
and how much slower are they than an a100?

`probe` runs the same fixed eval twice, at two item counts, and reports the
slope: items per second with the one-off cost (clone warm-up, page cache,
model load, context build fixed cost) differenced out.
"""

import time
from pathlib import Path

from roach.slurm import Resources, submit

# The pinned RT-P fine-tune checkpoint `submit_ens_only.ckpt_for` copied for
# this task; read-only here, and not deleted by this probe's teardown.
CKPT = "/dfs/user/ranjanr/ckpts/rtv2/fine-tune-pinned/rel-f1__driver-dnf/swa_steps=4000.safetensors"


def probe(*, run_id: str, num_workers: int, out_root: str) -> None:
    """Two eval passes over the same task, at 512 and 2560 test items."""
    import torch

    from rt.eval._eval import main as eval_main
    from rt.progress import log

    p = torch.cuda.get_device_properties(0)
    log(gpu=p.name.replace(" ", "-"), capability=f"{p.major}.{p.minor}")

    times = {}
    for n in (512, 2560):
        tic = time.time()
        eval_main(
            load_ckpt_path=CKPT,
            embedder="all-MiniLM-L12-v2",
            d_text=384,
            num_blocks=12,
            d_model=512,
            num_heads=8,
            d_ff=2048,
            splits=["test"],
            db_task_list=[("rel-f1", "driver-dnf")],
            pre_dir="/dfs/user/ranjanr/share/stanford-star/relbench-preprocessed",
            tokens_per_gpu=2**18,
            num_workers=num_workers,
            prefetch_factor=2,
            num_walks=10_000,
            walk_length=20,
            items_per_task={"test": n},
            mmap_populate=True,
            shuffle_seed=0,
            context_seed=0,
            vector_db_path=None,
            ctx_size_list=[2048],
            lcs_bw_pl_grid=[(2048, 128, True)],
            val_ensemble_size=1,
            test_ensemble_size=1,
            run_id=f"{run_id}-{n}",
            run_name=None,
            targets={},
            project="probe-gpu",
            entity="rtv2",
            out_root=out_root,
            wandb_disabled=True,
        )
        times[n] = time.time() - tic
        log(probe_items=n, elapsed=f"{times[n]:.1f}s")
    log(
        probe_slope_items_per_s=f"{2048 / (times[2560] - times[512]):.2f}",
        t512=f"{times[512]:.1f}s",
        t2560=f"{times[2560]:.1f}s",
    )


def nodes() -> list[tuple[str, str, str | None, str | None, int]]:
    """(label, gpu type, constraint, nodelist, cpus) per card to probe.

    cpus is what the node can give one gpu without --exclusive: the turing
    nodes have 8 cores per card against the a100 nodes' 14, so their data
    workers are part of what is being measured.
    """
    return [
        ("a100", "a100:1", "ampere", None, 14),
        ("rtx8000", "rtx8000:1", None, "hyperturing2", 14),
        ("2080ti", "2080ti:1", None, "turing3", 8),
        ("titanxp", "titanxp:1", None, "hyperion3", 14),
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
            args=dict(
                num_workers=cpus,
                out_root="/dfs/user/ranjanr/ckpts/probe-gpu",
            ),
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
