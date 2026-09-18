import dataclasses
import json
import subprocess
from pathlib import Path

from roach.slurm import submit
from roach.slurm.clusters import marlowe

from expts.repaper.config import LOG_ROOT, OUT_ROOT, PRE_DIR

PIECES = (
    ("rel-stack", "post-votes", 0),
    ("rel-stack", "post-votes", 1),
    ("rel-stack", "post-votes", 2),
    ("rel-hm", "user-churn", 0),
    ("rel-hm", "user-churn", 1),
    ("rel-hm", "user-churn", 2),
    ("rel-stack", "user-badge", 2),
)


def queued() -> set[str]:
    out = subprocess.run(
        ["ssh", "marlowe", "squeue -h -u $USER -o %j"],
        capture_output=True,
        text=True,
        check=True,
    )
    return set(out.stdout.split())


def main() -> None:
    busy = queued()
    cfgs = json.loads(
        (Path(__file__).parent.parent / "tune" / "tuned_configs.json").read_text()
    )
    for db, table, rank in PIECES:
        name = f"subp-cfg{rank}-{db}-{table}-mw"
        if name in busy:
            print(f"  {name:44s} queued already")
            continue
        ctx, lcs, bw, pl = cfgs[f"{db}/{table}"]["top_cfgs"][rank]
        submit(
            "expts.repaper.enscurve.run:main",
            args=dict(
                variant=f"cfg{rank}",
                db=db,
                table=table,
                ctx_size=int(ctx),
                local_ctx_size=int(lcs),
                bfs_width=int(bw),
                prefer_latest=bool(pl),
                n_seeds=4,
                items_per_task=10_000_000,
                split="test",
                pre_dir=PRE_DIR,
                out_dir=f"{OUT_ROOT}/repaper-submit-rt-plurel/cfg{rank}",
                num_walks=10_000,
                walk_length=20,
                shuffle_seed=0,
                context_seed=0,
                tokens_per_gpu=2**18,
                num_workers=8,
                prefetch_factor=2,
                mmap_populate=True,
                db_cutoff=None,
                ckpt="~/scratch/hf/stanford-star/rt-plurel",
            ),
            resources=dataclasses.replace(
                marlowe.H100,
                gpus="1",
                exclusive=False,
                cpus_per_task=14,
                time="6:00:00",
            ),
            name=name,
            repo_root=str(Path(__file__).resolve().parents[3]),
            cluster=marlowe.MARLOWE,
            job_env="expts/job_env.sh",
            log_root=f"{LOG_ROOT}/repaper/submit/slurm-logs",
            clone_root="~/roach_clones",
            secrets_dir="~/scratch/.secrets",
        )


if __name__ == "__main__":
    main()
