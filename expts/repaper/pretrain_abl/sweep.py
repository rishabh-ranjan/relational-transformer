import dataclasses
import subprocess
from pathlib import Path

from roach.slurm import submit
from roach.slurm.clusters import ilc, marlowe

from expts.repaper.config import CLONE_ROOT, LOG_ROOT, SECRETS_DIR

ARMS = (
    ("mask25", 0.25, "expts/pretrain/all_5gb_cutoff.json"),
    ("mask50", 0.5, "expts/pretrain/all_5gb_cutoff.json"),
    ("mask75", 0.75, "expts/pretrain/all_5gb_cutoff.json"),
    ("mix-forecast", 0.0, "expts/repaper/pretrain_abl/cutoff-forecast.json"),
    ("mix-autocomplete", 0.0, "expts/repaper/pretrain_abl/cutoff-autocomplete.json"),
)


def queued(host: str | None) -> set[str]:
    cmd = (
        ["squeue", "-h", "-u", "ranjanr", "-o", "%j"]
        if host is None
        else ["ssh", host, "squeue -h -u $USER -o %j"]
    )
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return set(out.stdout.split())


def main() -> None:
    placements = (
        (
            ilc.ILC,
            dataclasses.replace(ilc.AMPERE_LO, nodes=1, exclude="ampere4"),
            "",
            2**17,
            16,
        ),
        (
            marlowe.MARLOWE,
            # 2026-09-19 23:40: torch import dies on n26 (libnvJitLink.so.13 missing
            # there; the same env runs on n23/n24)
            dataclasses.replace(marlowe.H100, cpus_per_task=14, exclude="n26"),
            "-mw",
            2**17,
            14,
        ),
    )
    busy = {"": queued(None), "-mw": queued("marlowe")}
    for cluster, resources, suffix, tokens_per_gpu, num_workers in placements:
        for run_name, mask_prob_max, db_task_list in ARMS:
            name = f"abl-{run_name}{suffix}"
            if name in busy[suffix]:
                print(f"  {name:28s} queued already")
                continue
            submit(
                "rt.train:main",
                args=dict(
                    embedder="all-MiniLM-L12-v2",
                    d_text=384,
                    num_blocks=12,
                    d_model=512,
                    num_heads=8,
                    d_ff=2048,
                    compile=True,
                    materialize_attn_masks=True,
                    loss_fn="huber",
                    load_ckpt_path="~/scratch/hf/stanford-star/rt-plurel",
                    db_task_list=db_task_list,
                    train_splits=["train"],
                    pre_dir="~/scratch/hf/stanford-star/the-join-lite-preprocessed",
                    stage_dir=None,
                    tokens_per_gpu=tokens_per_gpu,
                    num_workers=num_workers,
                    prefetch_factor=2,
                    ctx_size_list=[512, 1024, 2048, 4096, 8192],
                    local_ctx_size_list=[128, 256, 512, 1024, 2048, 4096, 8192],
                    bfs_width_list=[8, 16, 32, 64, 128, 256],
                    prefer_latest_list=[False, True],
                    num_walks=10_000,
                    walk_length=20,
                    mask_prob_max=mask_prob_max,
                    items_per_task=100_000,
                    delta_finetune=False,
                    optimizer="muon",
                    lr=5e-4,
                    wd=0.1,
                    lr_warmup_steps=2_000,
                    lr_decay_steps=0,
                    grad_norm_max=1.0,
                    total_bs=1024,
                    total_steps=2**15 + 1,
                    early_stop_after_steps=10_000,
                    can_select_init_model=False,
                    swa_momentum=0.9995,
                    seed=0,
                    mmap_populate=True,
                    timeout_per_item=10.0,
                    eval_freq=1_000,
                    keep_all_ckpts=False,
                    vector_db_path=None,
                    db_cutoff=None,
                    resume_save_mins=20.0,
                    eval_splits=["val"],
                    eval_db_task_list="expts/pretrain/eval-tasks.json",
                    eval_pre_dir="~/scratch/hf/stanford-star/relbench-preprocessed",
                    eval_tokens_per_gpu=2**17,
                    eval_num_workers=3,
                    eval_prefetch_factor=2,
                    eval_num_walks=10_000,
                    eval_walk_length=20,
                    eval_items_per_task=1024,
                    eval_ctx_size_list=[8192],
                    eval_mmap_populate=True,
                    eval_shuffle_seed=0,
                    eval_context_seed=0,
                    eval_ensemble_size=1,
                    eval_vector_db_path=None,
                    eval_lcs_bw_pl_grid=[(256, 32, True)],
                    targets={},
                    project="2026-09-14-repaper-pretrain-abl",
                    entity="rtv2",
                    run_name=f"{run_name}{suffix}",
                    wandb_disabled=False,
                    out_root="~/scratch/relational-transformer/pretrain",
                ),
                resources=resources,
                name=name,
                run_id=None,
                inside=None,
                repo_root=str(Path(__file__).resolve().parents[3]),
                cluster=cluster,
                job_env="expts/job_env.sh",
                log_root=f"{LOG_ROOT}/repaper/pretrain_abl/slurm-logs",
                clone_root=CLONE_ROOT if suffix == "" else "~/roach_clones",
                secrets_dir=SECRETS_DIR if suffix == "" else "~/scratch/.secrets",
            )


if __name__ == "__main__":
    main()
