"""Submit the pretraining ablations: 5 runs, each one knob off the base recipe.

The base arm of both ablation figures is the released pretraining run itself
(wandb rtv2/2026-08-07-pretrain, run rt-j): every argument here is that run's
(see ../pretrain/submit.py) except the one being ablated, plus early stopping
(10k-step patience on the in-loop val metric) so an arm stops spending nodes
once its curve has flattened, and keep_all_ckpts=False (nothing selects a
checkpoint from an ablation).

Arms:
  mask0 / mask25 / mask75    mask_prob_max in {0.0, 0.25, 0.75} (base: 0.5)
  mix-forecast               trains on rt-j's forecast tasks only
  mix-autocomplete           trains on rt-j's autocomplete tasks only

The mix arms' task lists are the rt-j mixture intersected with the published
forecast/autocomplete lists (3050 and 7765 of the 10815 pairs), written once
to the shared repaper directory so compute nodes can read them.

    pixi run python -m expts.repaper.pretrain_abl.submit           # new runs
    pixi run python -m expts.repaper.pretrain_abl.submit <arm> <run_id>  # resume one
"""

import dataclasses
import json
import sys
from pathlib import Path

from roach.slurm.clusters.ilc import AMPERE_LO, ILC

from expts.repaper.config import (
    CKPT_ROOT,
    CLONE_ROOT,
    JOIN_PRE_DIR,
    LOG_ROOT,
    PRE_DIR,
    SECRETS_DIR,
    SHARE,
    project,
)
from roach.slurm import submit

REPO_ROOT = Path(__file__).resolve().parents[3]
PRE_DIR_JOIN = JOIN_PRE_DIR
LOG_ROOT = f"{LOG_ROOT}/repaper/pretrain_abl"

EVAL_TASKS = [
    (db, task)
    for db, task in json.loads(
        (REPO_ROOT / "expts" / "pretrain" / "eval-tasks.json").read_text()
    )
]


def mix_list(kind: str) -> str:
    """Write (once) and return the path of the rt-j mixture filtered to one
    task family."""
    out = Path(SHARE).expanduser() / "db-task-lists" / f"rt-j-{kind}.json"
    if not out.exists():
        base = Path(PRE_DIR_JOIN) / "db-task-lists"
        rtj = {tuple(p) for p in json.loads((base / "rt-j.json").read_text())}
        fam = {tuple(p) for p in json.loads((base / f"{kind}.json").read_text())}
        pairs = sorted(rtj & fam)
        assert pairs, f"empty {kind} intersection"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps([list(p) for p in pairs], indent=1))
        print(f"wrote {out} ({len(pairs)} pairs)")
    return str(out)


ARMS = {
    "mask0": dict(mask_prob_max=0.0),
    "mask25": dict(mask_prob_max=0.25),
    "mask75": dict(mask_prob_max=0.75),
    "mix-forecast": dict(db_task_list=mix_list("forecast")),
    "mix-autocomplete": dict(db_task_list=mix_list("autocomplete")),
}


def submit_arm(arm: str, run_id: str | None) -> None:
    submit(
        "rt.train:main",
        args=dict(
            # model: RT-J's dims (verbatim from expts/pretrain/submit.py)
            embedder="all-MiniLM-L12-v2",
            d_text=384,
            num_blocks=12,
            d_model=512,
            num_heads=8,
            d_ff=2048,
            compile=True,
            materialize_attn_masks=True,
            loss_fn="huber",
            load_ckpt_path=None,
            # data: the Join's mixture
            db_task_list=f"{PRE_DIR_JOIN}/db-task-lists/rt-j.json",
            train_splits=["train"],
            pre_dir=PRE_DIR_JOIN,
            tokens_per_gpu=2**17,
            num_workers=16,
            prefetch_factor=2,
            ctx_size_list=[512, 1024, 2048, 4096, 8192],
            local_ctx_size_list=[256, 512, 1024, 2048, 4096, 8192],
            bfs_width_list=[8, 16, 32, 64, 128, 256],
            prefer_latest_list=[False, True],
            num_walks=10_000,
            walk_length=20,
            mask_prob_max=0.5,
            items_per_task=100_000,
            # optimization
            delta_finetune=False,
            optimizer="muon",
            lr=5e-4,
            wd=0.1,
            lr_warmup_steps=2_000,
            lr_decay_steps=0,
            grad_norm_max=1.0,
            total_bs=1024,
            total_steps=100_001,
            # the one thing every arm adds over the base run: stop once the
            # in-loop val metric has not improved for 10k steps.
            early_stop_after_steps=10_000,
            can_select_init_model=False,
            swa_momentum=0.9995,
            seed=0,
            mmap_populate=True,
            timeout_per_item=10.0,
            eval_freq=1_000,
            keep_all_ckpts=False,
            vector_db_path=None,
            # The base run predates the db_cutoff knob entirely (its commit's
            # RustlerDataset had none), so no cutoff is what matches it --
            # and "test" would resolve 475 Join sources through the Hub and
            # crash on databases without a test timestamp.
            db_cutoff=None,
            resume_save_mins=20.0,
            # in-loop validation: identical to the base run
            eval_splits=["val"],
            eval_db_task_list=EVAL_TASKS,
            eval_pre_dir=PRE_DIR,
            eval_tokens_per_gpu=2**17,
            eval_num_workers=1,
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
            # logging
            targets={},
            project=project("pretrain-abl"),
            entity="rtv2",
            run_name=arm,
            wandb_disabled=False,
            out_root=CKPT_ROOT,
        )
        | ARMS[arm],
        resources=dataclasses.replace(
            AMPERE_LO,
            qos="il-lo",
            time="21-00:00:00",
            exclusive=True,
            cpus_per_task=16,
        ),
        name=f"pabl-{arm}",
        run_id=run_id,
        repo_root=str(REPO_ROOT),
        cluster=ILC,
        job_env="expts/job_env.sh",
        log_root=LOG_ROOT,
        clone_root=CLONE_ROOT,
        secrets_dir=SECRETS_DIR,
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        submit_arm(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        for arm in ARMS:
            submit_arm(arm, None)
