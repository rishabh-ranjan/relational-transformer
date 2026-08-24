import functools
import json
from pathlib import Path

from roach.slurm.clusters.ilc import ILC

from roach.slurm import Resources, submit

HERE = Path(__file__).parent

TASKS = (
    ("rel-event", "user-repeat"),
    ("rel-f1", "driver-dnf"),
    ("rel-f1", "driver-top3"),
    ("rel-f1", "driver-position"),
    ("rel-trial", "study-outcome"),
    ("rel-avito", "ad-ctr"),
    ("rel-event", "user-attendance"),
    ("rel-event", "user-ignore"),
    ("rel-trial", "study-adverse"),
    ("rel-trial", "site-success"),
    ("rel-avito", "user-visits"),
    ("rel-avito", "user-clicks"),
    ("rel-hm", "user-churn"),
    ("rel-stack", "user-engagement"),
    ("rel-hm", "item-sales"),
    ("rel-stack", "post-votes"),
    ("rel-amazon", "item-churn"),
    ("rel-amazon", "item-ltv"),
    ("rel-stack", "user-badge"),
    ("rel-amazon", "user-churn"),
    ("rel-amazon", "user-ltv"),
)


@functools.cache
def published_best() -> dict[str, float]:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    stds = json.load(
        open(
            hf_hub_download(
                "stanford-star/relbench", "regression_stds.json", repo_type="dataset"
            )
        )
    )["stds"]

    raw = pd.read_csv(HERE / "results.csv")
    raw["pair"] = raw.dataset + "/" + raw.task
    dflt = raw[raw.config_tag == "default"].assign(arm="D")
    hpo = raw[raw.selected].assign(arm="H")
    d = pd.concat([dflt, hpo])
    d["row"] = d.model + " (" + d.arm + ")"

    out: dict[str, float] = {}
    for task_type, metric, higher in [
        ("BINARY_CLASSIFICATION", "auroc", True),
        ("REGRESSION", "nmae", False),
    ]:
        sub = d[d.task_type == task_type]
        best = max if higher else min
        for split in ("val", "test"):
            v = sub[f"{split}_score"] * 100
            if not higher:
                v = v / sub.pair.map(stds)
            for pair, x in v.groupby(sub.pair):
                out[f"{metric}/{split}/{pair}"] = float(best(x))
            out[f"{metric}/{split}/mean"] = float(best(v.groupby(sub.row).mean()))
    return out


def targets_for(db: str, task: str) -> dict[str, float]:
    keys = {k: v for k, v in published_best().items() if "/test/" in k}
    metrics = {k.split("/")[0] for k in keys if k.endswith(f"/{db}/{task}")}
    return {
        k: v
        for k, v in keys.items()
        if k.endswith(f"/{db}/{task}")
        or (k.endswith("/mean") and k.split("/")[0] in metrics)
    }


def task_type_for(db: str, task: str) -> str:
    import pandas as pd

    raw = pd.read_csv(HERE / "results.csv")
    (task_type,) = set(raw[(raw.dataset == db) & (raw.task == task)].task_type)
    return task_type


def ckpt_for(db: str, task: str, release: str | None) -> str | None:
    if release is None:
        return None
    sub = {"BINARY_CLASSIFICATION": "classification", "REGRESSION": "regression"}
    head = sub[task_type_for(db, task)]
    return f"~/scratch/share/stanford-star/{release}/{head}"


def loss_fn_for(db: str, task: str) -> str:
    return {"BINARY_CLASSIFICATION": "bce", "REGRESSION": "l1"}[task_type_for(db, task)]


def b200(qos: str, time: str, dependency: str | None = None) -> Resources:
    return Resources(
        partition="il",
        account="infolab",
        qos=qos,
        time=time,
        gpus="b200:1",
        cpus_per_task=36,
        ntasks=None,
        exclusive=False,
        mem="375000M",
        mem_per_gpu=None,
        constraint=None,
        nodelist="blackwell1",
        reservation=None,
        dependency=dependency,
    )


def a100(
    qos: str, time: str, reservation: str | None = None, dependency: str | None = None
) -> Resources:
    assert reservation is None or qos == "il-lo", (
        "a reserved node is ours whatever the qos, so a high tier spent there "
        "buys nothing; see ../README.md#a-reservation-is-il-lo-only"
    )
    return Resources(
        partition="il",
        account="infolab",
        qos=qos,
        time=time,
        gpus="a100:1",
        cpus_per_task=14,
        ntasks=None,
        exclusive=False,
        mem=None,
        mem_per_gpu=None,
        constraint="ampere",
        nodelist=None,
        reservation=reservation,
        dependency=dependency,
    )


RESOURCES: dict[tuple[str, str], Resources] = {
    ("rel-f1", "driver-top3"): b200("il-interactive", "12:00:00"),
    ("rel-f1", "driver-dnf"): b200("il-interactive", "12:00:00"),
    ("rel-f1", "driver-position"): a100("il", "1-00:00:00"),
    ("rel-avito", "ad-ctr"): a100("il", "1-00:00:00"),
    ("rel-event", "user-repeat"): a100("il", "1-00:00:00"),
    ("rel-trial", "study-outcome"): a100("il", "1-00:00:00"),
    ("rel-event", "user-ignore"): a100("il", "1-00:00:00"),
    ("rel-event", "user-attendance"): a100("il", "1-00:00:00"),
    ("rel-trial", "study-adverse"): a100("il", "1-00:00:00"),
    ("rel-trial", "site-success"): a100("il", "1-00:00:00"),
    ("rel-avito", "user-visits"): a100("il", "1-00:00:00"),
    ("rel-avito", "user-clicks"): a100("il", "1-00:00:00"),
    ("rel-hm", "user-churn"): a100("il-lo", "2-00:00:00"),
    ("rel-stack", "user-engagement"): a100("il-lo", "2-00:00:00"),
    ("rel-hm", "item-sales"): a100("il-lo", "2-00:00:00"),
    ("rel-stack", "post-votes"): a100("il-lo", "2-00:00:00"),
    ("rel-amazon", "item-churn"): a100("il-lo", "2-00:00:00"),
    ("rel-amazon", "item-ltv"): a100("il-lo", "2-00:00:00"),
    ("rel-stack", "user-badge"): a100("il-lo", "2-00:00:00"),
    ("rel-amazon", "user-churn"): a100("il-lo", "2-00:00:00"),
    ("rel-amazon", "user-ltv"): a100("il-lo", "2-00:00:00"),
}


RUN_IDS: dict[tuple[str, str], str] = {}


def main() -> None:
    for db, task in TASKS:
        resources = RESOURCES[db, task]
        name = f"{db}/{task}"
        print(f"  {name:38s} {resources.gpus} {resources.qos:15s} {resources.time}")
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
                loss_fn=loss_fn_for(db, task),
                load_ckpt_path=ckpt_for(db, task, "rt-plurel"),
                db_task_list=[(db, task)],
                train_splits=["train", "val"],
                pre_dir="~/scratch/share/stanford-star/relbench-preprocessed",
                stage_dir=None,
                tokens_per_gpu=2**18 if resources.gpus.startswith("b200") else 2**17,
                num_workers=resources.cpus_per_task,
                prefetch_factor=2,
                ctx_size_list=[1024],
                local_ctx_size_list=[1024],
                bfs_width_list=[128],
                prefer_latest_list=[False],
                num_walks=0,
                walk_length=20,
                mask_prob_max=0.0,
                items_per_task=1_000_000_000,
                delta_finetune=True,
                optimizer="muon",
                lr=5e-4,
                wd=0.1,
                lr_warmup_steps=0,
                lr_decay_steps=0,
                grad_norm_max=1.0,
                total_bs=256,
                total_steps=25_000,
                early_stop_after_steps=None,
                can_select_init_model=False,
                swa_momentum=0.9995,
                seed=0,
                mmap_populate=True,
                timeout_per_item=10.0,
                eval_freq=100,
                keep_all_ckpts=False,
                vector_db_path=None,
                db_cutoff="test",
                resume_save_mins=20.0,
                eval_splits=["test"],
                eval_db_task_list=[(db, task)],
                eval_pre_dir="~/scratch/share/stanford-star/relbench-preprocessed",
                eval_tokens_per_gpu=2**18,
                eval_num_workers=resources.cpus_per_task,
                eval_prefetch_factor=2,
                eval_num_walks=0,
                eval_walk_length=20,
                eval_items_per_task=2**12,
                eval_ctx_size_list=[1024],
                eval_mmap_populate=True,
                eval_shuffle_seed=0,
                eval_context_seed=0,
                eval_ensemble_size=4,
                eval_vector_db_path=None,
                eval_lcs_bw_pl_grid=[(1024, 128, False)],
                targets=targets_for(db, task),
                project="2026-08-13-fine_tune",
                entity="rtv2",
                run_name=name,
                wandb_disabled=False,
                out_root="~/scratch/ckpts",
            ),
            resources=resources,
            name=f"{db}-{task}",
            repo_root="~/clones/rishabh-ranjan/relational-transformer",
            cluster=ILC,
            job_env="expts/job_env.sh",
            log_root="~/scratch/relational-transformer/fine_tune/slurm-logs",
            clone_root="~/roach_clones",
            secrets_dir="~/scratch/.secrets",
            run_id=RUN_IDS.get((db, task)),
        )


if __name__ == "__main__":
    main()
