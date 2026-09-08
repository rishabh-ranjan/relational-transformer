from rt.eval import main


def evaluate(*, checkpoint: str, run_id: str) -> None:
    main(
        load_ckpt_path=checkpoint,
        embedder="all-MiniLM-L12-v2",
        d_text=384,
        num_blocks=12,
        d_model=512,
        num_heads=8,
        d_ff=2048,
        splits=["val"],
        db_task_list="expts/pretrain/eval-tasks.json",
        pre_dir="~/scratch/hf/stanford-star/relbench-preprocessed",
        tokens_per_gpu=2**17,
        num_workers=3,
        prefetch_factor=2,
        num_walks=10_000,
        walk_length=20,
        val_items_per_task=1024,
        test_items_per_task=10_000_000,
        mmap_populate=True,
        shuffle_seed=0,
        context_seed=0,
        vector_db_path=None,
        db_cutoff=None,
        ctx_size_list=[8192],
        lcs_bw_pl_grid=[(256, 32, True)],
        val_ensemble_size=1,
        test_ensemble_size=1,
        run_id=run_id,
        run_name=f"rt-plurel-{checkpoint.rsplit('/', 1)[1]}",
        targets={},
        project="2026-08-19-repaper-pretrain-abl",
        entity="rtv2",
        out_root="~/scratch/relational-transformer/pretrain",
        wandb_disabled=True,
    )


def run() -> None:
    run_id = "26-09-08-rt-plurel-eval"
    for variant in ("classification", "regression"):
        evaluate(
            checkpoint=f"~/scratch/hf/stanford-star/rt-plurel/{variant}",
            run_id=f"{run_id}-{variant}",
        )
