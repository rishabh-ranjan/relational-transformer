from expts.pretrain.submit_marlowe import args
from expts.repaper.config import project
from rt.train import main


def run() -> None:
    for repo, variant in (
        ("rt-j", "classification"),
        ("rt-j", "regression"),
        # ("rt-plurel", "classification"),
        # ("rt-plurel", "regression"),
    ):
        main(
            **args()
            | dict(
                run_name=f"{repo}-{variant}",
                load_ckpt_path=f"~/scratch/hf/stanford-star/{repo}/{variant}",
                lr=0.0,
                total_steps=2,
                eval_freq=1,
                early_stop_after_steps=None,
                db_task_list="expts/pretrain/eval-tasks.json",
                pre_dir="~/scratch/hf/stanford-star/relbench-preprocessed",
                stage_dir=None,
                tokens_per_gpu=2**18,
                num_workers=8,
                keep_all_ckpts=False,
                targets={},
                project=project("pretrain-abl"),
                run_id=f"26-09-09-{repo}-eval-{variant}",
            )
        )
