"""Evaluate the released RT-v1 checkpoints on the RelBench leaderboard tasks.

Uses the task-wise ``pretrain_<db>_<task>.pt`` checkpoints (pretrained with
the target database held out, so the predictions are in-context) and writes
--out-dir as a RelBench submission directory. One leaderboard submission:
"RT-v1" (in-context).
"""

from dataclasses import dataclass

import tyro

from rt.cli.legacy._driver import LegacyEvalConfig, run_legacy_eval
from rt.model.legacy.v1 import V1_HUB_REPO, V1Transformer


@dataclass
class Config(LegacyEvalConfig):
    out_dir: str = "eval_v1_out"
    ckpt_repo: str = V1_HUB_REPO
    ckpt_pattern: str = "pretrain_{db}_{task}.pt"


def main(cfg: Config) -> None:
    def model_for_task(task):
        filename = cfg.ckpt_pattern.format(db=task.db_name, task=task.table_name)
        print(f"loading {cfg.ckpt_repo}/{filename}")
        return V1Transformer.from_pretrained(filename, repo_id=cfg.ckpt_repo)

    run_legacy_eval(cfg, model_for_task)


if __name__ == "__main__":
    main(tyro.cli(Config, description=__doc__))
