"""Evaluate the released RT-PluRel checkpoints on the RelBench leaderboard tasks.

Two leaderboard submissions:
- ``--mode synth``: the best synthetic-only pretraining checkpoint
  (``synthetic-pretrain_rdb_1024_size_4b.pt``, same for all tasks) ->
  "RT-PluRel (synth only)".
- ``--mode synth-real``: the task-wise continued-pretraining checkpoints
  (``cntd-pretrain_<db>_<task>.pt``, target task held out) ->
  "RT-PluRel (synth + real)".

Both are in-context: the model never trained on the target task's database
(synth) or task (synth-real). Writes --out-dir as a RelBench submission dir.
"""

from dataclasses import dataclass
from typing import Literal

import tyro

from rt.cli.legacy._driver import LegacyEvalConfig, run_legacy_eval
from rt.model.legacy.plurel import (
    PLUREL_HUB_REPO,
    PLUREL_SYNTH_CKPT,
    PluRelTransformer,
)


@dataclass
class Config(LegacyEvalConfig):
    out_dir: str = "eval_plurel_out"
    mode: Literal["synth", "synth-real"] = "synth"
    ckpt_repo: str = PLUREL_HUB_REPO


def main(cfg: Config) -> None:
    cache = {}

    def model_for_task(task):
        if cfg.mode == "synth":
            filename = PLUREL_SYNTH_CKPT
        else:
            filename = f"cntd-pretrain_{task.db_name}_{task.table_name}.pt"
        if filename not in cache:
            print(f"loading {cfg.ckpt_repo}/{filename}")
            cache.clear()  # at most one legacy checkpoint held in memory
            cache[filename] = PluRelTransformer.from_pretrained(
                filename, repo_id=cfg.ckpt_repo
            )
        return cache[filename]

    run_legacy_eval(cfg, model_for_task)


if __name__ == "__main__":
    main(tyro.cli(Config, description=__doc__))
