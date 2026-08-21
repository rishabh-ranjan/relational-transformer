"""Fetch the TabICL v2 checkpoints once, to a directory every node can read.

Compute nodes have no Hub access, so the predictor loads from this directory
(``tabicl_batched.CLF_CHECKPOINT`` / ``REG_CHECKPOINT``). Run on the login
node:

    pixi run python -m expts.repaper.baselines.fetch_tabicl
"""

import shutil
from pathlib import Path

from expts.repaper.baselines.rel2tab.tabicl_batched import (
    CLF_CHECKPOINT,
    HF_REPO,
    REG_CHECKPOINT,
)
from expts.repaper.config import SHARE

DEST = Path(SHARE).expanduser() / "tabicl"


def main() -> None:
    from huggingface_hub import hf_hub_download

    DEST.mkdir(parents=True, exist_ok=True)
    for filename in (CLF_CHECKPOINT, REG_CHECKPOINT):
        out = DEST / filename
        if out.exists():
            print(f"{out} exists, skipping")
            continue
        path = hf_hub_download(repo_id=HF_REPO, filename=filename)
        shutil.copyfile(path, out)
        print(f"fetched {filename} -> {out}")


if __name__ == "__main__":
    main()
