import shutil
from pathlib import Path

from expts.repaper.baselines.rel2tabv2.tabpfn import CHECKPOINT, HF_REPO
from expts.repaper.config import SHARE

DEST = Path(SHARE).expanduser() / "tabpfn"


def main() -> None:
    from huggingface_hub import hf_hub_download

    DEST.mkdir(parents=True, exist_ok=True)
    out = DEST / CHECKPOINT
    if out.exists():
        print(f"{out} exists, skipping")
        return
    path = hf_hub_download(repo_id=HF_REPO, filename=CHECKPOINT)
    shutil.copyfile(path, out)
    print(f"fetched {CHECKPOINT} -> {out}")


if __name__ == "__main__":
    main()
