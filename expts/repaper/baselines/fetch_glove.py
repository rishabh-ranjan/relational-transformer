from pathlib import Path

from expts.repaper.baselines.gnn_text_embedder import HF_REPO
from expts.repaper.config import SHARE

DEST = Path(SHARE).expanduser() / "glove"


def main() -> None:
    from huggingface_hub import snapshot_download

    if (DEST / "modules.json").is_file():
        print(f"{DEST} exists, skipping")
        return
    snapshot_download(HF_REPO, local_dir=DEST)
    print(f"fetched {HF_REPO} -> {DEST}")


if __name__ == "__main__":
    main()
