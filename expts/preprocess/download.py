"""Fetch the raw Join once, to a directory every node can read.

    pixi run python expts/preprocess/download.py

Once, and not per job: the collection is 639 databases in 28k files, and a job
that resolved its own database from the Hub would be one of 639 clients doing
it at the same time. The Hub answers that with HTTP 429 long before it finishes.

Retried in a loop because the Hub hands out 429s, and because `snapshot_download`
resumes -- a retry re-checks what is already on disk and fetches the rest, so the
loop converges rather than starting over.

Four workers, and no second downloader alongside it. Concurrency past that does
not go faster: the limit is the Hub's, and exceeding it turns a slow download
into 429s on every request, which is slower still -- measured, three extra
processes at eight workers each rate-limited the whole thing until the retry
loop gave up. The backoff is exponential to five minutes, because a 429 means
come back later rather than come back immediately.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from expts.preprocess.submit import RAW_DIR, SOURCE_REPO  # noqa: E402

ATTEMPTS = 60
WORKERS = 4


def download(repo: str = SOURCE_REPO, local_dir: str = RAW_DIR) -> None:
    from huggingface_hub import snapshot_download

    for attempt in range(1, ATTEMPTS + 1):
        try:
            snapshot_download(
                repo, repo_type="dataset", local_dir=local_dir, max_workers=WORKERS
            )
            break
        except Exception as e:
            print(f"attempt {attempt}: {type(e).__name__}: {str(e)[:160]}", flush=True)
            time.sleep(min(300, 10 * attempt))
    else:
        sys.exit(f"{repo} did not finish downloading in {ATTEMPTS} attempts")

    n = len(list(Path(local_dir).glob("*/manifest.yaml")))
    print(f"{n} databases in {local_dir}")


if __name__ == "__main__":
    download()
