"""Fetch the raw Join once, to a directory every node can read.

    pixi run python expts/preprocess/download.py

Once, and not per job: the collection is 639 databases in 28k files, and a job
that resolved its own database from the Hub would be one of 639 clients doing it
at the same time.

**By git, not by the Hub's file API.** `snapshot_download` resolves and fetches
each file separately, so 28k files means 28k `xet-read-token` calls, and the Hub
starts answering those with HTTP 429 long before the download finishes. git-lfs
negotiates through the *batch* API instead -- many objects per request -- so the
same bytes cost orders of magnitude fewer API calls. This is a protocol
difference, not a concurrency one: adding downloader processes to the API path
made it strictly worse (measured -- three extra processes rate-limited it to a
standstill), while one git-lfs pull runs unthrottled.

Resumable in both halves. The clone is skipped if the checkout already exists,
and `git lfs pull` re-checks what is already in `.git/lfs` and fetches the rest,
so an interrupted run converges rather than starting over.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from expts.preprocess.submit import RAW_DIR, SOURCE_REPO  # noqa: E402

ATTEMPTS = 20
# git-lfs opens this many transfers; the batch API is not what rate-limits, so
# this is about saturating the link rather than about staying under a cap.
CONCURRENT_TRANSFERS = 8


def _run(*args: str, cwd: str | None = None) -> int:
    return subprocess.run(args, cwd=cwd).returncode


def download(repo: str = SOURCE_REPO, local_dir: str = RAW_DIR) -> None:
    d = Path(local_dir)
    url = f"https://huggingface.co/datasets/{repo}"

    if not (d / ".git").is_dir():
        d.parent.mkdir(parents=True, exist_ok=True)
        # Pointers first: the clone is then seconds instead of the whole
        # collection, and `git lfs pull` below does the bytes in bulk.
        env_clone = ["env", "GIT_LFS_SKIP_SMUDGE=1", "git", "clone", "--quiet"]
        if _run(*env_clone, url, str(d)):
            sys.exit(f"could not clone {url} into {d}")
    else:
        _run("git", "-c", "lfs.fetchexclude=*", "pull", "--quiet", cwd=str(d))

    for attempt in range(1, ATTEMPTS + 1):
        code = _run(
            "git",
            "-c",
            f"lfs.concurrenttransfers={CONCURRENT_TRANSFERS}",
            "lfs",
            "pull",
            cwd=str(d),
        )
        if code == 0:
            break
        print(f"attempt {attempt}: git lfs pull exited {code}", flush=True)
        time.sleep(min(300, 10 * attempt))
    else:
        sys.exit(f"{repo} did not finish downloading in {ATTEMPTS} attempts")

    n = len(list(d.glob("*/manifest.yaml")))
    print(f"{n} databases in {d}")


if __name__ == "__main__":
    download()
