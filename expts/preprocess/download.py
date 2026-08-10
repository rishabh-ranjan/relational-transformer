"""Fetch the raw Join once, to a directory every node can read.

    pixi run python expts/preprocess/download.py
    pixi run python expts/preprocess/download.py --repair

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

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from expts.preprocess.submit import RAW_DIR, SOURCE_REPO  # noqa: E402
from huggingface_hub import HfApi, hf_hub_download

ATTEMPTS = 20
# git-lfs opens this many transfers; the batch API is not what rate-limits, so
# this is about saturating the link rather than about staying under a cap.
CONCURRENT_TRANSFERS = 8


def _run(*args: str, cwd: str | None = None) -> int:
    return subprocess.run(args, cwd=cwd).returncode


def _pointers(d: Path) -> list[Path]:
    """Files git-lfs left as pointers: the right name, 130 bytes, no data."""
    out = []
    for p in d.rglob("*"):
        if not p.is_file() or ".git" in p.parts or p.stat().st_size > 1000:
            continue
        try:
            if p.open("rb").read(64).startswith(b"version https://git-lfs"):
                out.append(p)
        except OSError:
            continue
    return out


def download() -> None:
    d, repo = Path(RAW_DIR), SOURCE_REPO
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

    # Without this, `git lfs pull` fetches every object and then says "Skipping
    # object checkout, Git LFS is not installed for this repository" -- leaving a
    # working tree of 130-byte pointers that have the right names and none of the
    # data. It exits 0 while doing it, so nothing downstream notices except the
    # size check, which then reports the whole collection as not downloaded.
    _run("git", "lfs", "install", "--local", cwd=str(d))

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

    # `git lfs pull` can exit 0 having checked out nothing, so the check is the
    # tree, not the exit code.
    left = _pointers(d)
    if left:
        sys.exit(f"{len(left)} files are still LFS pointers in {d}; e.g. {left[0]}")
    n = len(list(d.glob("*/manifest.yaml")))
    print(f"{n} databases in {d}")


def repair() -> int:
    """Bring the raw directory to exactly what the Hub says it is.

    Every file is checked against the Hub's recorded size and only mismatches
    are touched, so this is safe to run at any time and cheap when there is
    nothing wrong. Try `download()` first when there are many to mend -- git-lfs
    fetches in bulk, while this fetches one file at a time and thousands of
    those is what gets rate limited.

    Needed because a raw file can be the right name and the wrong content: an
    interrupted fetch truncates one, and an LFS pointer is a 130-byte file
    standing in for a parquet. `submit.py` refuses to preprocess a database in
    that state, so the symptom is a sweep that stops making progress rather
    than bad output.
    """

    d, repo = Path(RAW_DIR), SOURCE_REPO
    info = HfApi().repo_info(repo, repo_type="dataset", files_metadata=True)
    want = {f.rfilename: (f.size or 0) for f in info.siblings if "/" in f.rfilename}

    def wrong(rel: str, size: int) -> bool:
        p = d / rel
        return not p.is_file() or p.stat().st_size != size

    bad = [r for r, s in want.items() if wrong(r, s)]
    print(f"{len(want)} files in the collection, {len(bad)} wrong or missing")
    if not bad:
        return 0

    for i, rel in enumerate(bad, 1):
        for attempt in range(3):
            try:
                hf_hub_download(
                    repo,
                    rel,
                    repo_type="dataset",
                    local_dir=str(d),
                    force_download=True,
                )
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  FAILED {rel}: {type(e).__name__}: {str(e)[:70]}")
                time.sleep(5 * (attempt + 1))
        if i % 200 == 0:
            print(f"  {i}/{len(bad)}", flush=True)

    still = [r for r, s in want.items() if wrong(r, s)]
    print(f"{len(still)} files still wrong or missing")
    return len(still)


if __name__ == "__main__":
    if "--repair" in sys.argv:
        sys.exit(1 if repair() else 0)
    download()
