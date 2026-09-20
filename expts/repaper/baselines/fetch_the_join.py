from pathlib import Path


def main(*, repo: str, dest: str, max_workers: int, allow_patterns: list[str] | None) -> None:
    import shutil

    from huggingface_hub import snapshot_download

    out = Path(dest).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    before = shutil.disk_usage(out)
    print(
        f"{repo} -> {out}\n"
        f"  {before.free / 2**30:.0f} GiB free before, "
        f"allow_patterns={allow_patterns}",
        flush=True,
    )

    # snapshot_download resumes, so a requeued or restarted job picks up where
    # it stopped rather than refetching. local_dir gives a plain directory,
    # which is what a pre_dir has to be -- the cache layout is not one.
    snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        local_dir=str(out),
        max_workers=max_workers,
        allow_patterns=allow_patterns,
    )

    after = shutil.disk_usage(out)
    n = sum(1 for _ in out.rglob("*") if _.is_file())
    logical = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    # Logical vs on-disk: /dfs compresses, and the gap is large here because
    # nodes.rkyv packs ~6x. Both are printed so the next person sizing a
    # download does not have to rediscover that.
    print(
        f"done: {n} files, {logical / 2**30:.1f} GiB logical, "
        f"{(before.free - after.free) / 2**30:.1f} GiB consumed on disk, "
        f"{after.free / 2**30:.0f} GiB free",
        flush=True,
    )
