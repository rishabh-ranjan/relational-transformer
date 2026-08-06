"""Check the build, write its task lists, and publish it.

    pixi run python expts/preprocess/finalize.py <collection> verify
    pixi run python expts/preprocess/finalize.py <collection> task-lists
    pixi run python expts/preprocess/finalize.py <collection> upload

`verify` is not optional politeness: a database whose job was preempted between
rustler and the embedding step leaves a directory that looks finished, and
publishing it would put a hole in the pretraining data that only shows up in a
run weeks later.

`upload` mirrors -- it pushes the build and then deletes the database
directories the Hub has and this build does not, which is what makes it a
replacement rather than a merge with whatever was there before. Root files
(README.md, .gitattributes) are left alone; they are not this sweep's to own,
and neither is anything in the collection's `keep`.

A collection with a `legacy/` tree gets it swapped **only once it is complete**.
It is built in its own directory, verified on its own, and pushed as a separate
step; until then the Hub keeps the previous one. The RT-v1 checkpoints read
that tree, and half of a new one is worse than all of an old one.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from expts.preprocess.collection import Collection, pick  # noqa: E402
from expts.preprocess.submit import EMBEDDER  # noqa: E402

REQUIRED = (
    "meta.json",
    "table_info.json",
    "column_index.json",
    "nodes.rkyv",
    "offsets.rkyv",
    "p2f_adj.rkyv",
    "text.json",
    f"text_emb_{EMBEDDER}.bin",
)


def databases(out: Path) -> list[str]:
    return sorted(p.parent.name for p in out.glob("*/meta.json"))


def verify(c: Collection) -> list[str]:
    """Report every database that is missing, incomplete, or empty."""
    out, problems = Path(c.out_dir), []
    expected = sorted(p.parent.name for p in Path(c.raw_dir).glob("*/manifest.yaml"))
    built = set(databases(out))

    for name in expected:
        if name not in built:
            problems.append(f"{name}: not built")
            continue
        d = out / name
        for f in REQUIRED:
            path = d / f
            if not path.exists():
                problems.append(f"{name}: missing {f}")
            elif path.stat().st_size == 0:
                problems.append(f"{name}: empty {f}")
        meta = json.loads((d / "meta.json").read_text())
        if EMBEDDER not in meta.get("text_embeddings", {}):
            problems.append(f"{name}: meta.json does not record {EMBEDDER}")

    for name in sorted(built - set(expected)):
        problems.append(f"{name}: built but not in {c.raw_dir} (stale output)")

    print(f"{len(expected)} expected, {len(built)} built, {len(problems)} problem(s)")
    for p in problems[:40]:
        print(f"  {p}")
    if len(problems) > 40:
        print(f"  ... and {len(problems) - 40} more")
    return problems


def verify_legacy(c: Collection) -> list[str]:
    """The legacy tree, held to the same standard as the build."""
    if not c.legacy:
        return []
    out, problems = Path(c.legacy_dir), []
    expected = sorted(p.parent.name for p in Path(c.raw_dir).glob("*/manifest.yaml"))
    built = set(databases(out)) if out.is_dir() else set()
    for name in expected:
        if name not in built:
            problems.append(f"legacy/{name}: not built")
            continue
        for f in REQUIRED:
            path = out / name / f
            if not path.exists():
                problems.append(f"legacy/{name}: missing {f}")
            elif path.stat().st_size == 0:
                problems.append(f"legacy/{name}: empty {f}")
    print(
        f"legacy: {len(expected)} expected, {len(built)} built, {len(problems)} problem(s)"
    )
    for p in problems[:20]:
        print(f"  {p}")
    return problems


def upload_legacy(c: Collection, private: bool = False) -> None:
    """Swap the legacy tree in, once all of it is there."""
    from huggingface_hub import HfApi

    if not c.legacy:
        return
    if verify_legacy(c):
        print("legacy tree incomplete; leaving the published one alone")
        return
    api = HfApi()
    print(f"uploading {c.legacy_dir} -> {c.target_repo}/legacy")
    api.upload_large_folder(
        repo_id=c.target_repo,
        repo_type="dataset",
        folder_path=str(Path(c.legacy_dir)),
        # everything lands under legacy/, replacing that tree and nothing else
        path_in_repo="legacy",
    )
    print("legacy swapped")


def task_lists(c: Collection) -> None:
    """Write `db-task-lists/` from the metas this build just produced."""
    out = Path(c.out_dir)
    by_kind: dict[str, list[list[str]]] = defaultdict(list)
    every: list[list[str]] = []
    for name in databases(out):
        meta = json.loads((out / name / "meta.json").read_text())
        for task in meta.get("tasks", []):
            every.append([name, task["name"]])
            by_kind[task["kind"]].append([name, task["name"]])

    lists = {
        "all": every,
        "forecast": by_kind.get("forecast", []),
        "autocomplete": by_kind.get("autocomplete", []),
    }
    curated = set()
    if c.curated_path:
        curated = set(json.loads(c.curated_path.read_text()))
        lists["rt-j"] = [pair for pair in every if pair[0] in curated]
    d = out / "db-task-lists"
    d.mkdir(parents=True, exist_ok=True)
    for stem, pairs in lists.items():
        pairs = sorted(pairs)
        (d / f"{stem}.json").write_text(json.dumps(pairs, indent=1) + "\n")
        dbs = len({p[0] for p in pairs})
        print(f"  {stem + '.json':22s} {len(pairs):6d} tasks over {dbs:4d} databases")

    missing = curated - {p[0] for p in every}
    if missing:
        print(f"  {len(missing)} curated rt-j databases are not in this build:")
        for name in sorted(missing):
            print(f"    {name}")


def upload(c: Collection, private: bool = False) -> None:
    """Push the build, then delete the database directories it replaces."""
    from huggingface_hub import CommitOperationDelete, HfApi

    out, repo = Path(c.out_dir), c.target_repo
    local = set(databases(out))
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)

    print(f"uploading {out} -> {repo}")
    api.upload_large_folder(repo_id=repo, repo_type="dataset", folder_path=str(out))

    remote_dirs = {
        f.split("/", 1)[0]
        for f in api.list_repo_files(repo, repo_type="dataset")
        if "/" in f
    }
    stale = sorted(remote_dirs - local - set(c.keep))
    if not stale:
        print("nothing stale on the Hub; it already mirrors this build")
        return
    print(f"deleting {len(stale)} directory(ies) this build replaces:")
    for name in stale:
        print(f"  {name}")
    api.create_commit(
        repo_id=repo,
        repo_type="dataset",
        operations=[
            CommitOperationDelete(path_in_repo=f"{name}/", is_folder=True)
            for name in stale
        ],
        commit_message=f"drop {len(stale)} database(s) no longer in the-join",
    )
    print("done")


if __name__ == "__main__":
    words = [a for a in sys.argv[1:] if not a.startswith("-")]
    command = words[1] if len(words) > 1 else ""
    c = pick([sys.argv[0], words[0]] if words else sys.argv)
    if command == "verify":
        sys.exit(1 if verify(c) + verify_legacy(c) else 0)
    elif command == "task-lists":
        task_lists(c)
    elif command == "upload":
        if verify(c):
            sys.exit("refusing to upload an incomplete build; fix it and re-run")
        task_lists(c)
        upload(c)
        # separately, and only if all of it is there
        upload_legacy(c)
    else:
        sys.exit(__doc__)
