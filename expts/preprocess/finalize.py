"""Check the build, write its task lists, and publish it.

    pixi run python expts/preprocess/finalize.py verify
    pixi run python expts/preprocess/finalize.py task-lists
    pixi run python expts/preprocess/finalize.py upload

`verify` is not optional politeness: a database whose job was preempted between
rustler and the embedding step leaves a directory that looks finished, and
publishing it would put a hole in the pretraining data that only shows up in a
run weeks later.

`upload` mirrors -- it pushes the build and then deletes the database
directories the Hub has and this build does not, which is what makes it a
replacement rather than a merge with whatever was there before. Root files
(README.md, .gitattributes) are left alone; they are not this sweep's to own.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from expts.preprocess.submit import EMBEDDER, OUT_DIR, RAW_DIR  # noqa: E402

REPO = "stanford-star/the-join-preprocessed"
# The 475 databases rt-j trains on. Curated, not derivable from the data -- 126
# databases are excluded wholesale and none partially -- so it is carried from
# the previous build rather than recomputed. Saved before the mirror upload
# overwrites the copy it came from.
RT_J_DBS = Path(__file__).with_name("rt-j-dbs.json")

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


def verify(out_dir: str = OUT_DIR, raw_dir: str = RAW_DIR) -> list[str]:
    """Report every database that is missing, incomplete, or empty."""
    out, problems = Path(out_dir), []
    expected = sorted(p.parent.name for p in Path(raw_dir).glob("*/manifest.yaml"))
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
        problems.append(f"{name}: built but not in {raw_dir} (stale output)")

    print(f"{len(expected)} expected, {len(built)} built, {len(problems)} problem(s)")
    for p in problems[:40]:
        print(f"  {p}")
    if len(problems) > 40:
        print(f"  ... and {len(problems) - 40} more")
    return problems


def task_lists(out_dir: str = OUT_DIR) -> None:
    """Write `db-task-lists/` from the metas this build just produced."""
    out = Path(out_dir)
    by_kind: dict[str, list[list[str]]] = defaultdict(list)
    every: list[list[str]] = []
    for name in databases(out):
        meta = json.loads((out / name / "meta.json").read_text())
        for task in meta.get("tasks", []):
            every.append([name, task["name"]])
            by_kind[task["kind"]].append([name, task["name"]])

    curated = set(json.loads(RT_J_DBS.read_text()))
    lists = {
        "all": every,
        "forecast": by_kind.get("forecast", []),
        "autocomplete": by_kind.get("autocomplete", []),
        "rt-j": [pair for pair in every if pair[0] in curated],
    }
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


def upload(out_dir: str = OUT_DIR, repo: str = REPO, private: bool = False) -> None:
    """Push the build, then delete the database directories it replaces."""
    from huggingface_hub import CommitOperationDelete, HfApi

    out = Path(out_dir)
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
    stale = sorted(remote_dirs - local - {"db-task-lists"})
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
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "verify":
        sys.exit(1 if verify() else 0)
    elif command == "task-lists":
        task_lists()
    elif command == "upload":
        if verify():
            sys.exit("refusing to upload an incomplete build; fix it and re-run")
        task_lists()
        upload()
    else:
        sys.exit(__doc__)
