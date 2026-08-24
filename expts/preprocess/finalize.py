import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from huggingface_hub import CommitOperationDelete, HfApi

from expts.preprocess.submit import (  # noqa: E402
    CURATED,
    EMBEDDER,
    KEEP,
    LEGACY_DIR,
    OUT_DIR,
    RAW_DIR,
    SOURCE_REPO,
    TARGET_REPO,
)

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


def verify() -> list[str]:
    out, problems = Path(OUT_DIR).expanduser(), []
    expected = sorted(
        p.parent.name for p in Path(RAW_DIR).expanduser().glob("*/manifest.yaml")
    )
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

    for name in sorted(built - set(expected) - set(KEEP)):
        problems.append(f"{name}: built but not in {RAW_DIR} (stale output)")

    print(f"{len(expected)} expected, {len(built)} built, {len(problems)} problem(s)")
    for p in problems[:40]:
        print(f"  {p}")
    if len(problems) > 40:
        print(f"  ... and {len(problems) - 40} more")
    return problems


def verify_legacy() -> list[str]:
    if not LEGACY_DIR:
        return []
    out, problems = Path(LEGACY_DIR).expanduser(), []
    expected = sorted(
        p.parent.name for p in Path(RAW_DIR).expanduser().glob("*/manifest.yaml")
    )
    built = set(databases(out)) if out.is_dir() else set()
    for extra in sorted(built - set(expected)):
        problems.append(f"legacy/{extra}: not a database of this build")
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


SUPPORTED_TASK_TYPES = ("binary_classification", "regression")


def _has_target(task: dict, column_index: dict) -> bool:
    target = task["target_col"]
    return any(
        f"{target} of {table}" in column_index
        for table in (task["entity_table"], task["name"])
    )


def task_lists() -> None:
    out = Path(OUT_DIR).expanduser()
    by_kind: dict[str, list[list[str]]] = defaultdict(list)
    every: list[list[str]] = []
    dropped: list[str] = []
    unsupported: list[str] = []
    for name in databases(out):
        meta = json.loads((out / name / "meta.json").read_text())
        column_index = json.loads((out / name / "column_index.json").read_text())
        for task in meta.get("tasks", []):
            if task.get("task_type") not in SUPPORTED_TASK_TYPES:
                unsupported.append(f"{name}/{task['name']} ({task.get('task_type')})")
                continue
            if not _has_target(task, column_index):
                dropped.append(f"{name}/{task['name']}")
                continue
            every.append([name, task["name"]])
            kind = "autocomplete" if task["kind"] == "autocomplete" else "forecast"
            by_kind[kind].append([name, task["name"]])

    if unsupported:
        print(f"  {len(unsupported)} task(s) left out: unsupported task type, e.g.")
        for task_name in unsupported[:5]:
            print(f"    {task_name}")

    if dropped:
        print(
            f"  {len(dropped)} task(s) left out: target column dropped in "
            "preprocessing (constant), e.g."
        )
        for task_name in dropped[:5]:
            print(f"    {task_name}")

    lists = {
        "all": every,
        "forecast": by_kind.get("forecast", []),
        "autocomplete": by_kind.get("autocomplete", []),
    }
    curated = set()
    if Path(__file__).with_name(CURATED) if CURATED else None:
        curated = set(
            json.loads(
                (Path(__file__).with_name(CURATED) if CURATED else None).read_text()
            )
        )
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


def upload(private: bool = False) -> None:
    problems = verify() + verify_legacy()
    if problems:
        raise SystemExit(
            f"{len(problems)} problem(s); publishing nothing. "
            "Re-run submit.py to fill the gaps, then try again."
        )
    task_lists()

    out, repo = Path(OUT_DIR).expanduser(), TARGET_REPO
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
    stale = sorted(remote_dirs - local - set(KEEP))
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
        commit_message=f"drop {len(stale)} database(s) no longer in {SOURCE_REPO}",
    )
    print("done")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "verify":
        sys.exit(1 if verify() + verify_legacy() else 0)
    elif command == "task-lists":
        task_lists()
    elif command == "upload":
        upload()
    else:
        sys.exit(__doc__)
