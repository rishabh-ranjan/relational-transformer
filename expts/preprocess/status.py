"""Where the sweep is, and when it will finish.

    pixi run python expts/preprocess/status.py
    watch -n60 pixi run python expts/preprocess/status.py   # if you want it live

Progress is measured in bytes, not in databases. Counting databases would say
this sweep was 97% done while a quarter of the work remained, because 20 of the
639 are half the output. Each database's share is what the previous build's
output measured (`sizes.py`), so a database that is finished contributes its
real weight and the ETA is against work, not against a count.

The rate comes from samples this script leaves in `progress.jsonl` next to the
job logs, so an ETA is available from the second invocation onward and does not
assume the sweep started when you happened to look.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from expts.preprocess.preprocess import is_done  # noqa: E402
from expts.preprocess.sizes import load as load_sizes  # noqa: E402
from expts.preprocess.submit import (  # noqa: E402
    EMBEDDER,
    LOG_ROOT,
    OUT_DIR,
    RAW_DIR,
    SOURCE_REPO,
)

# Only rate samples inside this window are used, so an ETA reflects how the
# sweep is going now rather than averaging in a slow start or a stall.
WINDOW = timedelta(hours=1)
SAMPLES = Path(LOG_ROOT) / "progress.jsonl"


def gib(n: float) -> str:
    return f"{n / 2**30:,.1f} GiB"


# Both stages, because a database is "being worked on" in either. Reading only
# one prefix is how a database whose embed job is running gets reported as a
# failure, on the strength of a superseded rustler job from before a fix.
STAGES = ("pre-", "emb-")


def _strip_stage(name: str) -> str | None:
    for s in STAGES:
        if name.startswith(s):
            return name[len(s) :]
    return None


def squeue_states() -> dict[str, int]:
    out = subprocess.run(
        ["squeue", "-h", "-o", "%j %t", "--name=" + ",".join(_job_names())],
        capture_output=True,
        text=True,
    )
    states: dict[str, int] = {}
    for line in out.stdout.splitlines():
        if _strip_stage(line):
            states[line.split()[1]] = states.get(line.split()[1], 0) + 1
    return states


def _job_names() -> list[str]:
    return [
        f"{s}{p.parent.name}"
        for p in Path(RAW_DIR).glob("*/manifest.yaml")
        for s in STAGES
    ]


def running_now() -> list[tuple[str, str]]:
    out = subprocess.run(
        [
            "squeue",
            "-h",
            "-u",
            subprocess.run(
                ["id", "-un"], capture_output=True, text=True
            ).stdout.strip(),
            "-t",
            "RUNNING",
            "-o",
            "%j|%M|%R",
        ],
        capture_output=True,
        text=True,
    )
    rows = []
    for line in out.stdout.splitlines():
        name, elapsed, node = (line.split("|") + ["", ""])[:3]
        db = _strip_stage(name)
        if db:
            rows.append((db, f"{name[:3]} {elapsed} on {node}"))
    return rows


def _queued() -> set[str]:
    out = subprocess.run(
        [
            "squeue",
            "-h",
            "-u",
            subprocess.run(
                ["id", "-un"], capture_output=True, text=True
            ).stdout.strip(),
            "-o",
            "%j",
        ],
        capture_output=True,
        text=True,
    )
    return {db for n in out.stdout.split() if (db := _strip_stage(n))}


def stuck(done: set[str], limit: int = 8) -> list[str]:
    """Databases whose last attempt failed and that nothing is retrying.

    Not "jobs that failed today": a database that failed, was resubmitted and
    succeeded would be reported as broken forever, which trains you to ignore
    the line. What matters is whether it is finished or on its way -- a failure
    with a successful retry behind it is history, not a problem.
    """
    out = subprocess.run(
        [
            "sacct",
            "-n",
            "-X",
            "-P",
            "-S",
            "today",
            "--format=JobName%60,State",
            "-u",
            subprocess.run(
                ["id", "-un"], capture_output=True, text=True
            ).stdout.strip(),
        ],
        capture_output=True,
        text=True,
    )
    live = {name for name, _ in (r for r in running_now())} | _queued()
    bad = []
    for line in out.stdout.splitlines():
        name, _, state = line.partition("|")
        db = _strip_stage(name) if state else None
        if not db:
            continue
        if db in done or db in live:
            continue
        if state.split()[0] not in {
            "COMPLETED",
            "RUNNING",
            "PENDING",
            "REQUEUED",
            "RESIZING",
            "SUSPENDED",
        }:
            bad.append(f"{db}: {state}")
    # A database is stuck once, however many attempts it took to get there.
    seen, unique = set(), []
    for entry in reversed(bad):
        db = entry.split(":", 1)[0]
        if db not in seen:
            seen.add(db)
            unique.append(entry)
    return unique[:limit]


# An ETA from one or two finished databases is arithmetic, not information:
# extrapolating the first minutes of a sweep whose work spans three orders of
# magnitude gives answers off by weeks. Say so instead.
MIN_COMPLETIONS = 5


def sample(done_bytes: int, done_count: int) -> tuple[float, int, int] | None:
    """Append a sample; return the oldest one inside the window, if any."""
    now = time.time()
    SAMPLES.parent.mkdir(parents=True, exist_ok=True)
    history = []
    if SAMPLES.exists():
        for line in SAMPLES.read_text().splitlines():
            try:
                row = json.loads(line)
                history.append((row["t"], row["bytes"], row.get("n", 0)))
            except (json.JSONDecodeError, KeyError):
                continue
    with SAMPLES.open("a") as f:
        f.write(json.dumps({"t": now, "bytes": done_bytes, "n": done_count}) + "\n")
    inside = [h for h in history if now - h[0] <= WINDOW.total_seconds()]
    return (inside or history or [None])[0]


def collection() -> list[str]:
    """Every database the build will contain, downloaded yet or not.

    Taken from the source repo rather than from what has landed locally: while
    the download is still running, a local count makes the total grow under you
    and the percentage go backwards.
    """
    try:
        from huggingface_hub import HfApi

        return sorted(
            {
                f.split("/", 1)[0]
                for f in HfApi().list_repo_files(SOURCE_REPO, repo_type="dataset")
                if f.count("/") >= 1 and f.startswith("join-")
            }
        )
    except Exception:  # offline: fall back to what is on disk
        return sorted(p.parent.name for p in Path(RAW_DIR).glob("*/manifest.yaml"))


def report() -> None:
    sizes = load_sizes()
    names = collection()
    out = Path(OUT_DIR)
    # A database the previous build never had has no expected size; charge it
    # the median so it is neither invisible nor dominant.
    known = sorted(sizes[n] for n in names if n in sizes)
    fallback = known[len(known) // 2] if known else 0
    weight = {n: sizes.get(n, fallback) for n in names}

    done = [n for n in names if is_done(out / n, EMBEDDER)]
    total_bytes = sum(weight.values())
    done_bytes = sum(weight[n] for n in done)

    print(f"=== {datetime.now():%Y-%m-%d %H:%M:%S}  the-join preprocessing")
    print(f"databases : {len(done)}/{len(names)}")
    print(
        f"work      : {gib(done_bytes)} / {gib(total_bytes)}"
        f"  ({done_bytes / max(total_bytes, 1):.1%})"
    )

    earlier = sample(done_bytes, len(done))
    if earlier and done_bytes > earlier[1]:
        elapsed = time.time() - earlier[0]
        rate = (done_bytes - earlier[1]) / elapsed
        finished = len(done) - earlier[2]
        print(
            f"rate      : {gib(rate * 3600)}/h over the last {elapsed / 60:.0f} min"
            f"  ({finished} databases)"
        )
        if finished >= MIN_COMPLETIONS:
            remaining = (total_bytes - done_bytes) / rate
            print(
                f"eta       : {timedelta(seconds=int(remaining))}"
                f"  -> {datetime.now() + timedelta(seconds=remaining):%a %H:%M}"
            )
        else:
            print(
                f"eta       : too early -- needs {MIN_COMPLETIONS} databases finished "
                f"inside the window, has {finished}"
            )
    else:
        print("rate      : nothing finished yet (run again in a few minutes)")

    states = squeue_states()
    if states:
        print("jobs      : " + "  ".join(f"{v} {k}" for k, v in sorted(states.items())))
    live = running_now()
    if live:
        print(f"running   : {len(live)}")
        for name, where in sorted(live, key=lambda r: -weight.get(r[0], 0))[:8]:
            print(f"            {name:42s} {gib(weight.get(name, 0)):>12s}  {where}")
    failures = stuck(set(done))
    if failures:
        print(
            f"stuck     : {len(failures)} not done and not retrying (re-run submit.py)"
        )
        for f in failures:
            print(f"            {f}")


if __name__ == "__main__":
    report()
