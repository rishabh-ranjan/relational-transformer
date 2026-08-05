"""Where the sweep is, and when it will finish.

    pixi run python expts/preprocess/status.py           # one report
    pixi run python expts/preprocess/status.py --watch   # refresh every 60s

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
from expts.preprocess.submit import EMBEDDER, LOG_ROOT, OUT_DIR, RAW_DIR  # noqa: E402

# Only rate samples inside this window are used, so an ETA reflects how the
# sweep is going now rather than averaging in a slow start or a stall.
WINDOW = timedelta(hours=1)
SAMPLES = Path(LOG_ROOT) / "progress.jsonl"


def gib(n: float) -> str:
    return f"{n / 2**30:,.1f} GiB"


def squeue_states() -> dict[str, int]:
    out = subprocess.run(
        ["squeue", "-h", "-o", "%j %t", "--name=" + ",".join(_job_names())],
        capture_output=True,
        text=True,
    )
    states: dict[str, int] = {}
    for line in out.stdout.splitlines():
        if line.startswith("pre-"):
            states[line.split()[1]] = states.get(line.split()[1], 0) + 1
    return states


def _job_names() -> list[str]:
    return [f"pre-{p.parent.name}" for p in Path(RAW_DIR).glob("*/manifest.yaml")]


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
        if name.startswith("pre-"):
            rows.append((name[4:], f"{elapsed} on {node}"))
    return rows


def recent_failures(limit: int = 8) -> list[str]:
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
    bad = []
    for line in out.stdout.splitlines():
        name, _, state = line.partition("|")
        if name.startswith("pre-") and state.split()[0] not in {
            "COMPLETED",
            "RUNNING",
            "PENDING",
            "REQUEUED",
            "RESIZING",
            "SUSPENDED",
        }:
            bad.append(f"{name[4:]}: {state}")
    return bad[-limit:]


def sample(done_bytes: int) -> tuple[float, int] | None:
    """Append a sample; return the oldest one inside the window, if any."""
    now = time.time()
    SAMPLES.parent.mkdir(parents=True, exist_ok=True)
    history = []
    if SAMPLES.exists():
        for line in SAMPLES.read_text().splitlines():
            try:
                row = json.loads(line)
                history.append((row["t"], row["bytes"]))
            except (json.JSONDecodeError, KeyError):
                continue
    with SAMPLES.open("a") as f:
        f.write(json.dumps({"t": now, "bytes": done_bytes}) + "\n")
    inside = [h for h in history if now - h[0] <= WINDOW.total_seconds()]
    return (inside or history or [None])[0]


def report() -> None:
    sizes = load_sizes()
    names = sorted(p.parent.name for p in Path(RAW_DIR).glob("*/manifest.yaml"))
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

    earlier = sample(done_bytes)
    if earlier and done_bytes > earlier[1]:
        elapsed = time.time() - earlier[0]
        rate = (done_bytes - earlier[1]) / elapsed
        remaining = (total_bytes - done_bytes) / rate
        print(f"rate      : {gib(rate * 3600)}/h over the last {elapsed / 60:.0f} min")
        print(
            f"eta       : {timedelta(seconds=int(remaining))}"
            f"  -> {datetime.now() + timedelta(seconds=remaining):%a %H:%M}"
        )
    else:
        print("rate      : not enough samples yet (run again in a few minutes)")

    states = squeue_states()
    if states:
        print("jobs      : " + "  ".join(f"{v} {k}" for k, v in sorted(states.items())))
    live = running_now()
    if live:
        print(f"running   : {len(live)}")
        for name, where in sorted(live, key=lambda r: -weight.get(r[0], 0))[:8]:
            print(f"            {name:42s} {gib(weight.get(name, 0)):>12s}  {where}")
    failures = recent_failures()
    if failures:
        print(f"failures  : {len(failures)} today (resubmit with submit.py)")
        for f in failures:
            print(f"            {f}")


if __name__ == "__main__":
    while True:
        report()
        if "--watch" not in sys.argv:
            break
        time.sleep(60)
        print()
