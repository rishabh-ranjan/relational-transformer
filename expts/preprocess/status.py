import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from huggingface_hub import HfApi

from expts.preprocess.preprocess import is_done, is_rustler_done  # noqa: E402
from expts.preprocess.submit import (  # noqa: E402
    EMBEDDER,
    LOG_ROOT,
    NAME,
    OUT_DIR,
    RAW_DIR,
    SOURCE_REPO,
    load_sizes,
)

WINDOW = timedelta(hours=1)

RUSTLER_S_PER_GIB_OUT = 21
EMBED_S_PER_GIB_TEXT = 2424 / 4.1

SAMPLE_UNIT = "job-seconds-v1"


def gib(n: float) -> str:
    return f"{n / 2**30:,.1f} GiB"


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
        for p in Path(RAW_DIR).expanduser().glob("*/manifest.yaml")
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


def stuck(names: set[str], done: set[str], limit: int = 8) -> list[str]:
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
        if not db or db not in names:
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
    seen, unique = set(), []
    for entry in reversed(bad):
        db = entry.split(":", 1)[0]
        if db not in seen:
            seen.add(db)
            unique.append(entry)
    return unique[:limit]


MIN_COMPLETIONS = 5


def sample(done_bytes: int, done_count: int) -> tuple[float, int, int] | None:
    now = time.time()
    SAMPLES = Path(LOG_ROOT).expanduser() / "progress.jsonl"
    SAMPLES.parent.mkdir(parents=True, exist_ok=True)
    history = []
    if SAMPLES.exists():
        for line in SAMPLES.read_text().splitlines():
            try:
                row = json.loads(line)
                if row.get("unit") != SAMPLE_UNIT:
                    continue
                history.append((row["t"], row["bytes"], row.get("n", 0)))
            except (json.JSONDecodeError, KeyError):
                continue
    with SAMPLES.open("a") as f:
        f.write(
            json.dumps(
                {"t": now, "bytes": done_bytes, "n": done_count, "unit": SAMPLE_UNIT}
            )
            + "\n"
        )
    inside = [h for h in history if now - h[0] <= WINDOW.total_seconds()]
    return (inside or history or [None])[0]


def databases() -> list[str]:
    try:
        return sorted(
            f.split("/", 1)[0]
            for f in HfApi().list_repo_files(SOURCE_REPO, repo_type="dataset")
            if f.endswith("/manifest.yaml") and f.count("/") == 1
        )
    except Exception:
        return sorted(
            p.parent.name for p in Path(RAW_DIR).expanduser().glob("*/manifest.yaml")
        )


def cost(names: list[str]) -> dict[str, tuple[float, float]]:
    sizes, out = load_sizes(), Path(OUT_DIR).expanduser()
    known_out = sorted(v.get("out", 0) for v in sizes.values())
    known_text = sorted(v.get("text", 0) for v in sizes.values())
    med_out = known_out[len(known_out) // 2] if known_out else 0
    med_text = known_text[len(known_text) // 2] if known_text else 0

    est = {}
    for n in names:
        s = sizes.get(n, {})
        o = s.get("out", med_out)
        local_text = out / n / "text.json"
        text = (
            local_text.stat().st_size
            if local_text.exists()
            else s.get("text", med_text)
        )
        est[n] = (
            o / 2**30 * RUSTLER_S_PER_GIB_OUT,
            text / 2**30 * EMBED_S_PER_GIB_TEXT,
        )
    return est


def report() -> None:
    names = databases()
    out = Path(OUT_DIR).expanduser()
    est = cost(names)

    done = [n for n in names if is_done(out / n, EMBEDDER)]
    total = sum(r + e for r, e in est.values())
    finished = sum(
        (est[n][0] if is_rustler_done(out / n) else 0)
        + (est[n][1] if is_done(out / n, EMBEDDER) else 0)
        for n in names
    )

    print(f"=== {datetime.now():%Y-%m-%d %H:%M:%S}  {NAME} preprocessing")
    print(
        f"databases : {len(done)}/{len(names)}"
        f"   (rustler {sum(is_rustler_done(out / n) for n in names)}/{len(names)})"
    )
    print(
        f"work      : {finished / 3600:,.1f} / {total / 3600:,.1f} estimated job-hours"
        f"  ({finished / max(total, 1):.1%})"
    )
    done_bytes, total_bytes = finished, total

    earlier = sample(int(done_bytes), len(done))
    if earlier and done_bytes > earlier[1]:
        elapsed = time.time() - earlier[0]
        rate = (done_bytes - earlier[1]) / elapsed
        completed = len(done) - earlier[2]
        print(
            f"rate      : {rate * 3600 / 3600:,.1f} job-hours of work per hour"
            f" over the last {elapsed / 60:.0f} min  ({completed} databases)"
        )
        if (
            completed >= MIN_COMPLETIONS
            or (total_bytes - done_bytes) / max(rate, 1e-9) < 3600
        ):
            remaining = (total_bytes - done_bytes) / rate
            print(
                f"eta       : {timedelta(seconds=int(remaining))}"
                f"  -> {datetime.now() + timedelta(seconds=remaining):%a %H:%M}"
            )
        else:
            print(
                f"eta       : too early -- needs {MIN_COMPLETIONS} databases finished "
                f"inside the window, has {completed}"
            )
    else:
        print("rate      : nothing finished yet (run again in a few minutes)")

    states = squeue_states()
    if states:
        print("jobs      : " + "  ".join(f"{v} {k}" for k, v in sorted(states.items())))
    live = running_now()
    if live:
        print(f"running   : {len(live)}")
        for name, where in sorted(live, key=lambda r: -sum(est.get(r[0], (0, 0))))[:8]:
            r, e = est.get(name, (0, 0))
            print(
                f"            {name:34s} rustler {r / 60:5.0f}m  embed {e / 60:5.0f}m  {where}"
            )
    failures = stuck(set(names), set(done))
    if failures:
        print(
            f"stuck     : {len(failures)} not done and not retrying (re-run submit.py)"
        )
        for f in failures:
            print(f"            {f}")


if __name__ == "__main__":
    report()
