"""Keep the pretraining run on the largest shape the cluster will give it now.

    pixi run python -m expts.pretrain.autoscale <run_id>          # act
    pixi run python -m expts.pretrain.autoscale <run_id> --dry-run

Run it on a timer. One pass reads what is free, decides the shape, and gets
there; it is idempotent, so a pass that has nothing to do prints one line and
exits.

The policy, in one paragraph. More nodes is strictly better -- the run is
data-parallel and resumes at any GPU count -- so take 4 whole nodes when 4 are
free, else 2, else 1. Only *whole* nodes count: the job is `--exclusive`
because the mixture wants each node's page cache, so a node carrying anyone
else's work is no use. Never queue for an upgrade: a bigger shape is worth
having only if slurm starts it immediately, and the way to be sure of that is
to name exactly the nodes that are already idle. A single node is the one shape
that fits under `il`, whose 10-a100-per-user cap stops at one node's eight but
which is not preemptible -- so a 1-node run goes there when the cap allows, and
only falls back to il-lo when it does not.

Among nodes that are equally free, it asks for the ones this run has been on in
the last few hours: they still hold the mixture in their page cache, which is
nearly all of time-to-first-step. That is an ordering, never a wait.

What it will not do is trade a running job for a marginal gain: an upgrade
cancels a job that is making progress and pays for page-cache population again
-- ~25-45 minutes on a node the run has not used recently, a few minutes on one
still holding the mixture -- so it happens only when the node count goes up.
Downgrades are free by comparison -- the job is already stopped when they
happen (preempted, or never started) -- so they are taken eagerly.
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
REPO_ROOT = HERE.parents[1]
JOB_NAME = "pretrain"
# `il` caps a100 at 10 per user. A whole node is 8, so the run fits there only
# when the rest of this user's jobs are holding 2 or fewer.
IL_A100_CAP = 10
NODE_GPUS = 8
SHAPES = (4, 2, 1)
# How far back a node counts as "warm" -- still holding the mixture in its page
# cache. Short on purpose: the guess is free when right and costs nothing when
# wrong, but a stale one would keep steering the run at a cold node.
WARM_HOURS = 12


def all_amperes() -> list[str]:
    return [f"ampere{i}" for i in range(1, 10)]


def sh(*cmd: str) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def idle_ampere_nodes() -> list[str]:
    """Ampere nodes with nothing at all running on them, in name order.

    Allocated CPUs, not free GPUs, is the test: an `--exclusive` job needs the
    whole node, so a node whose GPUs are idle but which carries someone's
    cpu-only job cannot take it.
    """
    out = []
    for line in sh("sinfo", "-p", "il", "-h", "-o", "%n|%t|%C").splitlines():
        name, state, cpus = line.split("|")
        if not name.startswith("ampere") or state.startswith(("down", "drain")):
            continue
        if int(cpus.split("/")[0]) == 0:
            out.append(name)
    return sorted(out, key=lambda n: int(n.removeprefix("ampere")))


def il_a100_held(exclude_job: str | None) -> int:
    """a100s this user holds under the `il` QOS, ignoring one job."""
    held = 0
    fmt = "%i|%q|%b|%T"
    for line in sh("squeue", "-u", sh("whoami").strip(), "-h", "-o", fmt).splitlines():
        job_id, qos, tres, state = line.split("|")
        if job_id == exclude_job or qos != "il" or state != "RUNNING":
            continue
        m = re.search(r"gpu:a100:(\d+)", tres)
        if m:
            held += int(m.group(1))
    return held


def warm_nodes(within_hours: int = WARM_HOURS) -> list[str]:
    """Ampere nodes this run has been on recently, most recent first.

    The mixture stays in a node's page cache after the job ends, and populating
    it is nearly all of time-to-first-step: coming back to a node the run left
    an hour ago costs minutes where a cold node costs tens of them. So among
    nodes that are *equally free right now*, ask for the warm ones by name.

    This never makes the run wait: the caller only ever orders nodes it has
    already established are idle, so naming them still starts immediately --
    the never-queue-for-a-shape rule is untouched.

    Warmth decays invisibly (another job's data evicts ours), so this is a
    preference, not a guarantee, and the window is deliberately short.
    """
    fmt = "JobID,NodeList,End"
    out = sh(
        "sacct",
        "-u",
        sh("whoami").strip(),
        "-n",
        "-X",
        "--name",
        JOB_NAME,
        "--starttime",
        f"now-{within_hours}hours",
        "-o",
        fmt,
        "-P",
    )
    seen: dict[str, str] = {}
    for line in out.splitlines():
        _job, nodelist, end = (line.split("|") + ["", "", ""])[:3]
        if not nodelist or nodelist.startswith("None"):
            continue
        for host in sh("scontrol", "show", "hostnames", nodelist).split():
            if host.startswith("ampere"):
                # "Unknown" is a job still running: warmest there is.
                seen[host] = "9999" if end.startswith("Unknown") else end
    return [h for h, _ in sorted(seen.items(), key=lambda kv: kv[1], reverse=True)]


def current_job() -> dict | None:
    """The pretraining job, as ``{id, state, nodes, nodelist, qos}``."""
    fmt = "%i|%T|%D|%N|%q|%b"
    out = sh("squeue", "-u", sh("whoami").strip(), "-h", "-n", JOB_NAME, "-o", fmt)
    for line in out.splitlines():
        job_id, state, nodes, nodelist, qos, tres = line.split("|")
        # This policy is the ampere one; a b200 run is placed by hand.
        assert "a100" in tres, f"job {job_id} holds {tres}, not an a100 shape"
        hosts = (
            sh("scontrol", "show", "hostnames", nodelist).split() if nodelist else []
        )
        return {
            "id": job_id,
            "state": state,
            "nodes": int(nodes),
            "hosts": hosts,
            "qos": qos,
        }
    return None


def plan(available: list[str], held_il_a100: int) -> tuple[int, str, list[str]]:
    """The best shape ``available`` supports: ``(nodes, qos, nodelist)``."""
    for n in SHAPES:
        if len(available) < n:
            continue
        nodes = available[:n]
        # One node fits under il's cap, and il cannot be preempted.
        if n == 1 and held_il_a100 + NODE_GPUS <= IL_A100_CAP:
            return n, "il", nodes
        return n, "il-lo", nodes
    return 0, "", []


def submit(run_id: str, nodes: int, qos: str, hosts: list[str], dry: bool) -> None:
    cmd = [
        sys.executable,
        "-m",
        "expts.pretrain.submit",
        run_id,
        "--gpus",
        "a100:8",
        "--nodes",
        str(nodes),
        "--qos",
        qos,
        "--nodelist",
        ",".join(hosts),
    ]
    print("  $ " + " ".join(cmd))
    if dry:
        return
    # A failing submit says nothing on stdout -- the reason is on stderr and the
    # exit code -- so both are printed. Every path that reaches here after a
    # `scancel` has already given the nodes up, and a swallowed error would
    # leave the run with no job at all.
    #
    # `submit.py` refuses a dirty or unpushed tree, and this clone is shared with
    # other sessions, so one pass can fail on a working tree that is clean again
    # a minute later. Retry once before giving up.
    for attempt in (1, 2):
        out = subprocess.run(cmd, text=True, capture_output=True, cwd=REPO_ROOT)
        if out.returncode == 0 and out.stdout.strip():
            print(out.stdout.strip())
            return
        print(f"  submit failed (attempt {attempt}, exit {out.returncode})")
        for stream in (out.stdout, out.stderr):
            if stream.strip():
                print("    " + stream.strip().replace("\n", "\n    "))
        if attempt == 1:
            time.sleep(30)
    raise SystemExit(
        f"  submit failed twice: run {run_id} now has NO job. Fix the cause above "
        f"and resubmit with\n  $ " + " ".join(cmd)
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_id")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    job = current_job()
    idle = idle_ampere_nodes()
    # This job's own nodes come back to the pool if it is cancelled, so an
    # upgrade from 2 to 4 needs two *more* free nodes, not four.
    running = job and job["state"] == "RUNNING"
    free = set(idle) | set(job["hosts"] if running else [])
    # Warm first (most recently used first), then the rest by node number. Only
    # the ordering changes -- every node here is already free, so preferring a
    # warm one buys a first step in minutes instead of tens of minutes without
    # ever making the job wait for a particular host.
    warm = [n for n in warm_nodes() if n in free]
    available = warm + sorted(
        free - set(warm), key=lambda n: int(n.removeprefix("ampere"))
    )
    held = il_a100_held(exclude_job=job["id"] if job else None)
    nodes, qos, hosts = plan(available, held)
    state = f"{job['nodes']}-node {job['qos']} ({job['state']})" if job else "no job"
    print(
        f"job: {state}  idle: {idle or '-'}  warm+free: {warm or '-'}  "
        f"best: {nodes or '-'}-node {qos}"
    )

    if job and job["state"] not in ("RUNNING", "PENDING"):
        print("  job is neither running nor pending; leaving it alone")
        return

    if running:
        if nodes > job["nodes"]:
            print(f"  upgrading {job['nodes']} -> {nodes} nodes on {hosts}")
            if not args.dry_run:
                subprocess.run(["scancel", job["id"]], check=True)
                subprocess.run(["sleep", "10"], check=True)
            submit(args.run_id, nodes, qos, hosts, args.dry_run)
        else:
            print("  nothing bigger is free; leaving the run alone")
        return

    if job:  # pending: it is not making progress, so any startable shape wins
        if nodes:
            print(f"  replacing a pending job with a {nodes}-node job on {hosts}")
            if not args.dry_run:
                subprocess.run(["scancel", job["id"]], check=True)
            submit(args.run_id, nodes, qos, hosts, args.dry_run)
        else:
            print("  nothing is free; leaving it queued")
        return

    if nodes:
        print(f"  no job; submitting {nodes} nodes on {hosts}")
        submit(args.run_id, nodes, qos, hosts, args.dry_run)
    else:
        # Nothing whole is free. Queue the smallest shape rather than wait for a
        # pass that finds one: queued costs nothing, and an upgrade can follow.
        # A queued single node still belongs on `il` when the cap allows -- it
        # is not preemptible, and `plan` returns no qos to inherit here because
        # it found no hosts, so apply the same rule directly.
        queued_qos = "il" if held + NODE_GPUS <= IL_A100_CAP else "il-lo"
        print(f"  no job and nothing free; queueing 1 node on {queued_qos}")
        submit(args.run_id, 1, queued_qos, all_amperes(), args.dry_run)


if __name__ == "__main__":
    main()
