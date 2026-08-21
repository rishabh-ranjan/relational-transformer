"""Move the pretraining run to the widest shape the cluster starts right now.

    pixi run python -m expts.pretrain.autoscale <run_id>

Run it on a timer. A pass is idempotent; one with nothing to do prints a line
and exits. The rules it applies are in [README.md](README.md).
"""

import re
import subprocess
import sys
import time

from expts.pretrain import submit as pretrain


def sh(*cmd: str) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def idle_ampere_nodes() -> list[str]:
    """Ampere nodes with zero allocated cpus, in name order."""
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


def warm_nodes() -> list[str]:
    """Ampere nodes the run has been on in the last 12 hours, most recent first."""
    out = sh(
        "sacct",
        "-u",
        sh("whoami").strip(),
        "-n",
        "-X",
        "--name",
        "pretrain",
        "--starttime",
        "now-12hours",
        "-o",
        "JobID,NodeList,End",
        "-P",
    )
    seen: dict[str, str] = {}
    for line in out.splitlines():
        _job, nodelist, end = (line.split("|") + ["", "", ""])[:3]
        if not nodelist or nodelist.startswith("None"):
            continue
        for host in sh("scontrol", "show", "hostnames", nodelist).split():
            if host.startswith("ampere"):
                seen[host] = "9999" if end.startswith("Unknown") else end
    return [h for h, _ in sorted(seen.items(), key=lambda kv: kv[1], reverse=True)]


def current_job() -> dict | None:
    """The pretraining job, as ``{id, state, nodes, hosts, qos}``."""
    fmt = "%i|%T|%D|%N|%q|%b"
    out = sh("squeue", "-u", sh("whoami").strip(), "-h", "-n", "pretrain", "-o", fmt)
    for line in out.splitlines():
        job_id, state, nodes, nodelist, qos, tres = line.split("|")
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


def qos_for(nodes: int, held_il_a100: int) -> str:
    """`il` caps a100 at 10 per user; a node is 8."""
    return "il" if nodes == 1 and held_il_a100 + 8 <= 10 else "il-lo"


def plan(available: list[str], held_il_a100: int) -> tuple[int, str, list[str]]:
    """The widest shape ``available`` supports: ``(nodes, qos, nodelist)``."""
    for n in (4, 2, 1):
        if len(available) >= n:
            return n, qos_for(n, held_il_a100), available[:n]
    return 0, "", []


def submit(run_id: str, nodes: int, qos: str, hosts: list[str]) -> None:
    """Submit, retrying once: every caller has already cancelled the old job, so
    a failure here leaves the run with no job."""
    print(f"  submit {run_id} nodes={nodes} qos={qos} nodelist={','.join(hosts)}")
    for attempt in (1, 2):
        try:
            pretrain.main(run_id, "a100:8", nodes, qos, ",".join(hosts))
            return
        except Exception as e:
            print(f"  submit failed (attempt {attempt}): {e}")
            if attempt == 2:
                raise
            time.sleep(30)


def main() -> None:
    (run_id,) = sys.argv[1:]

    job = current_job()
    idle = idle_ampere_nodes()
    running = job and job["state"] == "RUNNING"
    free = set(idle) | set(job["hosts"] if running else [])
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
            subprocess.run(["scancel", job["id"]], check=True)
            time.sleep(10)
            submit(run_id, nodes, qos, hosts)
        else:
            print("  nothing bigger is free; leaving the run alone")
        return

    if job:
        if nodes:
            print(f"  replacing a pending job with a {nodes}-node job on {hosts}")
            subprocess.run(["scancel", job["id"]], check=True)
            submit(run_id, nodes, qos, hosts)
        else:
            print("  nothing is free; leaving it queued")
        return

    if nodes:
        print(f"  no job; submitting {nodes} nodes on {hosts}")
        submit(run_id, nodes, qos, hosts)
    else:
        qos = qos_for(1, held)
        print(f"  no job and nothing free; queueing 1 node on {qos}")
        submit(run_id, 1, qos, [f"ampere{i}" for i in range(1, 10)])


if __name__ == "__main__":
    main()
