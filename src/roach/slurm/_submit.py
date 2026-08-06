"""Submit a python function to slurm.

The submitting side owns everything that needs the repo: it refuses a dirty or
unpushed tree, records the commit, checks the arguments against the target's
signature, and hands slurm a script that reproduces the run from that commit.
Nothing here is site-specific -- paths, account and QOS are arguments.
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from importlib.resources import files
from pathlib import Path
from typing import Any, get_type_hints

from beartype.door import die_if_unbearable

from roach.slurm.resources import Resources
from roach.slurm.target import resolve


def timestamp() -> str:
    """Unique run id: ``yy-mm-dd_hh-mm-ss_ns``.

    Punctuated with ``-`` rather than ``:``: the id names a wandb run (which
    rejects ``:``), an output directory, and a build path (cargo fails under a
    ``:``).
    """
    now = time.time_ns()
    return f"{datetime.fromtimestamp(now / 1e9):%y-%m-%d_%H-%M-%S}_{now % 1_000_000_000:09d}"


@dataclass(frozen=True)
class Job:
    id: str
    run_id: str
    log: Path
    target: str

    @property
    def state(self) -> str:
        out = subprocess.run(
            ["sacct", "-j", self.id, "-n", "--format=State"],
            capture_output=True,
            text=True,
        )
        return out.stdout.split("\n")[0].strip() or "UNKNOWN"


def check_args(target: str, args: dict[str, Any]) -> None:
    """Fail here rather than forty minutes into a job.

    Missing and unknown arguments are caught by name; the values are checked
    against the target's annotations by beartype, so a str where a list of ints
    belongs is a submit-time error too.
    """
    fn = resolve(target)
    sig = inspect.signature(fn)
    hints = get_type_hints(fn)

    unknown = sorted(set(args) - set(sig.parameters))
    if unknown:
        raise TypeError(f"{target} takes no argument(s): {', '.join(unknown)}")
    missing = sorted(
        name
        for name, p in sig.parameters.items()
        if name not in args and p.default is inspect.Parameter.empty
    )
    if missing:
        raise TypeError(f"{target} is missing argument(s): {', '.join(missing)}")
    for name, value in args.items():
        if name in hints:
            try:
                die_if_unbearable(value, hints[name])
            except Exception as e:
                raise TypeError(f"{target}({name}=...): {e}") from None


def preflight(root: Path | str | None = None) -> tuple[str, str]:
    """(repo url, commit) of a clean, pushed tree -- the job clones that."""
    if root is not None:
        os.chdir(root)
    root = _git("rev-parse", "--show-toplevel")
    if _git("status", "--porcelain"):
        raise RuntimeError(f"{root}: working tree is dirty; commit or stash first")
    repo = _git("remote", "get-url", "origin")
    commit = _git("rev-parse", "HEAD")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    _git("fetch", "--quiet", "origin")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, f"origin/{branch}"]
    ).returncode:
        raise RuntimeError(f"{commit} is not on origin/{branch}; push first")
    return repo, commit


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def launch(
    resources: Resources, target: str, args_path: str, overlap: str | None
) -> str:
    """The srun line that starts the ranks, spliced into the batch script.

    One rank per GPU either way; what differs is whose allocation they run in.

    `--export=ALL`, on both. There is one rule, applied at two layers: **nothing
    from the submitting shell, everything from the job's own environment.** These
    sruns are the second layer -- they run *inside* the job, after env.sh has
    built the node-local HOME, the caches, the PATH to pixi and the tokens, and
    without ALL they would start the ranks nearly empty and find none of it. The
    calls that cross from the submitting shell (`sbatch --export=NONE`, and the
    driver step in `_overlap`) are the first layer, and they say the opposite,
    for the same reason: that shell's HOME does not exist on the node and its
    environment holds API tokens slurm would record.

    `--frozen`: the clone is shared, so a rank that re-solved would rewrite
    pixi.lock underneath every other job at this commit.
    """
    run = f'pixi run --frozen python -m roach.slurm.run "{target}" "{args_path}"'
    if overlap is None:
        return f"srun --export=ALL --label --kill-on-bad-exit=1 \\\n    {run}"
    # A step of somebody else's allocation. The shape has to be stated in full:
    # this script is itself running as a one-task step of that allocation, so
    # srun would otherwise inherit *its* shape (one task, one cpu, no gpu of its
    # own) through the SLURM_* variables slurm sets for a step. --overlap is what
    # lets the ranks take resources the driver step is nominally holding --
    # without it the sibling step waits for a step that is waiting for it.
    flags = [
        f"--jobid={overlap}",
        "--overlap",
        "--export=ALL",
        "--label",
        "--kill-on-bad-exit=1",
        f"--ntasks={resources.ranks}",
        f"--cpus-per-task={resources.cpus_per_task}",
    ]
    gpus = int(resources.gpus.rpartition(":")[2])
    if gpus:
        # Per step, not per task. --gpus-per-task hands each rank its own
        # CUDA_VISIBLE_DEVICES holding one card, which every rank then sees as
        # device 0 -- and rank 1, which sets LOCAL_RANK=1 like the batch path
        # does, asks for an ordinal that does not exist ("invalid device
        # ordinal"). The batch path gives every task the node's whole gres and
        # lets LOCAL_RANK index it; this matches that.
        flags.append(f"--gres=gpu:{gpus}")
    return "srun " + " ".join(flags) + f" \\\n    {run}"


def submit(
    target: str,
    args: dict[str, Any],
    resources: Resources,
    *,
    name: str,
    repo_root: Path | str,
    log_root: Path | str,
    clone_root: Path | str,
    secrets_dir: Path | str,
    clone_ttl_days: int,
    setup: tuple[str, ...] = (),
    run_id: str | None = None,
    after: str | None = None,
    overlap: str | None = None,
) -> Job:
    """Run ``target(**args)`` on ``resources``, one rank per GPU.

    ``run_id`` is minted here and injected into ``args`` if the target declares
    it; pass one to relaunch an existing run, which is how a run resumes from a
    checkpoint it wrote earlier.

    ``clone_ttl_days`` is how long an unused clone survives in ``clone_root``
    before a later job sweeps it. It is required for the same reason
    ``Resources`` has no defaults: the job reads nothing from the environment,
    so a value nobody passed would be roach choosing on the experiment's behalf.

    ``after`` is the id of a job this one waits for, so a pipeline whose stages
    want different hardware can be submitted in one pass instead of polling for
    the first stage to finish. The wait is on success: if the dependency fails,
    slurm cancels this job rather than leaving it pending forever.

    ``overlap`` is the id of an allocation somebody is *holding* (see
    ``roach.slurm.interactive``). The run then goes in as a step of that
    allocation instead of as a job of its own: nothing is queued, and a run that
    crashes takes its step down and leaves the allocation standing for the next
    attempt. ``resources`` still says what the ranks get -- gpus, cpus, ntasks --
    but partition, qos, time and node are the allocation's, not this call's.
    """
    os.chdir(repo_root)
    # The job runs from the repo root, so targets are importable relative to it
    # (examples.foo:main). Match that here, or the submit-time check would fail
    # on targets the job can import perfectly well.
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    # roach lives in this repo, so the commit that pins the project pins the
    # roach that runs it too -- there is no second thing to resolve or clone.
    repo, commit = preflight()
    run_id = run_id or timestamp()
    if "run_id" in inspect.signature(resolve(target)).parameters:
        args = {**args, "run_id": run_id}
    check_args(target, args)

    log_root, clone_root = Path(log_root), Path(clone_root)
    log_root.mkdir(parents=True, exist_ok=True)
    args_path = log_root / f"{run_id}.args.json"
    args_path.write_text(json.dumps(args, indent=1, sort_keys=True) + "\n")

    script = files("roach.slurm").joinpath("bootstrap.sh").read_text()
    env_sh = files("roach.slurm").joinpath("env.sh").read_text()
    for key, value in {
        "@REPO@": repo,
        "@COMMIT@": commit,
        "@RUN_ID@": run_id,
        "@NAME@": name,
        "@TARGET@": target,
        "@ARGS@": str(args_path),
        "@LOG_ROOT@": str(log_root),
        "@CLONE_ROOT@": str(clone_root),
        "@CLONE_TTL_DAYS@": str(clone_ttl_days),
        "@SECRETS_DIR@": str(secrets_dir),
        "@SETUP@": "\n".join(setup),
        "@ENV@": env_sh,
        "@LAUNCH@": launch(resources, target, str(args_path), overlap),
    }.items():
        script = script.replace(key, value)

    if overlap is not None:
        return _overlap(script, overlap, name, run_id, log_root, target, resources)

    log = log_root / f"{run_id}_%j.out"
    flags = [
        f"--job-name={name}",
        *resources.sbatch_flags(),
        # The submit dir is node-local to the submit node, so don't start in it.
        "--chdir=/tmp",
        "--propagate=MEMLOCK",
        "--requeue",
        "--open-mode=append",
        # Nothing from this shell belongs in the job: its env points at a home
        # that does not exist on the compute node and holds API tokens, which
        # --export=ALL (sbatch's default) would copy into slurm's job record.
        # The script carries everything it needs.
        "--export=NONE",
        f"--output={log}",
        f"--error={log}",
    ]
    if after:
        # kill-on-invalid-dep, or a dependency that can never be satisfied
        # leaves this job pending until someone notices it by hand.
        flags += [f"--dependency=afterok:{after}", "--kill-on-invalid-dep=yes"]
    # Slurm env vars outrank command-line flags when submitting from inside an
    # allocation, which would silently impose that job's shape on this one.
    env = {
        k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SBATCH_"))
    }
    out = subprocess.run(
        ["sbatch", *flags],
        input=script,
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    job_id = out.stdout.split()[-1]
    print(f"{name}: job {job_id}  run_id {run_id}")
    return Job(
        id=job_id,
        run_id=run_id,
        log=Path(str(log).replace("%j", job_id)),
        target=target,
    )


def _overlap(
    script: str,
    overlap: str,
    name: str,
    run_id: str,
    log_root: Path,
    target: str,
    resources: Resources,
) -> Job:
    """Run the same script as a step of an allocation somebody is holding.

    There is no sbatch here, so nothing queues and nothing requeues: the point of
    an overlapping run is that the *allocation* is the durable thing and the run
    is disposable. The driver step is one task -- it clones, builds and then
    starts the ranks as a sibling step (see `launch`) -- and it is detached from
    this shell, so closing the terminal that submitted it does not kill it.
    """
    log = log_root / f"{run_id}_{overlap}.out"
    flags = [
        f"--jobid={overlap}",
        # the ranks are a step of the same allocation; without this each waits
        # for the other's resources
        "--overlap",
        f"--job-name={name}",
        # One task -- it clones, builds, and starts the ranks -- but every cpu
        # the ranks will want. The ranks are a *nested* step: slurm puts them in
        # a cgroup under this one, so they can only ever be bound to the cpus
        # this step holds, whatever their own --cpus-per-task says. A driver on
        # one cpu produced a training run pinned to a single core (two
        # hyperthreads) with 16 dataloader workers fighting over it, while
        # `scontrol show step` cheerfully reported CPUs=36.
        "--ntasks=1",
        f"--cpus-per-task={resources.ranks * resources.cpus_per_task}",
        "--chdir=/tmp",
        # Nothing from this shell, exactly as the batch path's --export=NONE:
        # its HOME does not exist on the node and its environment holds API
        # tokens slurm would record. Not NONE itself, though, and this is the
        # one place the two layers cannot be spelled the same way: sbatch gives
        # a batch script a default environment with a PATH, while srun execve's
        # the step with what you exported and nothing else -- NONE here died at
        # `execve(): bash: No such file or directory`, before a line of the
        # script ran. So: the two variables that get bash started, and the
        # script builds the rest.
        f"--export=PATH=/usr/local/bin:/usr/bin:/bin,USER={os.environ.get('USER', '')}",
    ]
    env = {
        k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SBATCH_"))
    }
    with open(log, "ab") as fh:
        proc = subprocess.Popen(
            ["srun", *flags, "bash", "-s"],
            stdin=subprocess.PIPE,
            stdout=fh,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
    assert proc.stdin is not None
    proc.stdin.write(script.encode())
    proc.stdin.close()
    print(f"{name}: step of {overlap}  run_id {run_id}  log {log}")
    return Job(id=overlap, run_id=run_id, log=log, target=target)
