"""roach.slurm: the parts that are pure functions, and so are the parts that bit us.

Every check here corresponds to a failure that cost real time on the cluster:
arguments that do not match the target, a resource shape that slurm rejects, and
a batch script that has lost a placeholder.
"""

from __future__ import annotations

import inspect

import pytest

from roach.slurm import Resources, check_args, resolve, timestamp
from roach.slurm._submit import submit as submit_fn


def sample(a: int, b: str, c: list[int], run_id: str) -> None:  # noqa: ARG001
    pass


def test_resolve_requires_module_attr():
    assert resolve("roach.slurm._submit:timestamp") is timestamp
    with pytest.raises(ValueError):
        resolve("roach.slurm._submit")


def test_timestamp_has_no_characters_that_break_wandb_or_cargo():
    t = timestamp()
    assert ":" not in t and "/" not in t


def test_check_args_accepts_a_matching_call():
    check_args(f"{__name__}:sample", {"a": 1, "b": "x", "c": [1, 2], "run_id": "r"})


@pytest.mark.parametrize(
    "args, message",
    [
        ({"a": 1, "b": "x", "c": [1], "run_id": "r", "d": 0}, "no argument"),
        ({"a": 1, "b": "x"}, "missing argument"),
        ({"a": "not an int", "b": "x", "c": [1], "run_id": "r"}, "a="),
        ({"a": 1, "b": "x", "c": ["not an int"], "run_id": "r"}, "c="),
    ],
)
def test_check_args_rejects(args, message):
    with pytest.raises(TypeError, match=message):
        check_args(f"{__name__}:sample", args)


def ampere(**over) -> Resources:
    kwargs = dict(
        partition="il",
        account="infolab",
        qos="il",
        time="7-00:00:00",
        gpus="a100:8",
        cpus_per_task=16,
        ntasks=None,
        exclusive=True,
        mem=None,
        constraint="ampere",
        nodelist=None,
    )
    return Resources(**{**kwargs, **over})


def test_gpus_may_name_a_type_or_just_a_count():
    """A bare count is how one job shape reaches nodes carrying different cards:
    naming a type pins the job to the nodes that have it, which for a sweep that
    should land anywhere free is an artificial constraint."""
    assert "--gres=gpu:2" in ampere(gpus="2").sbatch_flags()
    assert ampere(gpus="2").ranks == 2
    assert "--gres=gpu:a100:8" in ampere().sbatch_flags()
    for bad in ("", ":4", "a100:", "a100"):
        with pytest.raises(ValueError, match="gpus must be"):
            ampere(gpus=bad)


def test_a_cpu_only_stage_asks_for_no_gpu():
    """A pipeline stage that does not use an accelerator must not hold one. GPUs
    are the scarcest thing on these nodes, so a cpu-only job that keeps one idle
    caps how many of its siblings can run -- which is a throughput bug that
    looks like a scheduling one."""
    flags = ampere(gpus="0").sbatch_flags()
    assert not [f for f in flags if f.startswith("--gres")]
    assert "--ntasks-per-node=1" in flags
    assert ampere(gpus="0").ranks == 1
    with pytest.raises(ValueError, match="no GPUs is"):
        ampere(gpus="a100:0")


def test_one_task_per_gpu():
    assert ampere().ranks == 8
    assert ampere(gpus="b200:4").ranks == 4


def test_ntasks_overrides_one_rank_per_gpu():
    """One rank per GPU is what DDP wants, not a law. A stage that parallelises
    inside one process -- sentence-transformers spawning a worker per device --
    needs every GPU visible to a single rank, and got one GPU each instead."""
    r = ampere(gpus="10", ntasks=1)
    assert r.ranks == 1
    assert "--ntasks-per-node=1" in r.sbatch_flags()
    assert "--gres=gpu:10" in r.sbatch_flags()
    with pytest.raises(ValueError, match="ntasks must be"):
        ampere(ntasks=0)


def test_sbatch_flags():
    flags = ampere().sbatch_flags()
    assert "--ntasks-per-node=8" in flags
    assert "--gres=gpu:a100:8" in flags
    assert "--cpus-per-task=16" in flags
    assert "--exclusive" in flags
    assert not any(f.startswith("--mem") for f in flags)
    assert "--mem=1500000M" in ampere(mem="1500000M").sbatch_flags()


@pytest.mark.parametrize(
    "bad", [{"gpus": "a100"}, {"gpus": "a100:0"}, {"cpus_per_task": 0}]
)
def test_resources_rejects_nonsense(bad):
    with pytest.raises(ValueError):
        ampere(**bad)


def test_every_placeholder_in_the_scripts_is_one_submit_fills():
    """A placeholder nobody fills reaches the compute node as a literal @NAME@,
    and fails there rather than here."""
    import re
    from importlib.resources import files

    used = set()
    for name in ("bootstrap.sh", "env.sh"):
        text = files("roach.slurm").joinpath(name).read_text()
        used |= set(re.findall(r"@[A-Z_]+@", text))
    filled = set(re.findall(r'"(@[A-Z_]+@)"', inspect.getsource(submit_fn)))
    assert used == filled


def test_no_placeholder_sits_inside_a_comment():
    """`setup` is spliced in verbatim, one command per line. A placeholder named
    in a comment gets the same treatment: the first line stays commented out and
    every line after it breaks out and runs as garbage -- which is a two-command
    setup silently corrupted, and a single-command one working fine."""
    import re
    from importlib.resources import files

    for name in ("bootstrap.sh", "env.sh"):
        text = files("roach.slurm").joinpath(name).read_text()
        bad = [
            line
            for line in text.splitlines()
            if line.lstrip().startswith("#") and re.search(r"@[A-Z_]+@", line)
        ]
        assert not bad, f"{name}: {bad}"


def test_the_job_scripts_take_no_configuration_from_the_environment():
    """A job's environment is what submit() put there. A ``${VAR:-default}`` is
    a knob nobody passed, silently answered by whatever the node exported --
    which is how the same submission produces two different runs."""
    import re
    from importlib.resources import files

    # who we are, and what slurm tells the job about itself: not configuration
    runtime = {"USER", "SLURM_RESTART_COUNT"}
    for name in ("bootstrap.sh", "env.sh"):
        text = files("roach.slurm").joinpath(name).read_text()
        read = set(re.findall(r"\$\{([A-Z_]+):-", text))
        assert read <= runtime, f"{name} reads {sorted(read - runtime)} from the env"


def test_job_env_is_not_inherited():
    """--export=ALL is sbatch's default and would copy this shell's home, PATH
    and API tokens into the job (and slurm's job record); the batch script
    carries everything it needs."""
    assert '"--export=NONE"' in inspect.getsource(submit_fn)


def test_bootstrap_lets_srun_inherit_the_job_environment():
    """--export=NONE (which keeps the submit shell out of the job) also stops
    srun from passing the job's own environment to its tasks, so `pixi` is not
    on their PATH; SLURM_EXPORT_ENV=ALL puts it back."""
    from importlib.resources import files

    script = files("roach.slurm").joinpath("bootstrap.sh").read_text()
    assert "srun --export=ALL" in script


def test_presets_are_one_rank_per_gpu():
    """The preset's cpus_per_task is per rank, so a preset that quietly asked
    for a node's worth of cores per rank would be rejected at submit."""
    from roach.slurm import AMPERE, AMPERE_LO, BLACKWELL

    for preset in (AMPERE, AMPERE_LO, BLACKWELL):
        assert preset.ranks == int(preset.gpus.rpartition(":")[2])
        assert preset.ranks * preset.cpus_per_task <= 288  # the widest node here


def test_a_dependent_job_is_cancelled_when_its_dependency_fails():
    """`after` chains a pipeline's stages in one submission pass. Without
    --kill-on-invalid-dep the second stage of a failed first stage sits PENDING
    forever, which looks like a slow queue rather than a failure."""
    src = inspect.getsource(submit_fn)
    assert '"--dependency=afterok:{after}"' in src or "--dependency=afterok:" in src
    assert "--kill-on-invalid-dep=yes" in src
