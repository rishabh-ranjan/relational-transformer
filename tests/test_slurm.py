"""roach.slurm: the parts that are pure functions, and so are the parts that bit us.

Every check here corresponds to a failure that cost real time on the cluster:
arguments that do not match the target, a resource shape that slurm rejects, and
a batch script that has lost a placeholder.
"""

import inspect

import pytest

from roach.slurm import Resources, check_args, resolve, timestamp
from roach.slurm._submit import _overlap, launch
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
        mem_per_gpu=None,
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
    # Subset, not equality: submit() also fills placeholders that only reach the
    # script through another one (@TARGET@ and @ARGS@ ride inside @LAUNCH@).
    assert used <= filled


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


def test_a_call_that_crosses_from_the_submitting_shell_exports_almost_nothing():
    """One rule, two layers: nothing from the submitting shell, everything from
    the job's own environment. sbatch says NONE; the driver step of an
    overlapping run cannot, because srun execve's a step with exactly what you
    exported and NONE could not even find bash -- so it names the two variables
    that get bash started, and nothing that came from this shell."""
    src = inspect.getsource(_overlap)
    assert "--export=ALL" not in src
    assert "--export=PATH=" in src and "USER=" in src


def test_the_launcher_lets_srun_inherit_the_job_environment():
    """--export=NONE (which keeps the submit shell out of the job) also stops
    srun from passing the job's own environment to its tasks, so `pixi` is not
    on their PATH; SLURM_EXPORT_ENV=ALL puts it back."""
    for overlap in (None, "12345"):
        line = launch(ampere(), "pkg:main", "/args.json", overlap)
        assert "--export=ALL" in line


def test_an_overlapping_run_states_its_shape_in_full():
    """The driver script runs as a one-task step of somebody else's allocation,
    so srun inherits *that* shape unless the ranks' own is spelled out -- one
    task on one cpu with no gpu, which is not a training job. --overlap is what
    keeps the two steps from waiting on each other."""
    line = launch(ampere(gpus="b200:4", cpus_per_task=36), "p:m", "/a.json", "999")
    for flag in (
        "--jobid=999",
        "--overlap",
        "--ntasks=4",
        "--cpus-per-task=36",
        # every rank sees every card, so LOCAL_RANK indexes them; --gpus-per-task
        # would give each rank one card as device 0 and rank 1 an ordinal that
        # does not exist
        "--gres=gpu:4",
    ):
        assert flag in line
    assert "--jobid" not in launch(ampere(), "p:m", "/a.json", None)


def test_a_cpu_only_overlapping_run_asks_for_no_gpu():
    assert "--gres" not in launch(ampere(gpus="0"), "p:m", "/a.json", "1")


def test_presets_are_one_rank_per_gpu():
    """The preset's cpus_per_task is per rank, so a preset that quietly asked
    for a node's worth of cores per rank would be rejected at submit."""
    from roach.slurm import (
        AMPERE,
        AMPERE_LO,
        BLACKWELL,
        BLACKWELL_INTERACTIVE,
        BLACKWELL_INTERACTIVE_1GPU,
    )

    for preset in (
        AMPERE,
        AMPERE_LO,
        BLACKWELL,
        BLACKWELL_INTERACTIVE,
        BLACKWELL_INTERACTIVE_1GPU,
    ):
        assert preset.ranks == int(preset.gpus.rpartition(":")[2])
        assert preset.ranks * preset.cpus_per_task <= 288  # the widest node here


def test_a_dependent_job_is_cancelled_when_its_dependency_fails():
    """`after` chains a pipeline's stages in one submission pass. Without
    --kill-on-invalid-dep the second stage of a failed first stage sits PENDING
    forever, which looks like a slow queue rather than a failure."""
    src = inspect.getsource(submit_fn)
    assert '"--dependency=afterok:{after}"' in src or "--dependency=afterok:" in src
    assert "--kill-on-invalid-dep=yes" in src


def test_mem_per_gpu_is_how_a_job_gets_a_whole_node_of_gpus():
    """A partition with DefMemPerGPU applies it when deciding whether a job
    fits, and --mem does not displace it: the most GPUs a job can hold becomes
    RealMemory / DefMemPerGPU however little memory it wants. Here that is 3
    GPUs on a 770G node. --mem-per-gpu replaces the default and lifts it."""
    flags = ampere(gpus="8", mem=None, mem_per_gpu="20G").sbatch_flags()
    assert "--mem-per-gpu=20G" in flags
    assert not [f for f in flags if f.startswith("--mem=")]
    with pytest.raises(ValueError, match="not both"):
        ampere(mem="10G", mem_per_gpu="10G")
