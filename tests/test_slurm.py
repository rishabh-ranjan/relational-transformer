"""rt.slurm: the parts that are pure functions, and so are the parts that bit us.

Every check here corresponds to a failure that cost real time on the cluster:
arguments that do not match the target, a resource shape that slurm rejects, and
a batch script that has lost a placeholder.
"""

from __future__ import annotations

import pytest

from rt.slurm import Resources, check_args, resolve, timestamp


def sample(a: int, b: str, c: list[int], run_id: str) -> None:  # noqa: ARG001
    pass


def test_resolve_requires_module_attr():
    assert resolve("rt.slurm.submit:timestamp") is timestamp
    with pytest.raises(ValueError):
        resolve("rt.slurm.submit")


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
        exclusive=True,
        mem=None,
        constraint="ampere",
        nodelist=None,
    )
    return Resources(**{**kwargs, **over})


def test_one_task_per_gpu():
    assert ampere().ntasks == 8
    assert ampere(gpus="b200:4").ntasks == 4


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


def test_bootstrap_template_placeholders_are_all_filled():
    from importlib.resources import files

    script = files("rt.slurm").joinpath("bootstrap.sh").read_text()
    expected = {
        "@REPO@",
        "@COMMIT@",
        "@RUN_ID@",
        "@NAME@",
        "@TARGET@",
        "@ARGS@",
        "@LOG_ROOT@",
        "@CLONE_ROOT@",
        "@SECRETS_DIR@",
    }
    import re

    assert set(re.findall(r"@[A-Z_]+@", script)) == expected


def test_job_env_is_not_inherited():
    """--export=ALL is sbatch's default and would copy this shell's home, PATH
    and API tokens into the job (and slurm's job record); the batch script
    carries everything it needs."""
    import inspect as _inspect

    from rt.slurm import submit as submit_mod

    assert '"--export=NONE"' in _inspect.getsource(submit_mod.submit)
