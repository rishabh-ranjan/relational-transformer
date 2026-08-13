"""The released wheel must not fall behind the source it is built from.

`src/rt` and `rustler/` ship inside a prebuilt `abi3` wheel attached to a GitHub
release, and downstreams install *that*, not this checkout -- relarena's `rt`
extra pins its URL. So a change here is invisible to every consumer until a new
wheel is built and released.

That failure mode is silent and it has already happened once: v1.1.0 shipped
before `db_cutoff` learned to accept an explicit timestamp, relarena's model
passed one, and every one of our own jobs was fine because they ran the
checkout. The consumer path was broken for a day and nothing said so.

These tests are the thing that says so.
"""

from __future__ import annotations

import re
import subprocess
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
#: What ships in the wheel. `expts/`, `docs/` and `tests/` do not, so they are
#: free to change without a release.
PACKAGED = ("src/", "rustler/")


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def _latest_release_tag() -> str | None:
    tags = [t for t in _git("tag", "-l", "v*").splitlines() if t]
    if not tags:
        return None
    # Sort by version, not lexically: v1.10.0 must beat v1.9.0.
    def key(t: str) -> tuple[int, ...]:
        return tuple(int(p) for p in re.findall(r"\d+", t))
    return max(tags, key=key)


@pytest.mark.skipif(
    not (ROOT / ".git").exists(), reason="not a git checkout (installed copy)"
)
def test__packaged_source__has_not_changed_since_the_last_release() -> None:
    tag = _latest_release_tag()
    assert tag is not None, "no v* tag: cut a release before relying on the wheel"
    changed = _git("log", "--oneline", f"{tag}..HEAD", "--", *PACKAGED)
    assert not changed, (
        f"{PACKAGED} changed since {tag}, so the released wheel is stale and "
        f"every consumer installing it gets the old code:\n{changed}\n\n"
        "Bump the version in pyproject.toml and rustler/Cargo.toml, rebuild "
        "against the *floor* interpreter\n"
        "  pixi run maturin build --release --interpreter <python3.11> \\\n"
        "      --out dist --compatibility manylinux_2_28\n"
        "then tag, `gh release create`, and repoint relarena's `rt` extra."
    )


@pytest.mark.skipif(
    not (ROOT / ".git").exists(), reason="not a git checkout (installed copy)"
)
def test__version__matches_the_latest_release_tag() -> None:
    tag = _latest_release_tag()
    assert tag is not None
    declared = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    assert f"v{declared}" == tag, (
        f"pyproject version {declared!r} and latest tag {tag!r} disagree. A "
        "released wheel is named by the version, so these drifting apart means "
        "the URL a consumer pins does not name the code they get."
    )


@pytest.mark.skipif(
    not (ROOT / ".git").exists(), reason="not a git checkout (installed copy)"
)
def test__rust_and_python_versions_agree() -> None:
    py = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    rs = tomllib.loads((ROOT / "rustler" / "Cargo.toml").read_text())["package"]["version"]
    assert py == rs, (
        f"pyproject {py!r} != rustler/Cargo.toml {rs!r}; the wheel takes its "
        "name from the former and its extension from the latter."
    )


def test__abi3_floor__matches_requires_python() -> None:
    # The wheel is tagged cp311-abi3 and must therefore be *built* against the
    # floor: an abi3 build on a newer interpreter is only as safe as pyo3's
    # symbol discipline, and the floor is where that assumption is free.
    cargo = (ROOT / "rustler" / "Cargo.toml").read_text()
    m = re.search(r'abi3-py(\d)(\d+)', cargo)
    assert m, "rustler/Cargo.toml declares no abi3-pyXY feature"
    floor = f"{m.group(1)}.{int(m.group(2))}"
    requires = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
        "requires-python"
    ]
    assert floor in requires, (
        f"abi3 floor is {floor} but requires-python is {requires!r}; the wheel "
        "would claim support for an interpreter its ABI does not cover."
    )
