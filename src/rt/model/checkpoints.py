"""Checkpoint IO: save/load model state (local file/dir or Hub repo)."""

import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from rt.data import resolve_repo
from safetensors.torch import save_file
from safetensors.torch import load_file
from huggingface_hub import snapshot_download

try:
    _RT_VERSION = version("relational-transformer")
except PackageNotFoundError:  # running from a source tree without an install
    _RT_VERSION = None

# huggingface_hub user-agent so Hub downloads are attributed to this library.
_HF_UA = {"library_name": "relational-transformer", "library_version": _RT_VERSION}

CONFIG_FILE = "config.json"
MODEL_FILE = "model.safetensors"
MODEL_DIM_KEYS = ("num_blocks", "d_model", "d_text", "num_heads", "d_ff")


def save_model(state_dict, path, metadata: dict | None = None) -> None:
    """Save a flat tensor ``state_dict`` to ``path`` as safetensors.

    ``metadata`` (e.g. ``{"step": 1000}``) is coerced to a str→str header, as
    safetensors metadata only holds strings.
    """

    meta = {str(k): str(v) for k, v in (metadata or {}).items()}
    save_file(state_dict, str(path), metadata=meta or None)


def load_model(path):
    """Load a flat tensor ``state_dict`` from a ``.safetensors`` checkpoint."""

    return load_file(str(path))


def _compat(config: dict) -> None:
    """In-place rename of legacy config.json keys (checkpoints written before
    ``embedding_model`` became ``embedder``)."""
    if "embedder" not in config and "embedding_model" in config:
        config["embedder"] = config.pop("embedding_model")


def resolve_checkpoint(
    spec, *, revision: str | None = None, subfolder: str | None = None
) -> tuple[dict, Path]:
    """Return ``(config, model_path)`` for a local or Hub checkpoint.

    ``spec`` may be: a local weights file (``model.safetensors``; config from a
    sibling ``config.json`` if present), a local directory, or a Hub
    ``org/repo[/subdir]``. ``subfolder`` selects a sub-directory within the
    repo/directory (the HuggingFace-idiomatic way to pick a checkpoint;
    equivalent to appending it to ``spec``). Within a directory, an explicit
    ``config["checkpoint_file"]`` wins, else ``model.safetensors``.
    """
    p = Path(spec).expanduser()
    if p.is_file():
        cfg_path = p.with_name(CONFIG_FILE)
        config = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        _compat(config)
        return config, p
    if p.is_dir():
        d = p / subfolder if subfolder else p
    else:
        repo_id, subdir = resolve_repo(spec)
        subdir = "/".join(part for part in (subdir, subfolder) if part)
        local = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=[f"{subdir}/*"] if subdir else None,
            **_HF_UA,
        )
        d = Path(local) / subdir if subdir else Path(local)
    config = json.loads((d / CONFIG_FILE).read_text())
    _compat(config)
    return config, d / config.get("checkpoint_file", MODEL_FILE)
