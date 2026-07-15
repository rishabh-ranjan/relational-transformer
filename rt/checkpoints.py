"""Load RT checkpoints from a local path *or* the HuggingFace Hub — one interface.

A released checkpoint is a directory (a local folder or a Hub model repo
``org/repo[/subdir]``) holding:

* ``model.safetensors`` -- the weights, a flat ``state_dict`` stored with
  :func:`safetensors.torch.save_file` (the format ``scripts/pretrain.py`` writes
  and ``scripts/release_checkpoints.py`` packages); and
* ``config.json``       -- the model dims + the text-embedding model used, so a
  loader needs no out-of-band knowledge.

For backward compatibility, legacy ``model.pt`` checkpoints (a
``{"model": state_dict}`` pickle, the older release/training format) still load:
if no ``.safetensors`` is found, or ``config["checkpoint_file"]`` ends in
``.pt``, the weights are read with ``torch.load(...)["model"]``.

This mirrors :mod:`rt.pre`'s local-or-Hub *data* resolver, so a checkpoint
reference like ``stanford-star/rt-j/classification`` "just works" (downloaded and cached
on demand), while a local training run's checkpoint dir works with the same call.
A local path always wins, so iterating locally never triggers a download.
"""

from __future__ import annotations

import os
import json
from pathlib import Path

from rt.pre import resolve_repo

CONFIG_FILE = "config.json"
MODEL_FILE = "model.safetensors"
LEGACY_MODEL_FILE = "model.pt"


def save_model(state_dict, path, metadata: dict | None = None) -> None:
    """Save a flat tensor ``state_dict`` to ``path`` as safetensors.

    ``metadata`` (e.g. ``{"step": 1000}``) is coerced to a str→str header, as
    safetensors metadata only holds strings.
    """
    from safetensors.torch import save_file

    meta = {str(k): str(v) for k, v in (metadata or {}).items()}
    save_file(state_dict, str(path), metadata=meta or None)


def load_model(path):
    """Load a flat tensor ``state_dict`` from a ``.safetensors`` (or legacy
    ``.pt``) checkpoint at ``path``."""
    path = Path(path)
    if path.suffix == ".pt":
        import torch

        ckpt = torch.load(path, map_location="cpu")
        return ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    from safetensors.torch import load_file

    return load_file(str(path))


def resolve_checkpoint(spec, *, revision: str | None = None) -> tuple[dict, Path]:
    """Return ``(config, model_path)`` for a local or Hub checkpoint.

    ``spec`` may be: a local weights file (``model.safetensors`` or legacy
    ``model.pt``; config from a sibling ``config.json`` if present), a local
    directory, or a Hub ``org/repo[/subdir]``. Within a directory, an explicit
    ``config["checkpoint_file"]`` wins, else ``model.safetensors`` is preferred
    over a legacy ``model.pt``.
    """
    p = Path(spec).expanduser()
    if p.is_file():
        cfg_path = p.with_name(CONFIG_FILE)
        config = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        return config, p
    if p.is_dir():
        d = p
    else:
        from huggingface_hub import snapshot_download

        repo_id, subdir = resolve_repo(spec)
        local = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=[f"{subdir}/*"] if subdir else None,
        )
        d = Path(local) / subdir if subdir else Path(local)
    config = json.loads((d / CONFIG_FILE).read_text())
    fname = config.get("checkpoint_file")
    if fname is None:
        # Prefer safetensors, fall back to a legacy .pt if that is all there is.
        fname = MODEL_FILE if (d / MODEL_FILE).exists() else LEGACY_MODEL_FILE
    return config, d / fname


def load_rt_model(
    spec,
    *,
    device: str = "cpu",
    compile: bool = False,
    revision: str | None = None,
    model_kwargs: dict | None = None,
):
    """Resolve a checkpoint, build the RelationalTransformer, load its weights.

    Returns ``(model, config)``. ``model_kwargs`` overrides/fills model dims when
    a checkpoint has no ``config.json`` (e.g. a raw internal ckpt during dev).
    Loads safetensors when present and falls back to legacy ``.pt`` pickles.
    """
    from rt.model import RelationalTransformer

    config, model_path = resolve_checkpoint(spec, revision=revision)
    m = {**config.get("model", {}), **(model_kwargs or {})}
    missing = [k for k in ("num_blocks", "d_model", "d_text", "num_heads", "d_ff") if k not in m]
    if missing:
        raise ValueError(
            f"checkpoint {spec!r} is missing model dims {missing}; provide a "
            f"config.json or pass model_kwargs."
        )
    net = RelationalTransformer(
        num_blocks=m["num_blocks"],
        d_model=m["d_model"],
        d_text=m["d_text"],
        num_heads=m["num_heads"],
        d_ff=m["d_ff"],
        compile=compile,
        # materialized masks are O(ctx^2) memory; RT_MATERIALIZE_ATTN_MASKS=0
        # forces the flex-attention path for long-ctx (>=16k) inference.
        materialize_attn_masks=(
            False if os.environ.get("RT_MATERIALIZE_ATTN_MASKS", "") == "0"
            else m.get("materialize_attn_masks", True)
        ),
    )
    net.load_state_dict(load_model(model_path))
    return net.to(device), config
