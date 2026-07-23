"""Shared glue for the legacy architectures: block-mask building, the
Evaluator-facing ``predict`` adapter, and Hub checkpoint loading."""

from __future__ import annotations

import torch
from torch.nn.attention.flex_attention import create_block_mask

# Both legacy papers use the same architecture dims.
LEGACY_MODEL_DIMS = dict(num_blocks=12, d_model=256, d_text=384, num_heads=8, d_ff=1024)
LEGACY_EMBEDDING_MODEL = "all-MiniLM-L12-v2"


def make_block_mask(mask, batch_size, seq_len, device):
    def _mod(b, h, q_idx, kv_idx):
        return mask[b, q_idx, kv_idx]

    return create_block_mask(
        mask_mod=_mod,
        B=batch_size,
        H=None,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=device,
        _compile=True,
    )


def predict(model, batch, eval_ctx_sizes, device, task, bool_as_num):
    """Evaluator-facing eval-time predictions; mirrors
    :meth:`rt.model.net.RelationalTransformer.predict` for the legacy nets
    (which keep token order, so ``is_targets`` needs no re-sort)."""
    val_key = "boolean" if task.task_type == "clf" and not bool_as_num else "number"
    preds = {}
    for ctx_size in eval_ctx_sizes:
        trunc = {
            k: v[:, :ctx_size].to(device, non_blocking=True) for k, v in batch.items()
        }
        _, yhat_dict = model(trunc)
        yhat = yhat_dict[val_key].squeeze(-1)  # (B, S)
        is_targets = trunc["is_targets"]
        preds[ctx_size] = (yhat * is_targets.to(yhat.dtype)).sum(dim=1).cpu()
    return preds


def load_legacy_checkpoint(cls, repo_id: str, filename: str, device: str = "cpu"):
    """Build ``cls`` with the shared legacy dims and load a released ``.pt``
    (flat bf16 state dict) from the Hub or a local path."""
    from pathlib import Path

    p = Path(filename).expanduser()
    if not p.is_file():
        from huggingface_hub import hf_hub_download

        p = Path(hf_hub_download(repo_id, filename))
    state_dict = torch.load(p, map_location="cpu", weights_only=True)
    if "model" in state_dict and isinstance(state_dict["model"], dict):
        state_dict = state_dict["model"]
    model = cls(**LEGACY_MODEL_DIMS)
    model.load_state_dict(state_dict)
    return model.to(device)
