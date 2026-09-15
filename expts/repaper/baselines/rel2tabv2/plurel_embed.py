from functools import partial

from rt.model.legacy._common import (
    LEGACY_EMBEDDER,
    LEGACY_MODEL_DIMS,
    make_block_mask,
)
from rt.model.legacy.plurel import (
    PLUREL_HUB_REPO,
    PLUREL_SYNTH_CKPT,
    PluRelTransformer,
)

SEM_TYPE_NAMES = ["number", "text", "datetime", "boolean"]

__all__ = [
    "LEGACY_EMBEDDER",
    "LEGACY_MODEL_DIMS",
    "PLUREL_HUB_REPO",
    "PLUREL_SYNTH_CKPT",
    "PluRelEmbedder",
]


class PluRelEmbedder(PluRelTransformer):
    def embed(self, batch):
        batch = {**batch, "boolean_values": batch["number_values"]}
        node_idxs = batch["node_idxs"]
        f2p_nbr_idxs = batch["f2p_nbr_idxs"]
        col_name_idxs = batch["col_name_idxs"]
        table_name_idxs = batch["table_name_idxs"]
        is_padding = batch["is_padding"]
        is_targets = batch["is_targets"]

        batch_size, seq_len = node_idxs.shape
        device = node_idxs.device

        pad = (~is_padding[:, :, None]) & (~is_padding[:, None, :])
        same_node = node_idxs[:, :, None] == node_idxs[:, None, :]
        kv_in_f2p = (node_idxs[:, None, :, None] == f2p_nbr_idxs[:, :, None, :]).any(-1)
        q_in_f2p = (node_idxs[:, :, None, None] == f2p_nbr_idxs[:, None, :, :]).any(-1)
        same_col_table = (col_name_idxs[:, :, None] == col_name_idxs[:, None, :]) & (
            table_name_idxs[:, :, None] == table_name_idxs[:, None, :]
        )

        attn_masks = {
            "feat": (same_node | kv_in_f2p) & pad,
            "nbr": q_in_f2p & pad,
            "col": same_col_table & pad,
        }
        for lvl in attn_masks:
            attn_masks[lvl] = attn_masks[lvl].contiguous()

        mbm = partial(
            make_block_mask, batch_size=batch_size, seq_len=seq_len, device=device
        )
        block_masks = {lvl: mbm(attn_mask) for lvl, attn_mask in attn_masks.items()}

        x = 0
        x = x + (
            self.norm_dict["col_name"](
                self.enc_dict["col_name"](batch["col_name_values"])
            )
            * (~is_padding)[..., None]
        )
        for i, t in enumerate(SEM_TYPE_NAMES):
            x = x + (
                self.norm_dict[t](self.enc_dict[t](batch[t + "_values"]))
                * ((batch["sem_types"] == i) & ~is_targets & ~is_padding)[..., None]
            )
            x = x + (
                self.mask_embs[t]
                * ((batch["sem_types"] == i) & is_targets & ~is_padding)[..., None]
            )

        for block in self.blocks:
            x = block(x, block_masks)

        return self.norm_out(x)
