"""Text embeddings for preprocessed datasets (sentence-transformers)."""

from __future__ import annotations

import numpy as np
import orjson
import torch
from ml_dtypes import bfloat16
from sentence_transformers import SentenceTransformer


# Rows encoded before the result is moved off the device. Bounds GPU memory at
# roughly CHUNK x d_text x 4 bytes, independent of how much text a database has.
CHUNK = 1_000_000


class TextEmbedder:
    def __init__(self, batch_size, embedder, device, chunk=CHUNK):
        device_type = torch.device(device).type
        self.model = SentenceTransformer(
            f"sentence-transformers/{embedder}",
            device=device,
            model_kwargs={
                "dtype": torch.bfloat16 if device_type == "cuda" else torch.float32,
            },
        )
        self.batch_size = batch_size
        self.chunk = chunk

    def __call__(self, text_list, device):
        if isinstance(device, list):
            # Multi-process path returns fp32 numpy regardless of flags.
            emb = self.model.encode(
                text_list,
                batch_size=self.batch_size,
                show_progress_bar=True,
                device=device,
            )
            return emb.astype(bfloat16)
        # In chunks, each moved off the device before the next is encoded.
        # `convert_to_tensor` keeps every embedding on the GPU until the whole
        # list is done, so a database with a lot of text needs the entire output
        # resident at once -- join-inaturalist's 698 MB of text wanted ~21 GiB
        # and died on a 12 GiB card, an hour of rustler work already behind it.
        # Chunked, the ceiling is CHUNK rows regardless of the database.
        out = []
        for i in range(0, len(text_list), self.chunk):
            emb = self.model.encode(
                text_list[i : i + self.chunk],
                batch_size=self.batch_size,
                convert_to_numpy=False,
                convert_to_tensor=True,
                show_progress_bar=True,
                device=device,
            )
            # bf16 → int16 bitcast so torch .numpy() accepts it, then relabel as
            # bf16. On CPU the SBERT model loaded with fp32 (line 15), so cast
            # first — the bitcast on raw fp32 silently misinterprets 4-byte
            # floats as 2×bf16 garbage and writes a .bin with NaN/inf patterns.
            out.append(emb.to(torch.bfloat16).cpu().view(torch.int16).numpy())
            del emb
            if torch.device(device).type == "cuda":
                torch.cuda.empty_cache()
        if not out:
            return np.empty((0, 0), dtype=bfloat16)
        return np.concatenate(out).view(bfloat16)


def embed_texts(
    dataset_name,
    pre_dir: str,
    device,
    batch_size,
    embedder,
):
    if device is None:
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            # Pass a string for 1 GPU. A list of len 1 routes SBERT into its
            # multi-process path, which skips length-sorted batching.
            device = [f"cuda:{i}" for i in range(n)] if n > 1 else "cuda:0"
            print(f"Using device(s): {device}")
        else:
            device = "cpu"

    init_device = device[0] if isinstance(device, list) else device

    text_path = f"{pre_dir}/{dataset_name}/text.json"
    # Read as bytes and drop them the moment they are parsed. Holding `raw`
    # alongside the parsed list doubles the peak for no reason, and these files
    # are not small -- join-overture-maps' text.json is 8 GiB, which is enough
    # for the difference to be an out-of-memory kill rather than a slow moment.
    with open(text_path, "rb") as f:
        raw = f.read()
    text_list = orjson.loads(raw)
    del raw
    print(f"Loaded {len(text_list)} texts from {text_path}")

    text_embedder = TextEmbedder(batch_size, embedder, init_device)
    emb = text_embedder(text_list, device=device)

    emb_path = f"{pre_dir}/{dataset_name}/text_emb_{embedder}.bin"
    emb.tofile(emb_path)
    print(f"Wrote {emb.shape} {emb.dtype} to {emb_path}")
