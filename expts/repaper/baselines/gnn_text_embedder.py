import math
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from torch import Tensor

HF_REPO = "sentence-transformers/average_word_embeddings_glove.6B.300d"


class GloveTextEmbedding:
    def __init__(self, model_path: str | Path, device: str | torch.device | None):
        path = Path(model_path).expanduser()
        assert (path / "modules.json").is_file(), (
            f"GloVe embedder {path} not found; fetch it once with "
            f"`pixi run -e gnn python -m expts.repaper.baselines.fetch_glove`"
        )
        self.model = SentenceTransformer(str(path), device=device)

    def __call__(self, sentences: list[str]) -> Tensor:
        cleaned = []
        for s in sentences:
            if s is None or (isinstance(s, float) and math.isnan(s)):
                cleaned.append("")
            elif isinstance(s, str):
                cleaned.append(s)
            else:
                cleaned.append(str(s))
        return self.model.encode(cleaned, convert_to_tensor=True)
