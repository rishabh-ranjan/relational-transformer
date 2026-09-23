import math

import torch
from torch import nn
from torch.nn import functional as F


class PoolHead(nn.Module):
    def __init__(self, d_model: int, n_queries: int, mean=None, scale=None):
        super().__init__()
        self.register_buffer(
            "mean", torch.zeros(d_model) if mean is None else torch.as_tensor(mean).float()
        )
        self.register_buffer(
            "scale", torch.ones(d_model) if scale is None else torch.as_tensor(scale).float()
        )
        self.q = nn.Parameter(torch.randn(n_queries, d_model) / math.sqrt(d_model))
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, n_queries, bias=False)
        self.w1 = nn.Linear(n_queries, n_queries, bias=True)
        self.w3 = nn.Linear(n_queries, n_queries, bias=True)
        self.w2 = nn.Linear(n_queries, n_queries, bias=False)
        for lin in (self.wk, self.wv, self.w1, self.w3):
            nn.init.xavier_uniform_(lin.weight)
        nn.init.zeros_(self.w1.bias)
        nn.init.zeros_(self.w3.bias)
        nn.init.zeros_(self.w2.weight)
        self.d_model = d_model

    def forward(self, x: torch.Tensor, is_padding: torch.Tensor) -> torch.Tensor:
        z = (x.float() - self.mean) / self.scale
        qk = self.wk.weight.t() @ self.q.t()
        logits = (z @ qk) / math.sqrt(self.d_model)
        logits = logits.masked_fill(is_padding[..., None], float("-inf"))
        attn = logits.softmax(dim=1)
        v = self.wv(z)
        h = (attn * v).sum(dim=1)
        return h + self.w2(F.silu(self.w1(h)) * self.w3(h))
