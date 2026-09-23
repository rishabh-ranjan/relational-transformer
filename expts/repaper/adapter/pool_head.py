import math

import torch
from torch import nn
from torch.nn import functional as F


class ContextNorm(nn.Module):
    def __init__(self, n_features: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_features))
        self.bias = nn.Parameter(torch.zeros(n_features))
        self.eps = eps

    def forward(self, h: torch.Tensor, n_ctx: int) -> torch.Tensor:
        ctx = h[:n_ctx]
        mean = ctx.mean(dim=0)
        var = ctx.var(dim=0, unbiased=False)
        return (h - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias


class PoolHead(nn.Module):
    def __init__(self, d_model: int, n_queries: int, swiglu_norm: str, mean=None, scale=None):
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
        self.norm = {
            "none": nn.Identity,
            "layer": lambda: nn.LayerNorm(n_queries),
            "context": lambda: ContextNorm(n_queries),
        }[swiglu_norm]()
        self.swiglu_norm = swiglu_norm
        for lin in (self.wk, self.wv, self.w1, self.w3):
            nn.init.xavier_uniform_(lin.weight)
        nn.init.zeros_(self.w1.bias)
        nn.init.zeros_(self.w3.bias)
        nn.init.zeros_(self.w2.weight)
        self.d_model = d_model

    def pool(self, x: torch.Tensor, is_padding: torch.Tensor) -> torch.Tensor:
        z = (x.float() - self.mean) / self.scale
        qk = self.wk.weight.t() @ self.q.t()
        logits = (z @ qk) / math.sqrt(self.d_model)
        logits = logits.masked_fill(is_padding[..., None], float("-inf"))
        attn = logits.softmax(dim=1)
        v = self.wv(z)
        return (attn * v).sum(dim=1)

    def mix(self, h: torch.Tensor, n_ctx: int | None) -> torch.Tensor:
        if self.swiglu_norm == "context":
            assert n_ctx is not None and 1 < n_ctx <= len(h), n_ctx
            u = self.norm(h, n_ctx)
        else:
            u = self.norm(h)
        return h + self.w2(F.silu(self.w1(u)) * self.w3(u))

    def forward(self, x: torch.Tensor, is_padding: torch.Tensor, n_ctx: int | None = None) -> torch.Tensor:
        return self.mix(self.pool(x, is_padding), n_ctx)
