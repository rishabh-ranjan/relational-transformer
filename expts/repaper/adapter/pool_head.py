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


TARGET_KEY = -1
PAD_KEY = -2


def cell_keys(col_name_idxs, table_name_idxs, is_targets, is_padding):
    key = col_name_idxs.long() * (1 << 32) + table_name_idxs.long()
    key = key.masked_fill(is_targets.bool(), TARGET_KEY)
    return key.masked_fill(is_padding.bool(), PAD_KEY)


def column_stats(tokens, keys, n_rows, n_ctx, chunk, eps=1e-8):
    flat = keys[:n_rows].reshape(-1)
    uniq, inv = torch.unique(flat, return_inverse=True)
    idx = inv.reshape(n_rows, -1)
    d = tokens.shape[-1]
    s1 = torch.zeros(len(uniq), d, dtype=torch.float64, device=tokens.device)
    s2 = torch.zeros_like(s1)
    cnt = torch.zeros(len(uniq), dtype=torch.float64, device=tokens.device)
    live = (keys[:n_ctx] != PAD_KEY)
    for i in range(0, n_ctx, chunk):
        j = min(i + chunk, n_ctx)
        m = live[i:j].reshape(-1)
        x = tokens[i:j].reshape(-1, d)[m].double()
        k = idx[i:j].reshape(-1)[m]
        s1.index_add_(0, k, x)
        s2.index_add_(0, k, x * x)
        cnt.index_add_(0, k, torch.ones_like(k, dtype=torch.float64))
    seen = cnt > 1
    mean = torch.where(seen[:, None], s1 / cnt.clamp_min(1)[:, None], torch.zeros_like(s1))
    var = torch.where(seen[:, None], s2 / cnt.clamp_min(1)[:, None] - mean * mean, torch.ones_like(s1))
    std = torch.sqrt(var.clamp_min(0) + eps)
    n_query_only = int((~seen & (uniq != PAD_KEY)).sum())
    return idx, mean.float(), std.float(), n_query_only


class PoolHead(nn.Module):
    def __init__(
        self, d_model: int, n_queries: int, swiglu_norm: str, mean=None, scale=None,
        input_norm: str = "fixed", signsoftmax_temp: float = 1.0,
        colnorm_tau: float | None = None, live_scale: bool = False, signsoftmax_scale_by_dim: bool = False,
    ):
        super().__init__()
        assert input_norm in ("fixed", "col_context", "col_context_signsoftmax"), input_norm
        assert colnorm_tau is None or input_norm != "fixed", (colnorm_tau, input_norm)
        self.input_norm = input_norm
        self.signsoftmax_temp = signsoftmax_temp
        self.colnorm_tau = colnorm_tau
        self.live_scale = live_scale
        assert not signsoftmax_scale_by_dim or input_norm == "col_context_signsoftmax", (signsoftmax_scale_by_dim, input_norm)
        self.signsoftmax_scale_by_dim = signsoftmax_scale_by_dim
        self.register_buffer(
            "mean", torch.zeros(d_model) if mean is None else torch.as_tensor(mean).float()
        )
        self.register_buffer(
            "scale", torch.ones(d_model) if scale is None else torch.as_tensor(scale).float()
        )
        d_in = d_model if input_norm == "fixed" else 2 * d_model
        if input_norm == "col_context":
            self.rms = nn.RMSNorm(d_model)
        self.q = nn.Parameter(torch.randn(n_queries, d_model) / math.sqrt(d_model))
        self.wk = nn.Linear(d_in, d_model, bias=False)
        self.wv = nn.Linear(d_in, n_queries, bias=False)
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

    def pool(self, x: torch.Tensor, is_padding: torch.Tensor, colnorm=None) -> torch.Tensor:
        if self.input_norm == "fixed":
            z = (x.float() - self.mean) / self.scale
        else:
            idx, col_mean, col_std = colnorm
            xf = x.float()
            if self.input_norm == "col_context":
                second = self.rms(xf)
            else:
                rms = xf.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-12)
                second = xf.sign() * torch.softmax(xf.abs() / (self.signsoftmax_temp * rms), dim=-1)
                if self.signsoftmax_scale_by_dim:
                    second = second * xf.shape[-1]
            first = (xf - col_mean[idx]) / col_std[idx]
            if self.colnorm_tau is not None:
                first = self.colnorm_tau * torch.tanh(first / self.colnorm_tau)
            z = torch.cat([first, second], dim=-1)
        qk = self.wk.weight.t() @ self.q.t()
        logits = (z @ qk) / math.sqrt(self.d_model)
        logits = logits.masked_fill(is_padding[..., None], float("-inf"))
        attn = logits.softmax(dim=1)
        v = self.wv(z)
        h = (attn * v).sum(dim=1)
        if self.live_scale:
            n_live = (~is_padding).sum(dim=1, keepdim=True).to(h.dtype)
            h = h * torch.sqrt(n_live / is_padding.shape[1])
        return h

    def mix(self, h: torch.Tensor, n_ctx: int | None) -> torch.Tensor:
        if self.swiglu_norm == "context":
            assert n_ctx is not None and 1 < n_ctx <= len(h), n_ctx
            u = self.norm(h, n_ctx)
        else:
            u = self.norm(h)
        return h + self.w2(F.silu(self.w1(u)) * self.w3(u))

    def forward(self, x: torch.Tensor, is_padding: torch.Tensor, n_ctx: int | None = None, colnorm=None) -> torch.Tensor:
        return self.mix(self.pool(x, is_padding, colnorm), n_ctx)
