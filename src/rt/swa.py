"""SWA (stochastic weight averaging) state.

``SwaState`` is a small parameter-averaging container used by
``rt.pretrain`` for in-loop SWA over training params. Backed by an fp32
dict; updates are in-place ``lerp_`` with alpha derived from ``momentum`` and
the current update count.
"""

import torch


class SwaState:
    """Equal-weight or EMA running average of a fixed set of named tensors.

    Backed by an fp32 dict of clones; updates are in-place via
    ``lerp_``. The averaging weight schedule depends on ``momentum``:

    - ``1.0``: equal-weight averaging (``alpha = 1/n``). After ``k``
      updates, ``params[name]`` equals the arithmetic mean of the
      ``k`` inputs.
    - ``< 1.0``: bias-corrected EMA (``alpha = (1-m) / (1-m^n)``).
      First update has ``alpha=1.0``, asymptotes to ``1-m`` as ``n``
      grows.

    Used from training (``rt.pretrain``: in-loop SWA over ``raw_net``
    parameters).
    """

    def __init__(self, named_tensors, momentum):
        """``named_tensors``: iterable of ``(name, tensor)`` pairs. The
        fp32 storage is allocated as clones on the source tensors'
        devices. Initial values are arbitrary — the first ``update``
        sets them exactly (``alpha=1.0``)."""
        self.momentum = momentum
        self.params = {name: t.detach().float().clone() for name, t in named_tensors}
        self.n = 0

    @torch.no_grad()
    def update(self, named_tensors):
        """Add one snapshot to the running average. Source key set must
        equal the stored key set."""
        self.n += 1
        if self.momentum == 1.0:
            alpha = 1.0 / self.n
        else:
            m = self.momentum
            alpha = (1.0 - m) / (1.0 - m**self.n)
        src = dict(named_tensors)
        assert src.keys() == self.params.keys(), (
            f"key mismatch:"
            f" extra={sorted(set(src) - set(self.params))}"
            f" missing={sorted(set(self.params) - set(src))}"
        )
        for name, target in self.params.items():
            target.lerp_(src[name].float(), alpha)

    def state_dict(self):
        """CPU-serializable snapshot for training resume."""
        return {
            "momentum": self.momentum,
            "n": self.n,
            "params": {k: v.detach().cpu().clone() for k, v in self.params.items()},
        }

    @torch.no_grad()
    def load_state_dict(self, state):
        assert state["momentum"] == self.momentum, (
            f"momentum mismatch: ckpt={state['momentum']} cfg={self.momentum}"
        )
        assert state["params"].keys() == self.params.keys(), (
            f"key mismatch:"
            f" extra={sorted(set(state['params']) - set(self.params))}"
            f" missing={sorted(set(self.params) - set(state['params']))}"
        )
        self.n = state["n"]
        for k, v in self.params.items():
            v.copy_(state["params"][k].to(v.device))

    @torch.no_grad()
    def sync_to(self, named_tensors):
        """Copy the running average into the target tensors in-place.
        Target key set must equal the stored key set."""
        dst = dict(named_tensors)
        assert dst.keys() == self.params.keys(), (
            f"key mismatch:"
            f" extra={sorted(set(dst) - set(self.params))}"
            f" missing={sorted(set(self.params) - set(dst))}"
        )
        for name, target in dst.items():
            target.copy_(self.params[name])
