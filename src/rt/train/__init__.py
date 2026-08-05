"""Pretraining: Muon+AdamW DDP training loop, SWA, preemption-safe resume."""

# _train rather than train/main: a module cannot share a name with a
# function the package re-exports, or the attribute shadows the module.
from rt.train._train import main

__all__ = [
    "main",
]
