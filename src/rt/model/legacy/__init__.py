"""Legacy model architectures for released checkpoints of earlier papers.

- :mod:`rt.model.legacy.v1`: RT-v1 (ICLR 2026, arXiv:2510.06377).
- :mod:`rt.model.legacy.plurel`: RT-PluRel (ICML 2026, arXiv:2602.04029).

Both are faithful copies of the original ``rt/model.py`` files, adapted only
at the edges to the current data pipeline: they consume the current rustler
batch dict (``is_targets`` instead of the old ``masks`` key) and expose the
``predict`` method the current :class:`rt.eval.evaluator.Evaluator` drives.
State-dict keys match the released ``.pt`` checkpoints exactly.
"""

from rt.model.legacy.plurel import PluRelTransformer
from rt.model.legacy.v1 import V1Transformer

__all__ = ["V1Transformer", "PluRelTransformer"]
