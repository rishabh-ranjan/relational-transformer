"""rel2tab, pruned to the one baseline this experiment runs.

Restored from ``src/rt/rel2tab/`` as of ``b999183^`` (the commit that removed it),
with everything the ``precomputed_rdblearn`` + ``tabicl_batched`` baseline does not
need stripped out: the RelBench-only SQL featurizer and its per-database query
registry, the xgboost / lgbm / ridge / linear / mean / identity predictors, the
unbatched ``TabPredictor``, and the entity / global / rt featurizers. Git history
has them if any is wanted back.

Two featurizers remain, and they are two halves of one pipeline:

* :class:`RDBLearnFeaturizer` -- computes depth-2 DFS features. Used offline, by
  ``expts/dbinfer/featurize.py``, to write feature matrices to disk.
* :class:`PrecomputedFeaturizer` -- reads those matrices back by ``node_idx`` at
  eval time. This is the one the baseline evaluates with, hence the method name
  ``precomputed_rdblearn``.
"""

from rel2tab.config import FeaturizerConfig, PredictorConfig, Rel2TabModelConfig
from rel2tab.featurizer import Featurizer
from rel2tab.featurizers import (
    PrecomputedFeaturizer,
    PrecomputedFeaturizerConfig,
    RDBLearnFeaturizer,
    RDBLearnFeaturizerConfig,
)
from rel2tab.model import Rel2TabModel
from rel2tab.predictor import Predictor
from rel2tab.predictors import (
    TabICLBatchedPredictor,
    TabICLBatchedPredictorConfig,
)

__all__ = [
    "Featurizer",
    "Predictor",
    "Rel2TabModel",
    "Rel2TabModelConfig",
    "FeaturizerConfig",
    "PredictorConfig",
    "PrecomputedFeaturizer",
    "PrecomputedFeaturizerConfig",
    "RDBLearnFeaturizer",
    "RDBLearnFeaturizerConfig",
    "TabICLBatchedPredictor",
    "TabICLBatchedPredictorConfig",
]
