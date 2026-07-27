from dataclasses import dataclass

from rel2tab.featurizers import (
    PrecomputedFeaturizerConfig,
    RDBLearnFeaturizerConfig,
)
from rel2tab.predictors import TabICLBatchedPredictorConfig

FeaturizerConfig = RDBLearnFeaturizerConfig | PrecomputedFeaturizerConfig
PredictorConfig = TabICLBatchedPredictorConfig


@dataclass
class Rel2TabModelConfig:
    """Config for Rel2TabModel.

    Fully independent of rt.config.ModelConfig.  Use ``build(device)`` to
    construct a ready-to-use Rel2TabModel.
    """

    featurizer: FeaturizerConfig
    predictor: PredictorConfig
    featurize_batch_size: int
    embedding_model: str
    d_text: int

    def build(self, device):
        from rel2tab.model import Rel2TabModel

        featurizer = self.featurizer.build(device)
        predictor = self.predictor.build()

        return Rel2TabModel(
            featurizer=featurizer,
            predictor=predictor,
            featurize_batch_size=self.featurize_batch_size,
        )
