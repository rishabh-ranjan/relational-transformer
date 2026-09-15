from expts.repaper.baselines.rel2tabv2.model import Rel2TabModel
from expts.repaper.baselines.rel2tabv2.precomputed import PrecomputedFeaturizer
from expts.repaper.baselines.rel2tabv2.retriever import SamplerRetriever


def build_rel2tab(
    *,
    method: str,
    db: str,
    table: str,
    features_root: str,
    tabicl_dir: str,
    tabicl_max_batch_size: int,
    tabicl_min_bin_size: int,
    tabicl_softmax_temperature: float,
    lgbm_n_jobs: int,
    exaone_ensemble_count: int,
    tabfm_backend: str,
) -> tuple[Rel2TabModel, str]:
    family, predictor_name = method.rsplit("_", 1)
    assert family in ("rdblearn", "sql", "rt", "plurel"), (
        f"unknown feature family {family!r} in method {method!r}"
    )
    assert predictor_name in ("lgbm", "tabicl", "baserate", "exaone", "tabfm"), (
        f"unknown predictor {predictor_name!r} in method {method!r}"
    )

    featurizer = PrecomputedFeaturizer(
        features_root,
        {
            "rdblearn": "rdblearn_features",
            "sql": "sql_features",
            "rt": "rt_features",
            "plurel": "plurel_features",
        }[family],
        [(db, table)],
    )

    if predictor_name == "tabicl":
        from expts.repaper.baselines.rel2tabv2.tabicl_batched import (
            TabICLBatchedPredictor,
        )

        device = "cuda"
        predictor = TabICLBatchedPredictor(
            max_batch_size=tabicl_max_batch_size,
            min_bin_size=tabicl_min_bin_size,
            softmax_temperature=tabicl_softmax_temperature,
            checkpoint_dir=tabicl_dir,
            device=device,
        )
    elif predictor_name == "exaone":
        from expts.repaper.baselines.rel2tabv2.exaone import ExaonePredictor

        device = "cuda"
        predictor = ExaonePredictor(ensemble_count=exaone_ensemble_count, device=device)
    elif predictor_name == "tabfm":
        from expts.repaper.baselines.rel2tabv2.tabfm import TabFMPredictor

        device = "cuda"
        predictor = TabFMPredictor(backend=tabfm_backend, device=device)
    elif predictor_name == "baserate":
        from expts.repaper.baselines.rel2tabv2.baserate import BaseRatePredictor

        device = "cpu"
        predictor = BaseRatePredictor()
    else:
        from expts.repaper.baselines.rel2tabv2.lgbm import LGBMPredictor

        device = "cpu"
        predictor = LGBMPredictor(n_jobs=lgbm_n_jobs)

    return Rel2TabModel(SamplerRetriever(), featurizer, predictor), device
