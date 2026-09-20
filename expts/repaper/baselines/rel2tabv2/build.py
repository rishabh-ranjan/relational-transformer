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
    tabpfn_dir: str,
    tabpfn_n_estimators: int | str,
    tabpfn_fit_mode: str,
    retriever: str,
    context_sampler: str,
    context_seed: int,
    context_split: str,
    n_rows_list: list[int],
    ctx_keys: list[int],
    pre_dir: str,
    embedder: str,
    d_text: int,
) -> tuple[Rel2TabModel, str]:
    family, predictor_name = method.rsplit("_", 1)
    # rttaps<unit> reads the multi-tap dump at that relational unit, e.g.
    # "rttaps12_tabpfn". The unit rides in the method string rather than in a
    # new run.main argument, so the other rounds' submitters keep working
    # against an unchanged entry point.
    tap_unit = int(family[len("rttaps") :]) if family.startswith("rttaps") else None
    assert family in (
        "rdblearn",
        "sql",
        "rt",
        "plurel",
        "relagent",
        "gnn",
    ) or tap_unit in (1, 4, 8, 12), (
        f"unknown feature family {family!r} in method {method!r}"
    )
    assert predictor_name in (
        "lgbm",
        "tabicl",
        "baserate",
        "exaone",
        "tabfm",
        "tabpfn",
    ), f"unknown predictor {predictor_name!r} in method {method!r}"

    if tap_unit is not None:
        from expts.repaper.baselines.rel2tabv2.taps import TapsFeaturizer

        featurizer = TapsFeaturizer(features_root, [(db, table)], tap_unit)
    else:
        featurizer = PrecomputedFeaturizer(
            features_root,
            {
                "rdblearn": "rdblearn_features",
                "sql": "sql_features",
                "rt": "rt_features",
                "plurel": "plurel_features",
                "relagent": "relagent_features",
                "gnn": "gnn_features",
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
    elif predictor_name == "tabpfn":
        from expts.repaper.baselines.rel2tabv2.tabpfn import TabPFNPredictor

        device = "cuda"
        predictor = TabPFNPredictor(
            n_estimators=tabpfn_n_estimators,
            fit_mode=tabpfn_fit_mode,
            checkpoint_dir=tabpfn_dir,
            device=device,
            # Only the taps dump has cells that are genuinely absent -- a
            # column null in the database, or a seed row with no parent. Every
            # other featurizer produces a dense vector, where a NaN would be a
            # bug to zero out rather than a fact to pass on.
            nan_as_missing=tap_unit is not None,
        )
    elif predictor_name == "baserate":
        from expts.repaper.baselines.rel2tabv2.baserate import BaseRatePredictor

        device = "cpu"
        predictor = BaseRatePredictor()
    else:
        from expts.repaper.baselines.rel2tabv2.lgbm import LGBMPredictor

        device = "cpu"
        predictor = LGBMPredictor(n_jobs=lgbm_n_jobs)

    labels = None
    if retriever == "sampler":
        retr = SamplerRetriever()
    elif retriever == "global_train":
        from expts.repaper.baselines.rel2tabv2.context_sampler import (
            UniformRandomSampler,
        )
        from expts.repaper.baselines.rel2tabv2.global_retriever import (
            GlobalContextRetriever,
        )
        from expts.repaper.baselines.rel2tabv2.labels import PreprocessedLabels

        assert context_sampler == "uniform", (
            f"unknown context sampler {context_sampler!r}"
        )
        retr = GlobalContextRetriever(
            sampler=UniformRandomSampler(seed=context_seed),
            db=db,
            table=table,
            pre_dir=pre_dir,
            context_split=context_split,
            n_rows_list=n_rows_list,
            ctx_keys=ctx_keys,
        )
        labels = PreprocessedLabels(pre_dir=pre_dir, embedder=embedder, d_text=d_text)
    else:
        raise AssertionError(f"unknown retriever {retriever!r}")

    return Rel2TabModel(retr, featurizer, predictor, labels), device
