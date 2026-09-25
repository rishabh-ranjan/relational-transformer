import re

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
    # rttaps<unit><subset> indexes the multi-tap dump: one relational unit,
    # one slot subset, e.g. "rttaps12full_tabpfn", "rttaps4target_tabpfn".
    # Nothing is re-featurized -- the blob holds every unit and every slot,
    # and this picks which of them become the feature vector. Both ride in the
    # method string rather than in new run.main arguments, so the other
    # rounds' submitters keep working against an unchanged entry point.
    taps_spec = re.fullmatch(r"rttaps(\d+)(full|target|other|proj)", family)
    tap_unit = int(taps_spec.group(1)) if taps_spec else None
    tap_subset = taps_spec.group(2) if taps_spec else None
    # rtadapter<name> is rt_features put through a trained linear adapter, with
    # "identity" the control that isolates the adapter from everything else the
    # arm changes. The name resolves under SHARE/adapters, so it rides in the
    # method string like the taps family and run.main's signature is unchanged.
    adapter_spec = re.fullmatch(r"rtadapter([A-Za-z0-9_-]+)", family)
    adapter_name = adapter_spec.group(1) if adapter_spec else None
    pool_spec = re.fullmatch(r"rtpool(head|swa)", family)
    pool_variant = pool_spec.group(1) if pool_spec else None
    assert (
        family
        in (
            "rdblearn",
            "sql",
            "rt",
            "plurel",
            "relagent",
            "gnn",
            "entity",
        )
        or tap_unit in (1, 4, 8, 12)
        or adapter_name is not None
        or pool_variant is not None
        or family == "rtrows"
    ), f"unknown feature family {family!r} in method {method!r}"
    assert predictor_name in (
        "lgbm",
        "tabicl",
        "baserate",
        "exaone",
        "tabfm",
        "tabpfn",
    ), f"unknown predictor {predictor_name!r} in method {method!r}"

    if family == "rtrows":
        from expts.repaper.baselines.rel2tabv2.rtj_rows import RowsFeaturizer

        featurizer = RowsFeaturizer(features_root, [(db, table)])
    elif pool_variant is not None:
        from expts.repaper.baselines.rel2tabv2.pool_features import PoolFeaturizer

        featurizer = PoolFeaturizer(features_root, [(db, table)], pool_variant)
    elif adapter_name is not None:
        from expts.repaper.baselines.rel2tabv2.adapter import AdapterFeaturizer
        from expts.repaper.config import SHARE

        featurizer = AdapterFeaturizer(
            features_root, [(db, table)], adapter_name, f"{SHARE}/adapters"
        )
    elif tap_unit is not None:
        from expts.repaper.baselines.rel2tabv2.taps import TapsFeaturizer

        featurizer = TapsFeaturizer(features_root, [(db, table)], tap_unit, tap_subset)
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
                "entity": "entity_features",
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
