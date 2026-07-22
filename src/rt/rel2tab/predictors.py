"""rel2tab predictors: fit on in-context labels, predict the target rows."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Literal
import math
from collections import defaultdict
from contextlib import contextmanager
import torch
import importlib as _importlib  # noqa: E402
import sys as _sys  # noqa: E402


# -------------------------------------------------------------------------- #
# predictor
# -------------------------------------------------------------------------- #
class Predictor(ABC):
    """Maps train rows and a test-row feature into a single scalar prediction.

    Called once per (batch-item, context-size) after the featurizer has
    selected/transformed the visible rows.
    """

    @abstractmethod
    def predict(self, train_features, train_labels, test_features, task_type):
        """Produce a prediction for one target row.

        Args:
            train_features: (num_train, d_feat) Tensor, or None if the
                featurizer does not produce features.
            train_labels: 1-D float Tensor of train labels (may be empty).
            test_features: (d_feat,) Tensor, or None.
            task_type: "clf" for binary classification, "reg" for regression.

        Returns:
            Scalar float: probability in [0, 1] for clf, real value for reg.
            Convention when no train data is available: 0.5 for clf, 0.0 for reg.
        """


# -------------------------------------------------------------------------- #
# mean_predictor
# -------------------------------------------------------------------------- #
@dataclass
class MeanPredictorConfig:
    """Config for MeanPredictor (no fields needed)."""

    def build(self):
        return MeanPredictor()


class MeanPredictor(Predictor):
    """Predict the mean of training labels (ignores features)."""

    def predict(self, train_features, train_labels, test_features, task_type):
        if len(train_labels) == 0:
            return 0.5 if task_type == "clf" else 0.0
        return train_labels.mean().item()


# -------------------------------------------------------------------------- #
# linear_predictor
# -------------------------------------------------------------------------- #
@dataclass
class LinearPredictorConfig:
    """Config for LinearPredictor (no fields needed)."""

    def build(self):
        return LinearPredictor()


class LinearPredictor(Predictor):
    """Fit an sklearn linear (regression) or logistic (classification) model."""

    def predict(self, train_features, train_labels, test_features, task_type):
        from sklearn.linear_model import LinearRegression, LogisticRegression

        if train_features is None or len(train_labels) < 2:
            return 0.5 if task_type == "clf" else 0.0

        X_train = train_features.float().cpu().numpy()
        y_train = train_labels.float().cpu().numpy()
        X_test = test_features.float().cpu().numpy().reshape(1, -1)

        if task_type == "clf":
            y_int = (y_train > 0).astype(int)
            if len(np.unique(y_int)) < 2:
                return float(y_int[0])
            model = LogisticRegression(max_iter=200, solver="lbfgs")
            model.fit(X_train, y_int)
            return float(model.predict_proba(X_test)[0, 1])
        else:
            model = LinearRegression()
            model.fit(X_train, y_train)
            return float(model.predict(X_test)[0])


# -------------------------------------------------------------------------- #
# ridge_predictor: Ridge predictor — regularized linear model for few-shot tabular prediction.
# -------------------------------------------------------------------------- #
@dataclass
class RidgePredictorConfig:
    alpha_clf: float
    alpha_reg: float

    def build(self):
        return RidgePredictor(alpha_clf=self.alpha_clf, alpha_reg=self.alpha_reg)


class RidgePredictor(Predictor):
    """Fit a regularized linear model per prediction."""

    def __init__(self, alpha_clf, alpha_reg):
        self.alpha_clf = alpha_clf
        self.alpha_reg = alpha_reg

    def predict(self, train_features, train_labels, test_features, task_type):
        from sklearn.linear_model import Ridge, LogisticRegression

        if train_features is None or len(train_labels) < 2:
            return 0.5 if task_type == "clf" else 0.0

        X_train = train_features.float().cpu().numpy()
        y_train = train_labels.float().cpu().numpy()
        X_test = test_features.float().cpu().numpy().reshape(1, -1)

        # Standardize features to prevent numerical blowup
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std[std == 0] = 1.0
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

        if task_type == "clf":
            y_int = (y_train > 0).astype(int)
            if len(np.unique(y_int)) < 2:
                return float(y_int[0])
            model = LogisticRegression(
                C=1.0 / max(self.alpha_clf, 1e-8),
                max_iter=500,
                solver="lbfgs",
            )
            model.fit(X_train, y_int)
            return float(model.predict_proba(X_test)[0, 1])
        else:
            model = Ridge(alpha=self.alpha_reg)
            model.fit(X_train, y_train)
            return float(model.predict(X_test)[0])


# -------------------------------------------------------------------------- #
# lgbm_predictor: LightGBM predictor for the rel2tab pipeline.
# -------------------------------------------------------------------------- #
@dataclass
class LGBMPredictorConfig:
    n_estimators: int
    num_leaves: int
    learning_rate: float
    min_child_samples: int
    reg_lambda: float

    def build(self):
        return LGBMPredictor(
            n_estimators=self.n_estimators,
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            min_child_samples=self.min_child_samples,
            reg_lambda=self.reg_lambda,
        )


class LGBMPredictor(Predictor):
    """Fit a LightGBM model per prediction."""

    def __init__(
        self,
        n_estimators,
        num_leaves,
        learning_rate,
        min_child_samples,
        reg_lambda,
    ):
        self.params = dict(
            n_estimators=n_estimators,
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            min_child_samples=min_child_samples,
            reg_lambda=reg_lambda,
            verbose=-1,
        )

    def predict(self, train_features, train_labels, test_features, task_type):
        from lightgbm import LGBMClassifier, LGBMRegressor

        if train_features is None or len(train_labels) < 2:
            return 0.5 if task_type == "clf" else 0.0

        X_train = train_features.float().cpu().numpy()
        y_train = train_labels.float().cpu().numpy()
        X_test = test_features.float().cpu().numpy().reshape(1, -1)

        if task_type == "clf":
            y_int = (y_train > 0).astype(int)
            if len(np.unique(y_int)) < 2:
                return float(y_int[0])
            model = LGBMClassifier(**self.params)
            model.fit(X_train, y_int)
            return float(model.predict_proba(X_test)[0, 1])
        else:
            model = LGBMRegressor(**self.params)
            model.fit(X_train, y_train)
            return float(model.predict(X_test)[0])


# -------------------------------------------------------------------------- #
# xgboost_predictor: XGBoost predictor for the rel2tab pipeline.
# -------------------------------------------------------------------------- #
@dataclass
class XGBoostHP:
    """One global hyperparameter set (used for either clf or reg).

    Args:
        n_estimators: max boosting rounds (upper bound when early stopping on).
        max_depth: tree depth.  2-4 is the sweet spot for <=400 rows.
        learning_rate: shrinkage.  Smaller is steadier; pairs with more rounds.
        min_child_weight: min sum of instance weight (hessian) per leaf —
            higher = stronger regularization (fewer, larger leaves).
        subsample: row subsampling per tree (stochastic regularization).
        colsample_bytree: column subsampling per tree.  Important for the
            high-dim RDBLearn rel-event features (~450 cols, few rows).
        reg_lambda: L2 penalty on leaf weights.
        reg_alpha: L1 penalty on leaf weights.
        early_stopping_frac: if > 0, hold out this fraction of the in-context
            labels (stratified for clf) to early-stop on n_estimators.
    """

    n_estimators: int
    max_depth: int
    learning_rate: float
    min_child_weight: float
    subsample: float
    colsample_bytree: float
    reg_lambda: float
    reg_alpha: float
    early_stopping_frac: float

    def xgb_params(self, n_jobs):
        return dict(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,
            tree_method="hist",
            n_jobs=n_jobs,
            verbosity=0,
        )


@dataclass
class XGBoostPredictorConfig:
    """Config for XGBoostPredictor.

    Carries two global HP sets — ``clf`` and ``reg`` — analogous to the Ridge
    predictor's ``alpha_clf`` / ``alpha_reg`` split.

    Args:
        clf: HP set for classification tasks.
        reg: HP set for regression tasks.
        n_jobs: XGBoost threads per fit.  Kept to 1 because the outer per-item
            loop is what we parallelize across (via SLURM/DDP workers);
            per-fit threading on tiny data mostly adds overhead.
    """

    clf: XGBoostHP
    reg: XGBoostHP
    n_jobs: int

    def build(self):
        return XGBoostPredictor(clf=self.clf, reg=self.reg, n_jobs=self.n_jobs)


class XGBoostPredictor(Predictor):
    """Fit an XGBoost model per prediction (separate clf / reg HP sets)."""

    def __init__(self, clf, reg, n_jobs):
        self.clf_hp = clf
        self.reg_hp = reg
        self.n_jobs = n_jobs

    def predict(self, train_features, train_labels, test_features, task_type):
        from xgboost import XGBClassifier, XGBRegressor

        if train_features is None or len(train_labels) < 2:
            return 0.5 if task_type == "clf" else 0.0

        X_train = train_features.float().cpu().numpy()
        y_train = train_labels.float().cpu().numpy()
        X_test = test_features.float().cpu().numpy().reshape(1, -1)

        if task_type == "clf":
            hp = self.clf_hp
            y_int = (y_train > 0).astype(int)
            if len(np.unique(y_int)) < 2:
                return float(y_int[0])
            # scale_pos_weight = (#neg / #pos) to counter class imbalance.
            n_pos = int(y_int.sum())
            n_neg = int(len(y_int) - n_pos)
            spw = (n_neg / n_pos) if n_pos > 0 else 1.0
            model = XGBClassifier(
                **hp.xgb_params(self.n_jobs),
                objective="binary:logistic",
                eval_metric="logloss",
                scale_pos_weight=spw,
            )
            self._fit(model, X_train, y_int, hp.early_stopping_frac, stratify=True)
            return float(model.predict_proba(X_test)[0, 1])
        else:
            hp = self.reg_hp
            model = XGBRegressor(
                **hp.xgb_params(self.n_jobs),
                objective="reg:squarederror",
                eval_metric="mae",
            )
            self._fit(model, X_train, y_train, hp.early_stopping_frac, stratify=False)
            return float(model.predict(X_test)[0])

    def _fit(self, model, X, y, frac, stratify):
        """Fit with optional early stopping on a held-out slice of X.

        When ``frac == 0``, a plain fixed-round fit is used.
        """
        n = len(y)
        # Need enough rows on both sides for a meaningful holdout; also require
        # both classes present in train+val for clf.
        n_val = int(round(frac * n)) if frac > 0 else 0
        if n_val < 1 or (n - n_val) < 2:
            model.set_params(early_stopping_rounds=None)
            model.fit(X, y)
            return

        rng = np.random.RandomState(0)
        if stratify and len(np.unique(y)) == 2:
            idx_val = []
            for cls in (0, 1):
                cls_idx = np.where(y == cls)[0]
                k = max(1, int(round(frac * len(cls_idx))))
                k = min(k, len(cls_idx) - 1) if len(cls_idx) > 1 else 0
                if k > 0:
                    idx_val.extend(rng.choice(cls_idx, size=k, replace=False).tolist())
            idx_val = np.array(sorted(idx_val), dtype=int)
        else:
            perm = rng.permutation(n)
            idx_val = perm[:n_val]

        if len(idx_val) < 1 or (n - len(idx_val)) < 2:
            model.set_params(early_stopping_rounds=None)
            model.fit(X, y)
            return

        mask = np.ones(n, dtype=bool)
        mask[idx_val] = False
        X_tr, y_tr = X[mask], y[mask]
        X_va, y_va = X[~mask], y[~mask]
        # Degenerate clf holdout (single class on either side): skip ES.
        if stratify and (len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2):
            model.set_params(early_stopping_rounds=None)
            model.fit(X, y)
            return

        model.set_params(early_stopping_rounds=20)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)


# -------------------------------------------------------------------------- #
# xgboost_tuned: Tuned global XGBoost hyperparameter sets for the precomputed-feature
# -------------------------------------------------------------------------- #
# Optional runtime override: if env var XGB_TUNED_JSON points to a JSON file,
# read tuned clf/reg HP sets from it. Lets a single job tune then eval with the
# just-found winners without a source edit/commit. JSON schema:
#   {"sql_features": {"clf": {<XGBoostHP fields>}, "reg": {...}},
#    "rdblearn_features": {"clf": {...}, "reg": {...}}}
_OVERRIDE_ENV = "XGB_TUNED_JSON"

# ===================== SQL features (val-tuned) =====================
SQL_TUNED_CLF = XGBoostHP(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    min_child_weight=5.0,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=5.0,
    reg_alpha=0.0,
    early_stopping_frac=0.0,
)
SQL_TUNED_REG = XGBoostHP(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    min_child_weight=5.0,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=5.0,
    reg_alpha=0.0,
    early_stopping_frac=0.0,
)

# =================== RDBLearn features (val-tuned) ===================
RDBLEARN_TUNED_CLF = XGBoostHP(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    min_child_weight=5.0,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=5.0,
    reg_alpha=0.0,
    early_stopping_frac=0.0,
)
RDBLEARN_TUNED_REG = XGBoostHP(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    min_child_weight=5.0,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=5.0,
    reg_alpha=0.0,
    early_stopping_frac=0.0,
)


def tuned_xgboost_config(features_subdir):
    """Return the val-tuned XGBoostPredictorConfig for a feature set.

    ``features_subdir`` is "sql_features" or "rdblearn_features".
    """
    override_path = os.environ.get(_OVERRIDE_ENV)
    if override_path and os.path.exists(override_path):
        with open(override_path) as f:
            blob = json.load(f)
        if features_subdir in blob:
            entry = blob[features_subdir]
            clf = XGBoostHP(**entry["clf"])
            reg = XGBoostHP(**entry["reg"])
            return XGBoostPredictorConfig(clf=clf, reg=reg, n_jobs=1)

    if features_subdir == "sql_features":
        clf, reg = SQL_TUNED_CLF, SQL_TUNED_REG
    elif features_subdir == "rdblearn_features":
        clf, reg = RDBLEARN_TUNED_CLF, RDBLEARN_TUNED_REG
    else:
        raise ValueError(
            f"No tuned XGBoost config for features_subdir={features_subdir!r}"
        )
    return XGBoostPredictorConfig(clf=clf, reg=reg, n_jobs=1)


# -------------------------------------------------------------------------- #
# tab_predictor
# -------------------------------------------------------------------------- #
@dataclass
class TabPredictorConfig:
    """Config for TabPredictor (unified TabICL/TabPFN)."""

    model: Literal["tabicl", "tabpfn"]
    num_workers: int

    def build(self):
        return TabPredictor(model=self.model, num_workers=self.num_workers)


class TabPredictor(Predictor):
    """Fit a tabular foundation model on train features and predict for test row.

    Supports TabICL and TabPFN via the ``model`` argument. Uses thread-local
    model instances for concurrent prediction via ``predict_batch``.
    """

    def __init__(self, model, num_workers):
        import torch

        self.model = model
        self.num_workers = num_workers
        self._device = f"cuda:{torch.cuda.current_device()}"
        self._local = threading.local()

    def _get_thread_models(self):
        """Return thread-local (clf, reg) pair, creating them if needed."""
        if not hasattr(self._local, "clf"):
            if self.model == "tabicl":
                from tabicl import TabICLClassifier, TabICLRegressor

                self._local.clf = TabICLClassifier(device=self._device, use_amp=False)
                self._local.reg = TabICLRegressor(device=self._device, use_amp=False)
            elif self.model == "tabpfn":
                from tabpfn import TabPFNClassifier, TabPFNRegressor

                self._local.clf = TabPFNClassifier(device=self._device)
                self._local.reg = TabPFNRegressor(device=self._device)
        return self._local.clf, self._local.reg

    def _predict_one(self, train_features, train_labels, test_features, task_type):
        """Single prediction using thread-local models."""
        if train_features is None or len(train_labels) < 2:
            return 0.5 if task_type == "clf" else 0.0

        X_train = train_features.float().cpu().numpy()
        y_train = train_labels.float().cpu().numpy()
        X_test = test_features.float().cpu().numpy().reshape(1, -1)

        if np.all(X_train == X_train[0]):
            return float(y_train.mean())

        clf, reg = self._get_thread_models()

        if task_type == "clf":
            y_int = (y_train > 0).astype(int)
            if len(np.unique(y_int)) < 2:
                return float(y_int[0])
            clf.fit(X_train, y_int)
            return float(clf.predict_proba(X_test)[0, 1])
        else:
            reg.fit(X_train, y_train)
            return float(reg.predict(X_test)[0])

    def predict(self, train_features, train_labels, test_features, task_type):
        return self._predict_one(train_features, train_labels, test_features, task_type)

    def predict_batch(self, work_items):
        """Predict many items concurrently using thread-local model copies.

        Args:
            work_items: list of (train_features, train_labels, test_features,
                task_type) tuples.

        Returns:
            list of scalar float predictions, one per work item.
        """
        with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
            futures = [
                pool.submit(self._predict_one, tf, tl, xf, tt)
                for tf, tl, xf, tt in work_items
            ]
            return [f.result() for f in futures]


# -------------------------------------------------------------------------- #
# tabicl_batched_predictor: Batched TabICL predictor.
# -------------------------------------------------------------------------- #
# --- tabicl module-path compatibility shim -----------------------------------
# The attention patch + TabICL imports below use ``tabicl.model.*``. Some
# installed tabicl builds (e.g. the 2.1.1 wheel pinned in pixi.lock) expose the
# internals under ``tabicl._model`` instead (identical submodules: attention,
# layers, ssmax, tabicl). Alias the public path to the private one when only the
# latter exists so the patched imports resolve regardless of wheel layout.

try:  # pragma: no cover - import-time environment shim
    _importlib.import_module("tabicl.model")
except ModuleNotFoundError:
    try:
        _priv = _importlib.import_module("tabicl._model")
        _sys.modules.setdefault("tabicl.model", _priv)
        for _sub in ("attention", "layers", "ssmax", "tabicl"):
            try:
                _sys.modules.setdefault(
                    f"tabicl.model.{_sub}",
                    _importlib.import_module(f"tabicl._model.{_sub}"),
                )
            except ModuleNotFoundError:
                pass
    except ModuleNotFoundError:
        pass

_PROFILE_PATH = os.environ.get("TABICL_BATCHED_PROFILE_PATH")


_CLF_CHECKPOINT = "tabicl-classifier-v2-20260212.ckpt"
_REG_CHECKPOINT = "tabicl-regressor-v2-20260212.ckpt"
_HF_REPO = "jingang/TabICL"


# --- Thread-local key-padding mask plumbing ----------------------------------
#
# Two thread-local pieces of state, both set together by ``_padded_forward``:
#
#   _local.mask       : (B, train_size) bool, True = padded position to ignore
#   _local.real_lens  : (B,) int, per-item real train_size before padding
#
# ``_local.mask`` is read by the patched
# ``MultiheadAttentionBlock.forward`` and injected as ``key_padding_mask``
# whenever the block's keys span the padded train portion.
#
# ``_local.real_lens`` is read by the patched
# ``sdpa_with_flattened_batch``, which would otherwise compute SSMax with
# ``src_len = bin_size``.  Per-item ``log(real_n)`` is required for SSMax
# correctness — without it, padded-vs-unpadded predictions diverge by
# ~25 percentage points on small items even when the mask is right, because
# every attention layer's queries are scaled by ``log(bin_size)`` instead of
# ``log(real_n[b])``.
#
# Both patches are pass-throughs when the thread-locals are unset.

_local = threading.local()
_patch_installed = False
_patch_lock = threading.Lock()


def _per_item_ssmax(ssmax_layer, q, n_tensor):
    """Apply ``ssmax_layer`` to ``q`` with per-flat-item ``n``.

    ``q`` has shape ``(flat_bs, n_heads, tgt_len, head_dim)`` and ``n_tensor``
    has shape ``(flat_bs,)``.  This mirrors the original SSMax forward but
    broadcasts ``logn`` along the leading dim instead of using a scalar.
    """
    from tabicl.model.ssmax import QASSMaxMLP, SSMax, SSMaxMLP

    logn = torch.log(n_tensor.clamp(min=1).to(q.dtype)).reshape(-1, 1)
    flat_bs, nh, _, hs = q.shape

    if isinstance(ssmax_layer, SSMax):
        s = ssmax_layer.scales.view(1, nh, 1, 1)
        return q * (s * logn.view(-1, 1, 1, 1))
    if isinstance(ssmax_layer, SSMaxMLP):
        out = ssmax_layer.mlp(logn)
        if ssmax_layer.elementwise:
            scales = out.view(flat_bs, nh, 1, hs)
        else:
            scales = out.view(flat_bs, nh, 1, 1)
        return q * scales
    if isinstance(ssmax_layer, QASSMaxMLP):
        if ssmax_layer.elementwise:
            base = ssmax_layer.base_mlp(logn).view(flat_bs, nh, 1, hs)
        else:
            base = ssmax_layer.base_mlp(logn).view(flat_bs, nh, 1, 1)
        modulation = 1 + torch.tanh(ssmax_layer.query_mlp(q))
        return q * (base * modulation)
    raise TypeError(f"unknown SSMax layer type: {type(ssmax_layer).__name__}")


def _install_attention_patch():
    """Idempotently patch the two TabICL attention sites.

    The patches are pass-throughs when ``_local.mask`` is unset, so direct
    use of TabICL elsewhere in the process is unaffected.
    """
    global _patch_installed
    with _patch_lock:
        if _patch_installed:
            return
        from torch.nn import functional as F

        from tabicl.model import attention as _attn_mod
        from tabicl.model.layers import MultiheadAttentionBlock

        orig_block_forward = MultiheadAttentionBlock.forward
        orig_sdpa = _attn_mod.sdpa_with_flattened_batch

        def patched_block_forward(
            self,
            q,
            k=None,
            v=None,
            cached_kv=None,
            key_padding_mask=None,
            attn_mask=None,
            train_size=None,
            rope=None,
            need_kv=False,
        ):
            m = getattr(_local, "mask", None)
            if m is not None and key_padding_mask is None and cached_kv is None:
                if train_size is not None:
                    eff_k_len = train_size
                elif k is not None:
                    eff_k_len = k.shape[-2]
                else:
                    eff_k_len = q.shape[-2]
                if eff_k_len == m.shape[-1]:
                    batch_shape = q.shape[:-2]
                    extra = len(batch_shape) - 1
                    view_shape = [m.shape[0]] + [1] * extra + [m.shape[1]]
                    key_padding_mask = m.view(*view_shape).expand(
                        *batch_shape, m.shape[1]
                    )
            return orig_block_forward(
                self,
                q,
                k,
                v,
                cached_kv=cached_kv,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                train_size=train_size,
                rope=rope,
                need_kv=need_kv,
            )

        def patched_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, ssmax_layer=None):
            m = getattr(_local, "mask", None)
            real_lens = getattr(_local, "real_lens", None)
            if m is None or real_lens is None:
                return orig_sdpa(q, k, v, attn_mask, dropout_p, ssmax_layer)

            q_shape = q.shape
            q_flat = q.reshape(-1, *q.shape[-3:])
            k_flat = k.reshape(-1, *k.shape[-3:])
            v_flat = v.reshape(-1, *v.shape[-3:])
            am = (
                None
                if attn_mask is None
                else attn_mask.reshape(-1, *attn_mask.shape[-3:])
            )

            if ssmax_layer is not None:
                src_len = k_flat.size(-2)
                if src_len == m.shape[-1]:
                    flat_bs = q_flat.shape[0]
                    B = real_lens.shape[0]
                    multiplier = flat_bs // B
                    if flat_bs != B * multiplier:
                        raise RuntimeError(f"flat_bs={flat_bs} not a multiple of B={B}")
                    flat_real_n = real_lens.repeat_interleave(multiplier).to(
                        q_flat.device
                    )
                    q_flat = _per_item_ssmax(ssmax_layer, q_flat, flat_real_n)
                else:
                    q_flat = ssmax_layer(q_flat, src_len)

            out = F.scaled_dot_product_attention(q_flat, k_flat, v_flat, am, dropout_p)
            return out.view(q_shape)

        MultiheadAttentionBlock.forward = patched_block_forward
        _attn_mod.sdpa_with_flattened_batch = patched_sdpa
        _patch_installed = True


@contextmanager
def _padded_forward(mask, real_lens):
    """Activate the masked + per-item-SSMax forward for the duration of the call.

    Args:
        mask: bool tensor ``(B, train_size)``, True at padded positions.
        real_lens: int tensor ``(B,)``, real train_size per item before padding.
    """
    prev_mask = getattr(_local, "mask", None)
    prev_lens = getattr(_local, "real_lens", None)
    _local.mask = mask
    _local.real_lens = real_lens
    try:
        yield
    finally:
        _local.mask = prev_mask
        _local.real_lens = prev_lens


# --- Predictor ---------------------------------------------------------------


@dataclass
class TabICLBatchedPredictorConfig:
    """Config for ``TabICLBatchedPredictor``.

    Args:
        max_batch_size: Maximum number of sequences stacked into one TabICL
            forward pass.  Larger values increase GPU utilization but use more
            memory; TabICL's internal auto-offload will further chunk if needed.
        min_bin_size: Smallest bin used for padding.  Items with
            ``train_size < min_bin_size`` are padded up to this value; raising
            it merges very small items into bigger groups at the cost of more
            padding overhead.
        softmax_temperature: Temperature applied to classification logits.
        use_amp: Kept for API/config compatibility, but the batched forward
            **always runs in fp32** regardless of this flag (see
            ``TabICLBatchedPredictor`` for why).  Set it to whatever; it no
            longer changes numerics.
    """

    max_batch_size: int
    min_bin_size: int
    softmax_temperature: float
    use_amp: bool

    def build(self):
        return TabICLBatchedPredictor(
            max_batch_size=self.max_batch_size,
            min_bin_size=self.min_bin_size,
            softmax_temperature=self.softmax_temperature,
            use_amp=self.use_amp,
        )


class TabICLBatchedPredictor(Predictor):
    """TabICL predictor that batches many independent sequences per forward.

    eval_bs-invariance and AMP
    --------------------------
    The whole point of this predictor is that an item's prediction must NOT
    depend on how many other items happen to share its TabICL forward pass
    (``eval_bs`` / chunk composition).  The thread-local key-padding mask and
    per-item SSMax make the *mathematical* result batch-independent, and with
    fp32 math the per-item predictions are bit-stable across ``max_batch_size``
    (only ~1e-6 SDPA reduction-order jitter remains).

    However, ``torch.autocast`` (AMP / bf16) breaks this: when ``B`` items are
    stacked into a single ``(B, ...)`` forward, the bf16 batched matmuls
    (cuBLAS) pick kernels / accumulation order as a function of the batch
    dimension ``B``.  The same item therefore gets a slightly different bf16
    result depending on the chunk it lands in.  This is purely a numerical
    (kernel-selection) effect — it is *uncorrelated with padding* (a barely
    padded item moves just as much as a heavily padded one) — but it is large
    enough in bf16 to swing AUROC by >1pt at large contexts (e.g. ctx 16384:
    ~0.766 at eval_bs=1 vs ~0.765 at eval_bs=8, with per-item probability
    deltas up to ~0.1).  It cannot be removed while keeping bf16 stacked
    matmuls, because the nondeterminism is inherent to batched bf16 GEMMs.

    Fix: run the batched forward in **fp32** (``use_amp`` is forced off in the
    inference configs below).  fp32 stacked matmuls are batch-invariant, so the
    per-item predictions become identical (to fp tolerance) across every
    ``max_batch_size``.  fp32 costs ~1.8x throughput at the largest contexts
    and is essentially free at moderate ones; even so, batched fp32 is *faster*
    than the previous production workaround (forcing eval_bs=1 in bf16), which
    serialised every item.  Accuracy is unchanged: eval_bs=1 bf16 and fp32
    agree to ~1e-3 / <0.01 AUROC.
    """

    def __init__(self, max_batch_size, min_bin_size, softmax_temperature, use_amp):
        self.max_batch_size = max_batch_size
        self.min_bin_size = min_bin_size
        self.softmax_temperature = softmax_temperature
        # Recorded for reference only; the batched forward always runs in fp32
        # (see class docstring) to guarantee eval_bs-invariance.
        self.use_amp = use_amp
        self._effective_use_amp = False

        self._device = f"cuda:{torch.cuda.current_device()}"
        self._clf_model = None
        self._reg_model = None
        self._clf_inference_config = None
        self._reg_inference_config = None

        _install_attention_patch()

    def _build_inference_config(self):
        from tabicl import InferenceConfig

        cfg = InferenceConfig()
        cfg.update_from_dict(
            {
                "COL_CONFIG": {
                    "device": self._device,
                    "use_amp": self._effective_use_amp,
                    "use_fa3": False,
                    "verbose": False,
                    "offload": "auto",
                    "disk_offload_dir": None,
                },
                "ROW_CONFIG": {
                    "device": self._device,
                    "use_amp": self._effective_use_amp,
                    "use_fa3": False,
                    "verbose": False,
                },
                "ICL_CONFIG": {
                    "device": self._device,
                    "use_amp": self._effective_use_amp,
                    "use_fa3": False,
                    "verbose": False,
                },
            }
        )
        return cfg

    def _load_model(self, filename):
        from huggingface_hub import hf_hub_download
        from tabicl.model.tabicl import TabICL

        path = hf_hub_download(repo_id=_HF_REPO, filename=filename)
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        model = TabICL(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval().to(self._device)
        return model

    def _ensure_clf(self):
        if self._clf_model is None:
            self._clf_model = self._load_model(_CLF_CHECKPOINT)
            self._clf_inference_config = self._build_inference_config()

    def _ensure_reg(self):
        if self._reg_model is None:
            self._reg_model = self._load_model(_REG_CHECKPOINT)
            self._reg_inference_config = self._build_inference_config()

    @staticmethod
    def _trivial_pred(train_features, train_labels, test_features, task_type):
        """Return early prediction if work item is trivial; else None.

        Trivial cases also catch any inputs we can't safely standardize: a
        feature tensor that's entirely non-finite, or a label tensor that's
        non-finite, would otherwise propagate NaN through TabICL.
        """
        if train_features is None or len(train_labels) < 2:
            return 0.5 if task_type == "clf" else 0.0

        X_train = train_features.float()
        y_train = train_labels.float()

        if not torch.isfinite(y_train).any():
            return 0.5 if task_type == "clf" else 0.0
        if test_features is not None and not torch.isfinite(test_features).any():
            return (
                0.5
                if task_type == "clf"
                else float(y_train[torch.isfinite(y_train)].mean().item())
            )

        # All-identical features ⇒ TabICL has no signal; predict label mean.
        first_row = X_train[0]
        if (X_train == first_row).all():
            finite_y = y_train[torch.isfinite(y_train)]
            if finite_y.numel() == 0:
                return 0.5 if task_type == "clf" else 0.0
            return float(finite_y.mean().item())

        if task_type == "clf":
            y_int = (y_train > 0).long()
            unique_count = int(torch.unique(y_int).numel())
            if unique_count < 2:
                return float(y_int[0].item())
        return None

    @staticmethod
    def _preprocess_none_batched(X_train, x_test, real_lens):
        """GPU-vectorized equivalent of TabICL's ``PreprocessingPipeline`` with
        ``normalization_method="none"``: ``CustomStandardScaler`` followed by
        ``OutlierRemover``.

        TabICL is trained on ``outlier_removing(threshold=4) → standard_scaling
        (clip=100)`` (see ``tabicl/prior/_reg2cls.py:_process_features``); at
        inference, ``"none"`` applies the same two ops in the reverse order
        (CustomStandardScaler → OutlierRemover).  This helper reproduces that
        inference-time path exactly while processing all ``B`` items in a chunk
        in one vectorized pass.

        ``X_train`` is pre-stacked and zero-padded to ``(B, T, H)``; per-item
        statistics are computed over the first ``real_lens[b]`` rows only.
        Padded train rows in the output have undefined values — the caller's
        key-padding mask hides them in attention.

        Constants and ddof choices match the sklearn classes exactly:

        * ``CustomStandardScaler``: ``mean = X.mean(axis=0)`` and
          ``scale = X.std(axis=0) + 1e-6`` (numpy ``std`` defaults to
          ``ddof=0``, additive epsilon), output clipped to ``[-100, 100]``.
        * ``OutlierRemover``: two-stage Z-score with
          ``ddof=1`` (since real_lens >= 2 by the trivial-case filter),
          ``threshold=4.0``, ``std`` floored at ``1e-6`` via ``np.maximum``;
          transform is the log-soft clip
          ``max(-log1p(|x|) + lower, x); min(log1p(|x|) + upper, x)``.

        Both train and test go through the same two transforms; the bounds
        are computed from the train slice only.
        """
        threshold = 4.0
        eps = 1e-6
        clip_min, clip_max = -100.0, 100.0

        device = X_train.device
        dtype = X_train.dtype
        B, T, _H = X_train.shape

        arange_T = torch.arange(T, device=device).unsqueeze(0)
        real_mask = (arange_T < real_lens.unsqueeze(-1)).unsqueeze(-1)  # (B, T, 1)
        n = real_lens.to(dtype).view(B, 1)  # (B, 1)
        eps_t = torch.tensor(eps, device=device, dtype=dtype)

        # ---- CustomStandardScaler ---- (numpy std default ddof=0; epsilon additive)
        masked_X = X_train * real_mask
        mean_cs = masked_X.sum(dim=1) / n
        diffs_cs = (X_train - mean_cs.unsqueeze(1)) * real_mask
        var_cs = (diffs_cs**2).sum(dim=1) / n
        scale_cs = torch.sqrt(var_cs) + eps

        X_scaled = ((X_train - mean_cs.unsqueeze(1)) / scale_cs.unsqueeze(1)).clamp(
            clip_min, clip_max
        )
        x_test_scaled = ((x_test - mean_cs) / scale_cs).clamp(clip_min, clip_max)

        # ---- OutlierRemover stage 1 ---- (ddof=1 since real_lens >= 2 here)
        masked_Xs = X_scaled * real_mask
        mean_or1 = masked_Xs.sum(dim=1) / n
        diffs_or1 = (X_scaled - mean_or1.unsqueeze(1)) * real_mask
        var_or1 = (diffs_or1**2).sum(dim=1) / (n - 1)
        std_or1 = torch.maximum(torch.sqrt(var_or1), eps_t)

        lower_or1 = mean_or1 - threshold * std_or1
        upper_or1 = mean_or1 + threshold * std_or1

        outlier_mask = (X_scaled < lower_or1.unsqueeze(1)) | (
            X_scaled > upper_or1.unsqueeze(1)
        )
        valid_mask = real_mask & ~outlier_mask  # (B, T, H)

        # ---- OutlierRemover stage 2 ---- (ddof=1; NaN where < 2 valid samples)
        valid_count = valid_mask.sum(dim=1).to(dtype)  # (B, H)

        sum_clean = (X_scaled * valid_mask).sum(dim=1)
        mean_or2 = torch.where(
            valid_count > 0,
            sum_clean / valid_count.clamp(min=1),
            torch.full_like(sum_clean, float("nan")),
        )
        diffs_or2 = (X_scaled - mean_or2.unsqueeze(1)) * valid_mask
        sq_diffs2 = (diffs_or2**2).sum(dim=1)
        var_or2 = torch.where(
            valid_count > 1,
            sq_diffs2 / (valid_count - 1).clamp(min=1),
            torch.full_like(sq_diffs2, float("nan")),
        )
        std_or2 = torch.maximum(torch.sqrt(var_or2), eps_t)

        lower_bounds = mean_or2 - threshold * std_or2  # NaN propagates if all-outlier
        upper_bounds = mean_or2 + threshold * std_or2

        # ---- OutlierRemover.transform: log-soft clip ----
        def _soft_clip(x, lo, hi):
            x = torch.maximum(-torch.log1p(x.abs()) + lo, x)
            x = torch.minimum(torch.log1p(x.abs()) + hi, x)
            return x

        X_out = _soft_clip(
            X_scaled, lower_bounds.unsqueeze(1), upper_bounds.unsqueeze(1)
        )
        x_test_out = _soft_clip(x_test_scaled, lower_bounds, upper_bounds)
        return X_out, x_test_out

    @staticmethod
    def _standardize_y_batched(y_train, real_lens):
        """Per-item ``StandardScaler`` matching sklearn's behavior on
        regression labels.

        sklearn's ``StandardScaler`` uses ``mean`` and ``std`` (``ddof=0``) and
        replaces a zero std with 1.0 via ``_handle_zeros_in_scale``.  This
        matches that exactly; padded label positions don't enter the per-item
        statistics.

        Args:
            y_train: ``(B, T)`` train labels, padded with anything beyond
                ``real_lens[b]`` (those positions are masked out).
            real_lens: ``(B,)`` per-item real n_train.

        Returns:
            ``y_n``: ``(B, T)`` scaled labels (padded positions undefined).
            ``y_mean``: ``(B,)`` per-item mean.
            ``y_std``: ``(B,)`` per-item std (with zero replaced by 1.0).
        """
        device = y_train.device
        dtype = y_train.dtype
        B, T = y_train.shape

        arange_T = torch.arange(T, device=device).unsqueeze(0)
        real_mask = arange_T < real_lens.unsqueeze(-1)
        n = real_lens.to(dtype)

        masked_y = y_train * real_mask
        y_mean = masked_y.sum(dim=1) / n
        diffs = (y_train - y_mean.unsqueeze(-1)) * real_mask
        var = (diffs**2).sum(dim=1) / n
        raw_std = torch.sqrt(var)
        y_std = torch.where(raw_std == 0, torch.ones_like(raw_std), raw_std)
        y_n = (y_train - y_mean.unsqueeze(-1)) / y_std.unsqueeze(-1)
        return y_n, y_mean, y_std

    def predict(self, train_features, train_labels, test_features, task_type):
        return self.predict_batch(
            [(train_features, train_labels, test_features, task_type)]
        )[0]

    @staticmethod
    def _bin_size(n_train, min_bin_size):
        """Return next-pow2 bin >= max(n_train, min_bin_size)."""
        n = max(n_train, min_bin_size)
        return 1 << (n - 1).bit_length()

    @staticmethod
    def _zero_pad(rows, target_n):
        """Right-pad ``rows`` (n, ...) to length ``target_n`` with zeros."""
        n = rows.shape[0]
        if n == target_n:
            return rows
        pad = torch.zeros(
            (target_n - n, *rows.shape[1:]), dtype=rows.dtype, device=rows.device
        )
        return torch.cat([rows, pad], dim=0)

    def predict_batch(self, work_items):
        """Predict many work items by batching forwards through TabICL.

        Args:
            work_items: list of ``(train_features, train_labels, test_features,
                task_type)`` tuples.  ``train_features`` and ``test_features``
                are float Tensors with the same feature dimension across the
                whole batch (they come from one ``compute_features`` call);
                ``train_labels`` is a 1-D float Tensor.

        Returns:
            list of scalar floats, one per work item, in input order.
        """

        n = len(work_items)
        results = [None] * n

        if _PROFILE_PATH is not None:
            sizes = [
                (
                    int(tl.shape[0]) if tl is not None else 0,
                    tt,
                    int(tf.shape[1]) if tf is not None else 0,
                )
                for (tf, tl, _xf, tt) in work_items
            ]
            with open(_PROFILE_PATH, "a") as f:
                f.write(json.dumps({"call": sizes}) + "\n")

        # Phase 1: handle trivial cases, prepare standardized tensors for the rest.
        # Group key: (task_type, bin_size, num_classes_or_None, num_features).
        # For clf, items in a group must share num_classes — TabICL asserts
        # ``len(unique(y_train[0])) == ... == len(unique(y_train[B-1]))``.
        # With our zero-pad, padded labels are 0 which adds class 0 to every
        # item; we therefore force ``num_classes = max(2, observed)`` so
        # binary items keep ``num_classes=2`` after padding.
        groups = defaultdict(list)

        for i, (tf, tl, xf, tt) in enumerate(work_items):
            # NaN imputation matching sklearn's mean strategy: featurizers like
            # SQL/rdblearn produce NaN when aggregates fall on empty groups;
            # propagating NaN through manual standardization gives NaN
            # predictions. The non-batched TabPredictor avoids this because
            # TabICLClassifier/Regressor's wrapper imputes internally.
            if tf is not None:
                tf = tf.float()
                xf = xf.float()
                col_means = torch.nan_to_num(
                    torch.nanmean(tf, dim=0, keepdim=True), nan=0.0
                )
                tf = torch.where(torch.isnan(tf), col_means.expand_as(tf), tf)
                xf = torch.where(torch.isnan(xf), col_means.squeeze(0), xf)
            tl = torch.nan_to_num(tl.float(), nan=0.0)

            triv = self._trivial_pred(tf, tl, xf, tt)
            if triv is not None:
                results[i] = triv
                continue

            X_train = tf  # already float, NaN-imputed
            y_train = tl  # already float, NaN-imputed
            x_test = xf  # already float, NaN-imputed

            n_train, d = X_train.shape
            bin_size = self._bin_size(n_train, self.min_bin_size)

            # Standardization is deferred to the batched preprocessor below;
            # we queue raw tensors here and stack + preprocess once per chunk.
            if tt == "clf":
                y_int = (y_train > 0).long()
                num_classes = max(int(y_int.max().item()) + 1, 2)
                fallback = float(y_int.float().mean().item())
                key = (tt, bin_size, num_classes, d)
                groups[key].append(
                    (
                        i,
                        n_train,
                        X_train,
                        y_int.float(),
                        x_test,
                        fallback,
                    )
                )
            else:
                key = (tt, bin_size, None, d)
                groups[key].append(
                    (
                        i,
                        n_train,
                        X_train,
                        y_train,
                        x_test,
                        0.0,  # fallback unused for reg; recomputed from per-item y_mean
                    )
                )

        # Phase 2: batched forward per group, in chunks.
        for key, items in groups.items():
            tt, bin_size, num_classes, d = key

            if tt == "clf":
                self._ensure_clf()
                model = self._clf_model
                cfg = self._clf_inference_config
            else:
                self._ensure_reg()
                model = self._reg_model
                cfg = self._reg_inference_config

            for chunk_start in range(0, len(items), self.max_batch_size):
                chunk = items[chunk_start : chunk_start + self.max_batch_size]
                bs = len(chunk)

                # Stack raw (un-standardized) tensors padded to bin_size.
                X_raw_stack = torch.stack(
                    [self._zero_pad(it[2], bin_size) for it in chunk]
                ).to(self._device, non_blocking=True)
                y_raw_stack = torch.stack(
                    [self._zero_pad(it[3], bin_size) for it in chunk]
                ).to(self._device, non_blocking=True)
                x_test_raw = torch.stack([it[4] for it in chunk]).to(
                    self._device, non_blocking=True
                )

                pad_mask = torch.zeros(
                    (bs, bin_size), dtype=torch.bool, device=self._device
                )
                real_lens = torch.empty(bs, dtype=torch.long, device=self._device)
                for j, it in enumerate(chunk):
                    real_n = it[1]
                    real_lens[j] = real_n
                    if real_n < bin_size:
                        pad_mask[j, real_n:] = True

                # Apply TabICL's "none" preprocessing (CustomStandardScaler +
                # OutlierRemover) over the whole chunk in one vectorized call.
                X_train_stack, x_test_stack = self._preprocess_none_batched(
                    X_raw_stack, x_test_raw, real_lens
                )

                if tt == "clf":
                    # Class labels (already 0/1 int as float) — no scaling.
                    y_train_stack = y_raw_stack
                    y_means = None
                    y_stds = None
                else:
                    y_train_stack, y_means, y_stds = self._standardize_y_batched(
                        y_raw_stack, real_lens
                    )

                X_full = torch.cat([X_train_stack, x_test_stack.unsqueeze(1)], dim=1)

                with torch.no_grad(), _padded_forward(pad_mask, real_lens):
                    if tt == "clf":
                        logits = model(
                            X=X_full,
                            y_train=y_train_stack,
                            return_logits=True,
                            softmax_temperature=self.softmax_temperature,
                            inference_config=cfg,
                        )  # (bs, 1, num_classes)
                        probs = torch.softmax(
                            logits.float() / self.softmax_temperature, dim=-1
                        )
                        pred_np = probs[:, 0, 1].detach().cpu().numpy()
                    else:
                        means = model.predict_stats(
                            X=X_full,
                            y_train=y_train_stack,
                            output_type="mean",
                            inference_config=cfg,
                        )  # (bs, 1)
                        pred_np = means[:, 0].float().detach().cpu().numpy()

                # NaN guard: if a prediction is non-finite (very rare numerical
                # corner case in TabICL — e.g., extreme inputs), fall back to a
                # sensible per-item value (the empirical label rate for clf, the
                # train-label mean for reg) instead of letting a NaN reach the
                # caller.
                finite_mask = np.isfinite(pred_np)
                if tt == "reg":
                    y_means_np = y_means.detach().cpu().numpy()
                    y_stds_np = y_stds.detach().cpu().numpy()

                for j, item in enumerate(chunk):
                    out_idx = item[0]
                    fallback = item[5]
                    if tt == "reg":
                        # Reg fallback is the per-item train-label mean.
                        fallback = float(y_means_np[j])
                    if not finite_mask[j]:
                        results[out_idx] = float(fallback)
                        continue
                    if tt == "reg":
                        out = float(pred_np[j]) * float(y_stds_np[j]) + float(
                            y_means_np[j]
                        )
                        if not math.isfinite(out):
                            out = float(fallback)
                        results[out_idx] = out
                    else:
                        results[out_idx] = float(pred_np[j])

        return results


# -------------------------------------------------------------------------- #
# identity_predictor: Identity predictor — returns the test feature directly as the prediction.
# -------------------------------------------------------------------------- #
@dataclass
class IdentityPredictorConfig:
    def build(self):
        return IdentityPredictor()


class IdentityPredictor(Predictor):
    """Return ``test_features[0]`` as the prediction.

    Falls back to 0.5 (clf) or 0.0 (reg) when test features are unavailable.
    """

    def predict(self, train_features, train_labels, test_features, task_type):
        if test_features is not None and len(test_features) > 0:
            return test_features[0].item()
        return 0.5 if task_type == "clf" else 0.0
