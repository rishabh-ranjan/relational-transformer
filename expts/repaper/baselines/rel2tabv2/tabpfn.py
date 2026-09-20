from pathlib import Path

import numpy as np
import torch

from expts.repaper.baselines.rel2tabv2.degenerate import trivial_prediction

HF_REPO = "Prior-Labs/tabpfn_3_5"
# tabpfn.model_loading.ModelSource.get_v3_5().default_filename: the base
# TabPFN-3.5, one multitask checkpoint backing both estimator types. Not
# tabpfn-v3.5-fast-20260909.safetensors and not the _multiclass one, which every
# task here is binary or regression anyway. The filename is load bearing rather
# than cosmetic: _resolve_model_version reads the architecture version out of it
# ("v3.5-fast" first, so this name cannot be mistaken for the fast variant), and
# it is the only thing that pins the version when model_path is given.
CHECKPOINT = "tabpfn-v3.5-20260909.safetensors"


class TabPFNPredictor:
    def __init__(self, n_estimators, fit_mode, checkpoint_dir, device, nan_as_missing):
        assert n_estimators == "auto" or (
            isinstance(n_estimators, int) and n_estimators > 0
        ), f"tabpfn n_estimators must be 'auto' or a positive int, got {n_estimators!r}"
        assert fit_mode in ("low_memory", "fit_preprocessors", "fit_with_cache"), (
            f"unknown tabpfn fit_mode {fit_mode!r}"
        )
        self.n_estimators = n_estimators
        self.fit_mode = fit_mode
        self.nan_as_missing = nan_as_missing
        self.checkpoint_dir = Path(checkpoint_dir).expanduser()
        self.device = device
        self._clf = None
        self._reg = None
        self._fitted = None

    def _model_path(self):
        # TabPFN would download a missing checkpoint to this path itself, which
        # on a compute node without internet surfaces as a download traceback
        # from inside fit(). Assert instead, naming the one command that fixes
        # it, the same way TabICLBatchedPredictor does.
        path = self.checkpoint_dir / CHECKPOINT
        assert path.exists(), (
            f"TabPFN checkpoint {path} not found; fetch it once with "
            f"`pixi run python -m expts.repaper.baselines.fetch_tabpfn`"
        )
        return path

    def _ensure_clf(self):
        if self._clf is None:
            from tabpfn import TabPFNClassifier

            self._clf = TabPFNClassifier(
                n_estimators=self.n_estimators,
                model_path=self._model_path(),
                device=self.device,
                fit_mode=self.fit_mode,
            )
        return self._clf

    def _ensure_reg(self):
        if self._reg is None:
            from tabpfn import TabPFNRegressor

            self._reg = TabPFNRegressor(
                n_estimators=self.n_estimators,
                model_path=self._model_path(),
                device=self.device,
                fit_mode=self.fit_mode,
            )
        return self._reg

    def _features(self, a):
        # TabPFN declares allow_nan and validates X with
        # ensure_all_finite=False, so NaN is its missing-value encoding, and
        # its preprocessing is written around that (KDITransformerWithNaN
        # restores the mask after transforming). A featurizer whose NaN means
        # "this cell does not exist in the database" therefore passes it
        # through; zero-filling would assert the cell exists and is zero.
        # +-inf is never meaningful and is clamped either way. Labels are not
        # cleaned here: y is validated with ensure_all_finite=True.
        return np.nan_to_num(
            a,
            nan=np.nan if self.nan_as_missing else 0.0,
            posinf=0.0,
            neginf=0.0,
        )

    def _already_fitted(self, train_features, train_labels, task_type):
        # Rel2TabModel caches the context tensors, so a query-independent
        # retriever hands the same objects to every batch. rel-amazon/user-churn
        # is 351,885 test rows at eval_bs 256, so refitting per call would be
        # ~1375 identical fits. The fitted tensors are held here rather than
        # their id()s, so nothing can be freed and have its address reused by a
        # different context.
        prev = self._fitted
        return (
            prev is not None
            and prev[0] is train_features
            and prev[1] is train_labels
            and prev[2] == task_type
        )

    def predict_batch(self, work_items):
        results = []
        for train_features, train_labels, test_features, task_type in work_items:
            X = train_features.float().cpu().numpy().astype(np.float64)
            y = np.nan_to_num(
                train_labels.float().cpu().numpy().astype(np.float64), nan=0.0
            )
            x_test = (
                test_features.float().cpu().numpy().astype(np.float64).reshape(1, -1)
            )
            X = self._features(X)
            x_test = self._features(x_test)

            y_int = (y > 0).astype(np.int64)
            triv = trivial_prediction(y_int, task_type, X.shape[0])
            if triv is not None:
                results.append(triv)
                continue

            # rt.eval.evaluator.evaluate_raw wraps the whole loop in
            # torch.inference_mode, so tensors made here would be inference
            # tensors, which carry no version counter and blow up inside the
            # model (183466: "Inference tensors do not track version
            # counter"). The inputs are numpy, so nothing crosses the
            # boundary; the load has to be inside too, or the parameters are
            # inference tensors themselves.
            with torch.inference_mode(False):
                if task_type == "clf":
                    model = self._ensure_clf().fit(X, y_int)
                    proba = model.predict_proba(x_test)
                    pos = int(np.flatnonzero(np.asarray(model.classes_) == 1)[0])
                    results.append(float(proba[0, pos]))
                else:
                    model = self._ensure_reg().fit(X, y)
                    results.append(float(model.predict(x_test)[0]))
        return results

    def predict_shared(self, train_features, train_labels, query_features, task_type):
        X = self._features(train_features.float().cpu().numpy().astype(np.float64))
        y = np.nan_to_num(
            train_labels.float().cpu().numpy().astype(np.float64), nan=0.0
        )
        X_query = self._features(
            query_features.float().cpu().numpy().astype(np.float64)
        )
        n_query = X_query.shape[0]

        y_int = (y > 0).astype(np.int64)
        triv = trivial_prediction(y_int, task_type, X.shape[0])
        if triv is not None:
            return [triv] * n_query

        # Exact, not a batched approximation: in tabpfn_v3_5's row attention K
        # and V are projected from x_BRE[:, :single_eval_pos] alone -- "self
        # attention where k/v are restricted to train rows" -- so a query row
        # attends to the context and to nothing else, whatever else is in the
        # same forward pass. This is the opposite of TabICL, whose row attention
        # is bidirectional over the whole sequence and whose predict_shared
        # therefore has to fall back to per-query. It is also why tabpfn ships
        # max_batched_test_rows chunking at all.
        fitted = self._already_fitted(train_features, train_labels, task_type)
        key = (train_features, train_labels, task_type)
        with torch.inference_mode(False):
            if task_type == "clf":
                model = self._ensure_clf()
                if not fitted:
                    model.fit(X, y_int)
                    self._fitted = key
                proba = model.predict_proba(X_query)
                pos = int(np.flatnonzero(np.asarray(model.classes_) == 1)[0])
                return [float(v) for v in proba[:, pos]]
            model = self._ensure_reg()
            if not fitted:
                model.fit(X, y)
                self._fitted = key
            return [float(v) for v in model.predict(X_query)]
