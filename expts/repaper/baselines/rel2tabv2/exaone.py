import numpy as np
import torch

from expts.repaper.baselines.rel2tabv2.degenerate import trivial_prediction


class ExaonePredictor:
    def __init__(self, ensemble_count, device):
        self.ensemble_count = ensemble_count
        self.device = device
        self._clf = None
        self._reg = None
        self._fitted = None

    def _ensure_clf(self):
        if self._clf is None:
            from exaonetabular import EXAONETabularClassifier

            self._clf = EXAONETabularClassifier.from_pretrained(
                device=self.device, ensemble_count=self.ensemble_count
            )
        return self._clf

    def _ensure_reg(self):
        if self._reg is None:
            from exaonetabular import EXAONETabularRegressor

            self._reg = EXAONETabularRegressor.from_pretrained(
                device=self.device, ensemble_count=self.ensemble_count
            )
        return self._reg

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
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            x_test = np.nan_to_num(x_test, nan=0.0, posinf=0.0, neginf=0.0)

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
                    # fit raises below two classes, so the shortcut above is load
                    # bearing, not an optimisation
                    model = self._ensure_clf().fit(X, y_int)
                    proba = model.predict_proba(x_test)
                    pos = int(np.flatnonzero(np.asarray(model.classes_) == 1)[0])
                    results.append(float(proba[0, pos]))
                else:
                    model = self._ensure_reg().fit(X, y)
                    results.append(float(model.predict(x_test)[0]))
        return results

    def predict_shared(self, train_features, train_labels, query_features, task_type):
        X = np.nan_to_num(
            train_features.float().cpu().numpy().astype(np.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        y = np.nan_to_num(
            train_labels.float().cpu().numpy().astype(np.float64), nan=0.0
        )
        X_query = np.nan_to_num(
            query_features.float().cpu().numpy().astype(np.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        n_query = X_query.shape[0]

        y_int = (y > 0).astype(np.int64)
        triv = trivial_prediction(y_int, task_type, X.shape[0])
        if triv is not None:
            return [triv] * n_query

        # One fit, then every query in one predict_proba. Equivalent to the
        # per-query path rather than an approximation of it: predict_proba runs
        # state["preprocessor"].transform, fitted during fit on the context
        # alone, so query rows cannot influence one another.
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
