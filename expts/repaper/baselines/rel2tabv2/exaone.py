import numpy as np
import torch

from expts.repaper.baselines.rel2tabv2.degenerate import trivial_prediction


class ExaonePredictor:
    def __init__(self, ensemble_count, device):
        self.ensemble_count = ensemble_count
        self.device = device
        self._clf = None
        self._reg = None
        # context key -> a model already fitted on it. One entry per distinct
        # context, not one per call: see _fit_shared.
        self._fitted = {}

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

    def _fit_shared(self, train_features, train_labels, task_type, X, y):
        # Rel2TabModel caches the context tensors, so a query-independent
        # retriever hands the same objects to every batch and the fit is
        # repeatable work: rel-amazon/user-churn is 351,885 test rows at eval_bs
        # 256, which is ~1375 identical fits.
        #
        # One model per context rather than one model and the last context's
        # fit: _predict_shared loops the context sizes *inside* each batch, so a
        # single fitted instance is invalidated by the next size and every size
        # refits on every batch -- 4 fits per batch, 48 for a 12-batch
        # driver-position run, where 4 would do. EXAONE's fit mutates the model,
        # so holding several fits means holding several models.
        #
        # The key holds the tensors, not their id()s, so nothing can be freed
        # and have its address reused by a different context.
        key = (train_features, train_labels, task_type)
        model = self._fitted.get(key)
        if model is not None:
            return model
        # A per-query retriever would put a distinct context in every call and
        # allocate a model for each; that is predict_batch's job, not this one.
        assert len(self._fitted) < 16, (
            f"{len(self._fitted)} distinct contexts fitted; predict_shared is "
            f"for a query-independent retriever, use predict_batch instead"
        )
        model = self._new_shared_model(task_type)
        model.fit(X, y)
        self._fitted[key] = model
        return model

    def _new_shared_model(self, task_type):
        if task_type == "clf":
            from exaonetabular import EXAONETabularClassifier

            return EXAONETabularClassifier.from_pretrained(
                device=self.device, ensemble_count=self.ensemble_count
            )
        from exaonetabular import EXAONETabularRegressor

        return EXAONETabularRegressor.from_pretrained(
            device=self.device, ensemble_count=self.ensemble_count
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

        # One fit per context, then every query in one predict_proba. Equivalent
        # to the per-query path rather than an approximation of it: predict_proba
        # runs state["preprocessor"].transform, fitted during fit on the context
        # alone, so query rows cannot influence one another.
        with torch.inference_mode(False):
            if task_type == "clf":
                model = self._fit_shared(
                    train_features, train_labels, task_type, X, y_int
                )
                proba = model.predict_proba(X_query)
                pos = int(np.flatnonzero(np.asarray(model.classes_) == 1)[0])
                return [float(v) for v in proba[:, pos]]
            model = self._fit_shared(train_features, train_labels, task_type, X, y)
            return [float(v) for v in model.predict(X_query)]
