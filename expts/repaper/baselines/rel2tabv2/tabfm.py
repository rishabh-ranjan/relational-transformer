import numpy as np
import torch


def _trivial(y_int, task_type, n_rows):
    if n_rows < 2:
        return 0.5 if task_type == "clf" else 0.0
    if task_type == "clf" and len(np.unique(y_int)) < 2:
        return float(y_int[0])
    return None


class TabFMPredictor:
    def __init__(self, backend, device):
        assert backend in ("pytorch", "jax"), f"unknown tabfm backend {backend!r}"
        self.backend = backend
        self.device = device
        self._clf = None
        self._reg = None

    def _load(self, model_type):
        import tabfm

        loader = (
            tabfm.tabfm_v1_0_0_pytorch
            if self.backend == "pytorch"
            else tabfm.tabfm_v1_0_0_jax
        )
        return loader.load(model_type=model_type, device=self.device)

    def _ensure_clf(self):
        if self._clf is None:
            from tabfm import TabFMClassifier

            self._clf = TabFMClassifier(model=self._load("classification"))
        return self._clf

    def _ensure_reg(self):
        if self._reg is None:
            from tabfm import TabFMRegressor

            self._reg = TabFMRegressor(model=self._load("regression"))
        return self._reg

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
            triv = _trivial(y_int, task_type, X.shape[0])
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
