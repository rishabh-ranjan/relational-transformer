import numpy as np


def _trivial(y_int, task_type, n_rows):
    if n_rows < 2:
        return 0.5 if task_type == "clf" else 0.0
    if task_type == "clf" and len(np.unique(y_int)) < 2:
        return float(y_int[0])
    return None


class ExaonePredictor:
    def __init__(self, ensemble_count, device):
        self.ensemble_count = ensemble_count
        self.device = device
        self._clf = None
        self._reg = None

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
