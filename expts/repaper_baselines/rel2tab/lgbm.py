"""LightGBM predictor: one model fit per (row, ctx) work item.

Stock library defaults (n_estimators=100, num_leaves=31, learning_rate=0.1,
min_child_samples=20, reg_lambda=0), mirroring the stock-defaults choice the
paper's tree baseline makes; only ``n_jobs`` is set, to the job's cpu count.
"""

import numpy as np


class LGBMPredictor:
    def __init__(self, n_jobs):
        self.params = dict(n_jobs=n_jobs, verbose=-1)

    def predict(self, train_features, train_labels, test_features, task_type):
        from lightgbm import LGBMClassifier, LGBMRegressor

        if train_features is None or len(train_labels) < 2:
            return 0.5 if task_type == "clf" else 0.0

        X_train = train_features.float().cpu().numpy()
        y_train = np.nan_to_num(train_labels.float().cpu().numpy(), nan=0.0)
        X_test = test_features.float().cpu().numpy().reshape(1, -1)

        if task_type == "clf":
            y_int = (y_train > 0).astype(int)
            if len(np.unique(y_int)) < 2:
                return float(y_int[0])
            model = LGBMClassifier(**self.params)
            model.fit(X_train, y_int)
            return float(model.predict_proba(X_test)[0, 1])
        model = LGBMRegressor(**self.params)
        model.fit(X_train, y_train)
        return float(model.predict(X_test)[0])
