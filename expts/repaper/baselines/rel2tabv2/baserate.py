class BaseRatePredictor:
    def predict_batch(self, work_items):
        results = []
        for _train_features, train_labels, _test_features, task_type in work_items:
            if len(train_labels) < 2:
                results.append(0.5 if task_type == "clf" else 0.0)
                continue
            y = train_labels.float()
            if task_type == "clf":
                results.append(float((y > 0).float().mean().item()))
            else:
                results.append(float(y.mean().item()))
        return results

    def predict_shared(self, train_features, train_labels, query_features, task_type):
        n_query = query_features.shape[0]
        if len(train_labels) < 2:
            return [0.5 if task_type == "clf" else 0.0] * n_query
        y = train_labels.float()
        if task_type == "clf":
            return [float((y > 0).float().mean().item())] * n_query
        return [float(y.mean().item())] * n_query
