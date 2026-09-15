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
