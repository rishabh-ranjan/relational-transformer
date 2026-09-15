import numpy as np


def trivial_prediction(y_int, task_type, n_rows):
    if n_rows < 2:
        return 0.5 if task_type == "clf" else 0.0
    if task_type == "clf" and len(np.unique(y_int)) < 2:
        return float(y_int[0])
    return None
