import numpy as np

def gini(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1 - np.sum(p ** 2)

def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return -np.sum(p * np.log2(p))
def r2_score(y_true, y_pred):
    y_true = _ensure_numpy_array(y_true)
    y_pred = _ensure_numpy_array(y_pred)
    mean_y = np.mean(y_true)
    ss_total = np.sum((y_true - mean_y) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    return r2

def _ensure_numpy_array(data):
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    return data    