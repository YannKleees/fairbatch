import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from . import config as cfg


def dp_diff(y, z):
    """P(y = 1 | z = 0) - P(y = 1 | z = 1). Negative when the unprivileged group gets fewer favorable outcomes."""
    return y[z == 0].mean() - y[z == 1].mean()


def dir_(y, z):
    """Disparate impact ratio P(y = 1 | z = 0) / P(y = 1 | z = 1); 1 is parity, below 0.8 fails the four fifths rule."""
    return y[z == 0].mean() / y[z == 1].mean()


def consistency(x, y, k=cfg.K_NEIGHBORS):
    """Zemel et al. (2013): 1 - mean |y_i - mean label of the k nearest neighbours of i|.
    Neighbours are found on standardized features and exclude the point itself."""
    x = StandardScaler().fit_transform(x)
    _, idx = NearestNeighbors(n_neighbors=k + 1).fit(x).kneighbors(x)
    return 1 - np.abs(y - y[idx[:, 1:]].mean(1)).mean()


def fairness(x, y, z):
    return {"dp_diff": dp_diff(y, z), "dir": dir_(y, z), "consistency": consistency(x, y)}


def evaluate(x, y_true, y_pred, z):
    return fairness(x, y_pred, z) | {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "positive_rate": y_pred.mean(),
    }
