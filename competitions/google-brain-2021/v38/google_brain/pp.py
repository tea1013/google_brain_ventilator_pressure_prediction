import numpy as np
from numpy import ndarray


def better_than_median(prediction: ndarray, spread_threshold: float = 0.45) -> ndarray:
    spread = prediction.max(axis=0) - prediction.min(axis=0)
    return np.where(spread < spread_threshold, np.mean(prediction, axis=0), np.median(prediction, axis=0))
