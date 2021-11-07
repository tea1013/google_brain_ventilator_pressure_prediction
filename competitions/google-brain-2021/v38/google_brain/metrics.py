from numpy import ndarray
from sklearn.metrics import mean_absolute_error
from teads.models.metrics.metrics import Metrics


class MAE(Metrics):
    def __init__(self) -> None:
        name = "mae"
        super().__init__(name)

    def execute(self, y_true: ndarray, y_pred: ndarray) -> float:
        return mean_absolute_error(y_true, y_pred)
