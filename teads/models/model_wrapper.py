from abc import ABC, abstractmethod
from typing import Dict, Union

from numpy import ndarray
from pandas import DataFrame, Series
from teads.models.metrics.metrics import Metrics
from teads.models.model_config import ModelConfig
from teads.models.types import FitResult
from teads.util.logger import Logger


class ModelWrapper(ABC):
    def __init__(self, config: ModelConfig, metrics: Metrics, score: Metrics, file_logger: Logger, std_logger: Logger) -> None:
        self.config = config
        self.metrics = metrics
        self.score = score
        self.file_logger = file_logger
        self.std_logger = std_logger

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def fit(
        self,
        X_train: Union[DataFrame, ndarray],
        y_train: Union[Series, ndarray],
        X_valid: Union[DataFrame, ndarray],
        y_valid: Union[Series, ndarray],
        **kwargs,
    ) -> FitResult:
        pass

    @abstractmethod
    def predict(self, X_test: Union[DataFrame, ndarray], **kwargs) -> ndarray:
        pass

    @abstractmethod
    def optimize(
        self,
        X_train: Union[DataFrame, ndarray],
        y_train: Union[Series, ndarray],
        X_valid: Union[DataFrame, ndarray],
        y_valid: Union[Series, ndarray],
        direction: str,
        n_trials: int,
        **kwargs,
    ) -> Dict:
        pass

    @abstractmethod
    def save_model(self):
        pass
