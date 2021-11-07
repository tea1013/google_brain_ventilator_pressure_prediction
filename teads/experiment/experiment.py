from abc import ABC, abstractmethod
from typing import List

from numpy import ndarray
from pandas.core.frame import DataFrame
from teads.experiment.context import Context
from teads.experiment.results import ExperimentResult, TestResult, TrainResult, ValidResult
from teads.models.model_config import ModelConfig
from teads.models.model_wrapper import ModelWrapper


class ExperimentConfig(ABC):
    def __init__(self) -> None:
        pass


class Experiment(ABC):
    def __init__(self, context: Context, config: ExperimentConfig, folds: List[int] = []) -> None:
        self.context = context
        self.config = config
        self.folds = folds

    @abstractmethod
    def build_conf(self) -> ModelConfig:
        pass

    @abstractmethod
    def build_model(conf: ModelConfig) -> ModelWrapper:
        pass

    @abstractmethod
    def run(self) -> ExperimentResult:
        pass

    @abstractmethod
    def train(self) -> TrainResult:
        pass

    @abstractmethod
    def test(self) -> TestResult:
        pass

    @abstractmethod
    def valid(self) -> ValidResult:
        pass

    @abstractmethod
    def optimize(self) -> None:
        pass

    @abstractmethod
    def save_model(self) -> None:
        pass

    @abstractmethod
    def save_oof(self, oof_prediction: ndarray, score: float) -> None:
        pass

    @abstractmethod
    def save_submission(self, test_prediction: DataFrame, score: float) -> DataFrame:
        pass

    @abstractmethod
    def remake_oof_submission(self) -> None:
        pass
