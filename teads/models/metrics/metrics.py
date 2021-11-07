from abc import ABC, abstractmethod

from numpy import ndarray


class Metrics(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def execute(self, y_true: ndarray, y_pred: ndarray) -> float:
        pass
