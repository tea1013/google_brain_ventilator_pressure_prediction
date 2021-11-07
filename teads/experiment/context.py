from abc import ABC, abstractmethod
from typing import Union

from numpy import ndarray
from pandas import DataFrame, Series


class Context(ABC):
    def __init__(
        self,
    ) -> None:
        pass

    @abstractmethod
    def make_oof(self, oof_prediction: Union[ndarray, Series, DataFrame]) -> DataFrame:
        pass

    @abstractmethod
    def make_submission(self, test_prediction: Union[ndarray, Series, DataFrame]) -> DataFrame:
        pass
