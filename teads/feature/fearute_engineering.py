from typing import Callable, List

import pandas as pd
from pandas import DataFrame
from teads.feature.feature_store import FeatureStore


class FeatureEngineering:
    def __init__(self, store: FeatureStore, fs: List[Callable]) -> None:
        self.store = store
        self.fs = fs

    def transform(self) -> DataFrame:
        features = []
        for f in self.fs:
            d = f(self.store)
            feature = pd.DataFrame().from_dict(d)
            features.append(feature)

        result = pd.concat(features, axis=1)

        return result
