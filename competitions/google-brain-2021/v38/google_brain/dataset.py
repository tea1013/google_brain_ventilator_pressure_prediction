from abc import ABC
from collections import namedtuple
from functools import reduce
from typing import List, Tuple

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame, Series
from pandas.core.indexes.base import Index
from sklearn.preprocessing import RobustScaler, StandardScaler
from teads.feature.fearute_engineering import FeatureEngineering
from teads.feature.feature_store import FeatureStore, StoreMode, feature
from teads.util.reduce_mem_usage import reduce_mem_usage

from google_brain.feature import get_feature_transforms


class GoogleBrainDataset(ABC):
    def __init__(self) -> None:
        pass


class GogoleBrainTorchDataset(GoogleBrainDataset):
    def __init__(self, train_X: DataFrame, train_y: Series, test_X: DataFrame, folds: DataFrame) -> None:
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.folds = folds

        if "pressure" in train_y.columns:
            self.unique_targets = np.array(sorted(train_y["pressure"].unique().tolist()))
            self.target_dict = {v: i for i, v in enumerate(self.unique_targets)}
            self.target_dict_inv = {v: k for k, v in self.target_dict.items()}

        self.scaling()
        self.reduce_mem_usage()

    def scaling(self):
        scaling_target_cols = []
        for col in self.continuous_features:
            if "u_out" in col:
                continue

            scaling_target_cols.append(col)

        scaler = RobustScaler()
        all_X = np.vstack([self.train_X[scaling_target_cols].values, self.test_X[scaling_target_cols].values])
        scaler.fit(all_X)

        self.train_X[scaling_target_cols] = scaler.transform(self.train_X[scaling_target_cols].values)
        self.test_X[scaling_target_cols] = scaler.transform(self.test_X[scaling_target_cols].values)

    def fold_index(self, fold: int) -> Tuple[Index, Index]:
        train_idx = self.folds[self.folds != fold].index
        valid_idx = self.folds[self.folds == fold].index

        return train_idx, valid_idx

    @property
    def categorical_features(self):
        return ["R_C_cate"]

    @property
    def continuous_features(self):
        exclude_features = self.categorical_features + ["id", "breath_id", "group_fold", "stratified_fold"]
        features = []
        for col in self.train_X.columns:
            if not col in exclude_features:
                features.append(col)

        return features

    def reduce_mem_usage(self) -> DataFrame:
        self.train_X = reduce_mem_usage(self.train_X, verbose=False)
        self.test_X = reduce_mem_usage(self.test_X, verbose=False)


class GoogleBrainTorchDatasetCreator:
    def __init__(
        self, train_X: DataFrame, train_y: ndarray, test_X: DataFrame, features: List[str], folds: DataFrame
    ) -> None:
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.features = features
        self.folds = folds

    def make(self) -> GogoleBrainTorchDataset:
        train_store = FeatureStore(target_df=self.train_X, mode=StoreMode.Train)
        test_store = FeatureStore(target_df=self.test_X, mode=StoreMode.Test)

        feature_transforms = get_feature_transforms(self.features)

        fe_train = FeatureEngineering(train_store, feature_transforms)
        fe_test = FeatureEngineering(test_store, feature_transforms)

        train_X = fe_train.transform()
        test_X = fe_test.transform()

        train_X = train_X.fillna(0)
        test_X = test_X.fillna(0)

        train_X = reduce_mem_usage(train_X, verbose=False)
        test_X = reduce_mem_usage(test_X, verbose=False)

        return GogoleBrainTorchDataset(train_X, self.train_y, test_X, self.folds)
