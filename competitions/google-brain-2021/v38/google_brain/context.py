import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from teads.experiment.context import Context
from teads.util.logger import Logger

from google_brain.dataset import GoogleBrainDataset


class GoogleBrainContext(Context):
    def __init__(self, dataset: GoogleBrainDataset, sample_submission_df: DataFrame, logger: Logger) -> None:
        self.dataset = dataset
        self.sample_submission_df = sample_submission_df
        self.logger = logger

    def make_oof(self, oof_prediction: ndarray) -> DataFrame:
        df = self.dataset.train_X[["id", "breath_id", "u_out"]].copy()
        df["pressure"] = self.dataset.train_y["pressure"].copy()
        df["pred"] = oof_prediction

        return df

    def make_submission(self, test_prediction: ndarray) -> DataFrame:
        sub = self.sample_submission_df.copy()
        sub["pressure"] = test_prediction

        return sub
