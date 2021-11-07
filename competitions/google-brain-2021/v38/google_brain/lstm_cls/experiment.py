import os
import time
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from google_brain.context import GoogleBrainContext
from google_brain.lstm_cls.lstm import GoogleBrainLSTM, GoogleBrainLSTMConfig
from google_brain.pp import better_than_median
from numpy import ndarray
from pandas import DataFrame
from teads.experiment.experiment import Experiment, ExperimentConfig
from teads.experiment.results import ExperimentResult, TestResult, TrainResult, ValidResult
from teads.models.metrics.metrics import Metrics
from teads.models.types import FitResult
from teads.util.logger import Logger
from teads.util.notification import Notification
from teads.util.timer import Timer
from tqdm import tqdm


class GoogleBrainLSTMExperimentConfig(ExperimentConfig):
    def __init__(
        self,
        exp_name: str,
        version: int,
        n_fold: int,
        metrics: Metrics,
        score: Metrics,
        file_logger: Logger,
        std_logger: Logger,
        notification: Notification,
        use_optimize_params=False,
    ) -> None:
        self.exp_name = exp_name
        self.version = version
        self.n_fold = n_fold
        self.metrics = metrics
        self.score = score
        self.file_logger = file_logger
        self.std_logger = std_logger
        self.notification = notification
        self.use_optimize_params = use_optimize_params
        self.model_confs: List[GoogleBrainLSTMConfig] = []

        os.makedirs(f"./oof/{exp_name}", exist_ok=True)
        os.makedirs(f"./models/{exp_name}", exist_ok=True)
        os.makedirs(f"./submission/{exp_name}", exist_ok=True)


class GoogleBrainLSTMExperiment(Experiment):
    def __init__(self, context: GoogleBrainContext, config: GoogleBrainLSTMExperimentConfig, folds: List[int]) -> None:
        super().__init__(context, config, folds)

    def build_conf(
        self,
        fold: int,
        categorical_features: List[str],
        continuous_features: List[str],
        unique_targets: List[float],
        target_dict: Dict,
        target_dict_inv: Dict,
    ) -> GoogleBrainLSTMConfig:
        conf = GoogleBrainLSTMConfig(
            save_dir=self.config.exp_name,
            save_file_name=f"fold-{fold}",
            model_file_type="pth",
            seed=430,
            epoch=120,
            batch_size=128,
            categorical_features=categorical_features,
            continuous_features=continuous_features,
            unique_targets=unique_targets,
            target_dict=target_dict,
            target_dict_inv=target_dict_inv,
            device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
            use_amp=True,
            num_workers=7,
            is_debug=False,
        )

        return conf

    def build_model(self, conf: GoogleBrainLSTMConfig) -> GoogleBrainLSTM:
        model = GoogleBrainLSTM(conf, self.config.metrics, self.config.score, self.config.file_logger, self.config.std_logger)

        return model

    def run(self, optimize=False) -> ExperimentResult:
        if optimize:
            self.optimize()
            return

        timer = Timer()
        timer.start()

        train_result = self.train()

        self.config.std_logger.info("Saveing models and oof ...")
        self.save_model()
        self.save_oof(train_result.oof_prediction, train_result.score)
        self.config.std_logger.info("done.")

        self.config.std_logger.info("Prediction ...")
        simple_median_test_result, pp_test_result = self.test()
        self.config.std_logger.info("done.")

        self.config.std_logger.info("Saving submission_df ...")
        submission_df = self.save_submission(simple_median_test_result.test_prediction, train_result.score, info=None)
        self.save_submission(pp_test_result.test_prediction, train_result.score, info="pp")

        timer.end()

        self.config.notification.notify(f"Experiment End. [score: {train_result.score}, time: {timer.result}]")

        return ExperimentResult(
            oof_prediction=train_result.oof_prediction,
            test_prediction=simple_median_test_result.test_prediction,
            submission_df=submission_df,
            metrics=train_result.metrics,
            score=train_result.score,
            importance=train_result.importance,
            time=timer.result,
        )

    def train(self) -> TrainResult:
        fit_results: List[FitResult] = []
        oof_prediction = np.zeros(len(self.context.dataset.train_X))
        for fold in range(self.config.n_fold):
            if fold not in self.folds:
                self.config.file_logger.info(f"Fold [{fold}] Skip.")
                self.config.std_logger.info(f"Fold [{fold}] Skip.")
                continue

            self.config.file_logger.info(f"Fold [{fold}] Start !!")
            self.config.std_logger.info(f"Fold [{fold}] Start !!")

            train_idx, valid_idx = self.context.dataset.fold_index(fold)

            conf = self.build_conf(
                fold,
                self.context.dataset.categorical_features,
                self.context.dataset.continuous_features,
                self.context.dataset.unique_targets,
                self.context.dataset.target_dict,
                self.context.dataset.target_dict_inv,
            )
            model = self.build_model(conf)
            model.build()

            train_X, train_y = self.context.dataset.train_X.iloc[train_idx].reset_index(drop=True), self.context.dataset.train_y.iloc[
                train_idx
            ].reset_index(drop=True)
            valid_X, valid_y = self.context.dataset.train_X.iloc[valid_idx].reset_index(drop=True), self.context.dataset.train_y.iloc[
                valid_idx
            ].reset_index(drop=True)

            fit_result = model.fit(train_X, train_y, valid_X, valid_y)

            self.config.notification.notify(f"Fold [{fold}] score: {fit_result.score}")

            oof_prediction[valid_idx] = fit_result.oof_prediction

            fit_results.append(fit_result)

        return TrainResult(
            fit_results,
            oof_prediction,
            self.to_metrics(oof_prediction),
            self.to_score(oof_prediction),
            self.to_importance([fit_result.importance for fit_result in fit_results]),
        )

    def test(self) -> Tuple[TestResult, TestResult]:
        test_predictions = []
        for fold in range(self.config.n_fold):
            conf = self.build_conf(
                fold,
                self.context.dataset.categorical_features,
                self.context.dataset.continuous_features,
                self.context.dataset.unique_targets,
                self.context.dataset.target_dict,
                self.context.dataset.target_dict_inv,
            )
            model = self.build_model(conf)
            model.load()

            test_prediction = model.predict(self.context.dataset.test_X)

            test_predictions.append(test_prediction)

        simple_median_test_prediction = np.median(test_predictions, axis=0)
        pp_test_prediction = better_than_median(np.array(test_predictions))

        return TestResult(test_prediction=simple_median_test_prediction), TestResult(test_prediction=pp_test_prediction)

    def valid(self) -> ValidResult:
        oof_prediction = np.zeros(len(self.context.dataset.train_X))
        for fold in range(self.config.n_fold):
            _, valid_idx = self.context.dataset.fold_index(fold)
            conf = self.build_conf(
                fold,
                self.context.dataset.categorical_features,
                self.context.dataset.continuous_features,
                self.context.dataset.unique_targets,
                self.context.dataset.target_dict,
                self.context.dataset.target_dict_inv,
            )
            model = self.build_model(conf)
            model.load()

            valid_X = self.context.dataset.train_X.iloc[valid_idx].reset_index(drop=True)
            oof_prediction[valid_idx] += model.predict(valid_X)

        return ValidResult(oof_prediction=oof_prediction, score=self.to_score(oof_prediction))

    def save_oof(self, oof_prediction: ndarray, score: float) -> None:
        oof_df = self.context.make_oof(oof_prediction)
        oof_df.to_csv(f"./oof/{self.config.exp_name}/tea_v{self.config.version}_cls_oof_cv{score:.4f}.csv", index=False)

    def save_submission(self, test_prediction: ndarray, score: float, info: Optional[str] = None) -> DataFrame:
        submission_df = self.context.make_submission(test_prediction)
        if info is None:
            submission_df.to_csv(f"./submission/{self.config.exp_name}/tea_v{self.config.version}_cls_submission_cv{score:.4f}.csv", index=False)
        else:
            submission_df.to_csv(
                f"./submission/{self.config.exp_name}/tea_v{self.config.version}_cls_{info}_submission_cv{score:.4f}.csv", index=False
            )

        return submission_df

    def remake_oof_submission(self) -> None:
        self.config.std_logger.info("Remake oof and submission ...")

        valid_result = self.valid()
        simple_median_test_result, pp_test_result = self.test()

        self.save_oof(valid_result.oof_prediction, valid_result.score)
        self.save_submission(simple_median_test_result.test_prediction, valid_result.score, info=None)
        self.save_submission(pp_test_result.test_prediction, valid_result.score, info="pp")

    def to_metrics(self, oof_prediction: ndarray) -> float:
        non_expiratory_phase_val_idx = self.context.dataset.train_X[self.context.dataset.train_X["u_out"] == 0].index
        return self.config.metrics.execute(
            self.context.dataset.train_y.iloc[non_expiratory_phase_val_idx].values, oof_prediction[non_expiratory_phase_val_idx]
        )

    def to_score(self, oof_prediction: ndarray) -> float:
        non_expiratory_phase_val_idx = self.context.dataset.train_X[self.context.dataset.train_X["u_out"] == 0].index
        return self.config.score.execute(
            self.context.dataset.train_y.iloc[non_expiratory_phase_val_idx].values, oof_prediction[non_expiratory_phase_val_idx]
        )

    def optimize(self) -> None:
        pass

    def save_model(self) -> None:
        pass

    def to_importance(self, importances: List[DataFrame]) -> DataFrame:
        pass
