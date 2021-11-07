from collections import namedtuple
from typing import List

TrainResult = namedtuple("TrainResult", ["fit_results", "oof_prediction", "metrics", "score", "importance"])

TestResult = namedtuple("TestResult", ["test_prediction"])

ValidResult = namedtuple("ValidResult", ["oof_prediction", "score"])

TrainResults = List[TrainResult]

TestResults = List[TestResult]

ExperimentResult = namedtuple("ExperimentResult", ["oof_prediction", "test_prediction", "submission_df", "metrics", "score", "importance", "time"])
