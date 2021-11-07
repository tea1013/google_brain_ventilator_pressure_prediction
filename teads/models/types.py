from collections import namedtuple

FitResult = namedtuple("FitResult", ["model", "oof_prediction", "metrics", "score", "importance"])
