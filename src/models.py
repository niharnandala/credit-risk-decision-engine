# models.py
# small helpers to build models consistently

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from .config import Config


def build_logistic(preprocessor, cfg: Config) -> Pipeline:
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=cfg.random_state,
        )),
    ])


def build_xgb(preprocessor, y_train, cfg: Config) -> Pipeline:
    # scale_pos_weight = negative / positive
    y_sum = float(np.sum(y_train))
    scale_pos_weight = (len(y_train) - y_sum) / y_sum if y_sum > 0 else 1.0

    params = dict(cfg.xgb_params)
    params["scale_pos_weight"] = scale_pos_weight
    params["random_state"] = cfg.random_state

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(**params)),
    ])