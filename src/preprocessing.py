# preprocessing.py
# sklearn ColumnTransformer exactly matching your notebook design

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler
from sklearn.impute import SimpleImputer


def build_preprocessor(
    X_train: pd.DataFrame,
    ordinal_cols: list[str],
    one_hot_cols: list[str],
    subgrade_order: list[str],
    verification_order: list[str],
    list_status_order: list[str],
) -> ColumnTransformer:

    numeric_cols = [c for c in X_train.columns if c not in (ordinal_cols + one_hot_cols)]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])

    ordinal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(categories=[subgrade_order, verification_order, list_status_order])),
    ])

    onehot_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("ord", ordinal_pipeline, ordinal_cols),
        ("ohe", onehot_pipeline, one_hot_cols),
    ])

    return preprocessor