# features.py
# everything here is “approval-time safe” feature engineering

import numpy as np
import pandas as pd
from .config import Config


_EMP_LENGTH_MAP = {
    "< 1 year": 0,
    "1 year": 1,
    "2 years": 2,
    "3 years": 3,
    "4 years": 4,
    "5 years": 5,
    "6 years": 6,
    "7 years": 7,
    "8 years": 8,
    "9 years": 9,
    "10+ years": 10,
}


def add_emp_length(df: pd.DataFrame) -> pd.DataFrame:
    if "emp_length" in df.columns:
        df["emp_length"] = df["emp_length"].map(_EMP_LENGTH_MAP)
    return df


def add_credit_history_years(df: pd.DataFrame) -> pd.DataFrame:
    # expects issue_d and earliest_cr_line to exist in df (don’t re-read CSV)
    if "issue_d" in df.columns and "earliest_cr_line" in df.columns:
        df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
        df["earliest_cr_line"] = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")

        df["credit_history_years"] = (df["issue_d"] - df["earliest_cr_line"]).dt.days / 365.0

        # raw date columns not needed after feature creation
        df = df.drop(columns=["issue_d", "earliest_cr_line"], errors="ignore")
    return df


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    # loan_to_income
    if "loan_amnt" in df.columns and "annual_inc" in df.columns:
        df["loan_to_income"] = np.where(
            df["annual_inc"] > 0,
            df["loan_amnt"] / df["annual_inc"],
            np.nan
        )

    # installment_to_income
    if "installment" in df.columns and "annual_inc" in df.columns:
        df["installment_to_income"] = np.where(
            df["annual_inc"] > 0,
            df["installment"] / (df["annual_inc"] / 12.0),
            np.nan
        )

    # credit_pressure_ratio
    if "tot_hi_cred_lim" in df.columns and "total_bal_ex_mort" in df.columns:
        df["credit_pressure_ratio"] = np.where(
            df["tot_hi_cred_lim"] > 0,
            df["total_bal_ex_mort"] / df["tot_hi_cred_lim"],
            np.nan
        )

    # intensity rates (guard against 0/NaN history length)
    if "credit_history_years" in df.columns:
        denom = df["credit_history_years"].replace(0, np.nan)

        if "delinq_2yrs" in df.columns:
            df["delinq_per_year"] = df["delinq_2yrs"] / (denom + 1)

        if "inq_last_6mths" in df.columns:
            df["inq_per_year"] = df["inq_last_6mths"] / (denom + 1)

    return df


def add_structural_missing_flags_split(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    cfg: Config
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # exactly like your notebook: do this after split
    cols = [c for c in cfg.structural_cols if c in X_train.columns]

    for c in cols:
        X_train[f"{c}_missing_flag"] = X_train[c].isna().astype(int)
        X_test[f"{c}_missing_flag"] = X_test[c].isna().astype(int)

        X_train[c] = X_train[c].fillna(cfg.structural_fill_value)
        X_test[c] = X_test[c].fillna(cfg.structural_fill_value)

    return X_train, X_test


def add_zero_meaning_flags_split(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    cfg: Config
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Option A: flag-only, keep original zeros
    cols = [c for c in cfg.zero_meaning_cols if c in X_train.columns]

    for c in cols:
        X_train[f"{c}_is_zero"] = (X_train[c] == 0).astype(int)
        X_test[f"{c}_is_zero"] = (X_test[c] == 0).astype(int)

    return X_train, X_test