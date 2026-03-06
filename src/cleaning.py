# cleaning.py
# keep this file only for dataset-level cleaning steps (no modeling)

import pandas as pd
from .config import Config


def load_data(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.data_path, low_memory=False)
    return df


def define_target(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    # keep completed loans only
    df = df[df["loan_status"].isin(cfg.allowed_statuses)].copy()
    df["target"] = df["loan_status"].map(cfg.target_map)
    return df


def remove_leakage(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    # drop only columns that exist
    drop_cols = [c for c in cfg.leakage_cols if c in df.columns]
    return df.drop(columns=drop_cols)


def drop_high_missing(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    missing_rate = df.isna().mean()
    high_missing = missing_rate[missing_rate > cfg.high_missing_threshold].index.tolist()
    return df.drop(columns=high_missing, errors="ignore")


def structural_drop(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    # remove text/id/operational columns as in your notebook
    return df.drop(columns=[c for c in cfg.cols_to_drop if c in df.columns], errors="ignore")


def normalize_basic_types(df: pd.DataFrame) -> pd.DataFrame:
    # term can be "36 months" or already numeric depending on dataset version
    if "term" in df.columns:
        if df["term"].dtype == object:
            df["term"] = df["term"].astype(str).str.extract(r"(\d+)").astype(float)
        else:
            df["term"] = pd.to_numeric(df["term"], errors="coerce")

    # percent fields sometimes come with %
    if "int_rate" in df.columns and df["int_rate"].dtype == object:
        df["int_rate"] = df["int_rate"].astype(str).str.replace("%", "", regex=False)
        df["int_rate"] = pd.to_numeric(df["int_rate"], errors="coerce")

    if "revol_util" in df.columns:
        if df["revol_util"].dtype == object:
            df["revol_util"] = df["revol_util"].astype(str).str.replace("%", "", regex=False)
        df["revol_util"] = pd.to_numeric(df["revol_util"], errors="coerce")

    return df