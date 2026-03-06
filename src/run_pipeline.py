# run_pipeline.py
# one command to run the whole project end-to-end

import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import Config
from .cleaning import (
    load_data, define_target, remove_leakage, drop_high_missing,
    structural_drop, normalize_basic_types
)
from .features import (
    add_emp_length, add_credit_history_years, add_ratio_features,
    add_structural_missing_flags_split, add_zero_meaning_flags_split
)
from .preprocessing import build_preprocessor
from .models import build_logistic, build_xgb
from .evaluation import evaluate_probs, best_f1_threshold, decile_table, profit_policy_table


def main() -> None:

    cfg = Config()

    # ---------------------------------
    # Create output folders
    # ---------------------------------
    Path("models").mkdir(exist_ok=True)
    Path("data_processed").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    Path("artifacts").mkdir(exist_ok=True)

    # ---------------------------------
    # 1) Load data + define target
    # ---------------------------------
    df = load_data(cfg)
    df = define_target(df, cfg)

    # ---------------------------------
    # 2) Cleaning
    # ---------------------------------
    df = remove_leakage(df, cfg)
    df = drop_high_missing(df, cfg)
    df = structural_drop(df, cfg)
    df = normalize_basic_types(df)

    # ---------------------------------
    # 3) Feature engineering
    # ---------------------------------
    df = add_emp_length(df)
    df = add_credit_history_years(df)
    df = add_ratio_features(df)

    # ---------------------------------
    # 4) Train / Test split
    # ---------------------------------
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        stratify=y,
        random_state=cfg.random_state
    )

    # structural missing flags (like notebook)
    X_train, X_test = add_structural_missing_flags_split(X_train, X_test, cfg)

    # zero meaning flags
    X_train, X_test = add_zero_meaning_flags_split(X_train, X_test, cfg)

    # ---------------------------------
    # 5) Feature groups
    # ---------------------------------
    ordinal_cols = ["sub_grade", "verification_status", "initial_list_status"]
    one_hot_cols = ["home_ownership", "purpose", "application_type"]

    ordinal_cols = [c for c in ordinal_cols if c in X_train.columns]
    one_hot_cols = [c for c in one_hot_cols if c in X_train.columns]

    subgrade_order = (
        sorted(X_train["sub_grade"].dropna().unique())
        if "sub_grade" in X_train.columns else []
    )

    verification_order = [
        "Not Verified",
        "Source Verified",
        "Verified"
    ]

    list_status_order = ["f", "w"]

    # ---------------------------------
    # Save feature defaults (for deployment)
    # ---------------------------------
    feature_defaults = {}

    for col in X_train.columns:

        if X_train[col].dtype == "object":
            mode_val = X_train[col].mode(dropna=True)
            feature_defaults[col] = mode_val.iloc[0] if len(mode_val) else ""

        else:
            feature_defaults[col] = float(X_train[col].median())

    feature_schema = list(X_train.columns)

    with open("artifacts/feature_defaults.json", "w") as f:
        json.dump(feature_defaults, f, indent=2)

    with open("artifacts/feature_schema.json", "w") as f:
        json.dump(feature_schema, f, indent=2)

    print(f"Saved deployment artifacts: {len(feature_schema)} features")

    # ---------------------------------
    # 6) Build preprocessing pipeline
    # ---------------------------------
    preprocessor = build_preprocessor(
        X_train=X_train,
        ordinal_cols=ordinal_cols,
        one_hot_cols=one_hot_cols,
        subgrade_order=subgrade_order,
        verification_order=verification_order,
        list_status_order=list_status_order,
    )

    # ---------------------------------
    # 7) Build models
    # ---------------------------------
    log_model = build_logistic(preprocessor, cfg)
    xgb_model = build_xgb(preprocessor, y_train, cfg)

    log_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)

    # ---------------------------------
    # 8) Evaluation
    # ---------------------------------
    log_prob = log_model.predict_proba(X_test)[:, 1]
    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]

    metrics = {
        "logistic_default_0_5": evaluate_probs(y_test.values, log_prob, threshold=0.5),
        "xgb_default_0_5": evaluate_probs(y_test.values, xgb_prob, threshold=0.5),
    }

    f1_thr, f1_val = best_f1_threshold(y_test.values, xgb_prob)

    metrics["xgb_best_f1_threshold"] = f1_thr
    metrics["xgb_best_f1"] = f1_val

    # ---------------------------------
    # 9) Business evaluation
    # ---------------------------------
    deciles = decile_table(y_test.values, xgb_prob)

    policy_table, best_policy = profit_policy_table(
        X_test,
        y_test.values,
        xgb_prob,
        lgd=cfg.LGD if hasattr(cfg, "LGD") else 0.85
    )

    # ---------------------------------
    # 10) Save artifacts
    # ---------------------------------

    # save processed dataset
    df.to_parquet(
        "data_processed/final_dataset.parquet",
        index=False
    )

    # save trained model
    joblib.dump(
        xgb_model,
        "models/xgb_credit_model.pkl"
    )

    # save reports
    deciles.to_csv(
        "reports/decile_table.csv",
        index=False
    )

    policy_table.to_csv(
        "reports/profit_policy_table.csv",
        index=False
    )

    # save metrics summary
    out = {
        "metrics": metrics,
        "best_profit_policy": best_policy,
        "notes": {
            "structural_cols": [
                c for c in cfg.structural_cols if c in X.columns
            ],
            "zero_meaning_cols": [
                c for c in cfg.zero_meaning_cols if c in X.columns
            ],
            "ordinal_cols": ordinal_cols,
            "one_hot_cols": one_hot_cols,
        },
    }

    with open("reports/metrics.json", "w") as f:
        json.dump(out, f, indent=2)

    print("DONE ✅")
    print("Saved: models/xgb_credit_model.pkl")
    print("Saved: data_processed/final_dataset.parquet")
    print("Saved: reports/metrics.json")
    print("Saved: reports/decile_table.csv")
    print("Saved: reports/profit_policy_table.csv")


if __name__ == "__main__":
    main()