# config.py
# one place to control thresholds, columns, and defaults

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Config:
    data_path: str = "data/loan.csv"
    random_state: int = 42
    test_size: float = 0.2

    # keep only completed outcomes
    allowed_statuses: List[str] = ("Fully Paid", "Charged Off", "Default")

    # binary mapping
    target_map = {
        "Fully Paid": 0,
        "Charged Off": 1,
        "Default": 1
    }

    # leakage fields (post-origination)
    leakage_cols: List[str] = (
        "total_pymnt",
        "total_pymnt_inv",
        "total_rec_prncp",
        "total_rec_int",
        "total_rec_late_fee",
        "recoveries",
        "collection_recovery_fee",
        "last_pymnt_d",
        "last_pymnt_amnt",
        "next_pymnt_d",
        "out_prncp",
        "out_prncp_inv",
    )

    # drop if missing > 60%
    high_missing_threshold: float = 0.60

    # structural / low value fields to drop (your notebook logic)
    cols_to_drop: List[str] = (
        "emp_title",
        "title",
        "zip_code",
        "addr_state",
        "loan_status",
        "policy_code",
        "pymnt_plan",
        "hardship_flag",
        "debt_settlement_flag",
        "disbursement_method",
        "chargeoff_within_12_mths",
        "last_credit_pull_d",
        "grade",
        "funded_amnt",
        "funded_amnt_inv",
    )

    # structural missing columns: NaN means "not applicable / never"
    structural_cols: List[str] = (
        "mths_since_last_delinq",
        "mths_since_recent_bc",
        "mths_since_recent_inq",
    )

    structural_fill_value: float = 999.0

    # zero has meaning (Option A): only add flags, don't change values
    zero_meaning_cols: List[str] = (
        "delinq_2yrs",
        "pub_rec",
        "tax_liens",
        "pub_rec_bankruptcies",
        "collections_12_mths_ex_med",
        "acc_now_delinq",
        "delinq_amnt",
        "num_tl_30dpd",
        "num_tl_120dpd_2m",
        "num_tl_90g_dpd_24m",
    )

    # model settings (match your notebook)
    xgb_params = {
        "n_estimators": 400,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "auc",
        "n_jobs": -1,
        "tree_method": "hist",
    }