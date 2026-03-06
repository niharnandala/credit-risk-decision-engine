# evaluation.py
# keep evaluation logic separate so run_pipeline stays readable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


def evaluate_probs(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred)),
        "threshold": float(threshold),
        "approval_rate_if_approve_low_pd": None,  # optional later
    }


def best_f1_threshold(y_true, y_prob) -> tuple[float, float]:
    thresholds = np.linspace(0.1, 0.9, 100)
    scores = [f1_score(y_true, (y_prob >= t).astype(int)) for t in thresholds]
    idx = int(np.argmax(scores))
    return float(thresholds[idx]), float(scores[idx])


def decile_table(y_true, y_prob) -> pd.DataFrame:
    df = pd.DataFrame({"y_true": y_true.astype(int), "p_default": y_prob})
    overall = df["y_true"].mean()

    # 10 = highest risk
    df["risk_decile"] = pd.qcut(df["p_default"], 10, labels=False) + 1
    df["risk_decile"] = 11 - df["risk_decile"]

    stats = (
        df.groupby("risk_decile")
        .agg(count=("y_true", "size"), defaults=("y_true", "sum"), default_rate=("y_true", "mean"))
        .reset_index()
        .sort_values("risk_decile", ascending=False)
    )
    stats["lift"] = stats["default_rate"] / overall
    stats["cum_defaults"] = stats["defaults"].cumsum()
    stats["cum_default_capture"] = stats["cum_defaults"] / stats["defaults"].sum()
    stats["cum_population"] = stats["count"].cumsum() / stats["count"].sum()

    return stats


def profit_policy_table(X_test, y_true, y_prob, lgd: float) -> tuple[pd.DataFrame, dict]:
    # profit proxy: installment * term - loan_amnt
    loan = X_test["loan_amnt"].astype(float).values
    inst = X_test["installment"].astype(float).values
    term = X_test["term"].astype(float).values

    profit_if_paid = inst * term - loan
    loss_if_default = lgd * loan

    eval_df = pd.DataFrame({
        "y_true": y_true.astype(int),
        "p_default": y_prob,
        "profit_if_paid": profit_if_paid,
        "loss_if_default": loss_if_default,
    })

    thresholds = np.linspace(0.05, 0.95, 91)
    rows = []
    for t in thresholds:
        approved = eval_df["p_default"] < t
        if approved.sum() == 0:
            continue

        approved_df = eval_df[approved]
        realized_value = np.where(
            approved_df["y_true"] == 0,
            approved_df["profit_if_paid"],
            -approved_df["loss_if_default"],
        ).sum()

        rows.append({
            "threshold": float(t),
            "approval_rate": float(approved.mean()),
            "approved_default_rate": float(approved_df["y_true"].mean()),
            "realized_value_total": float(realized_value),
            "realized_value_per_loan": float(realized_value / approved.sum()),
        })

    policy = pd.DataFrame(rows)
    best = policy.loc[policy["realized_value_total"].idxmax()].to_dict()

    return policy, best