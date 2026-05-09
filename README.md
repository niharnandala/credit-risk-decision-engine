# Credit Risk Scoring & Decision System

> Not just a model — a full lending decision system with real business impact.

**Live App → [Open in Streamlit](https://creditloanriskapp-ijs8szkybgwes6qkuynzh5.streamlit.app/)**

[▶️ Watch Demo Video](Credit-App-Recording.webm)

---

## What this is

A production-ready credit risk system that estimates a borrower's probability of default and translates it into an actionable lending decision.

Instead of just outputting a score:

```
Default probability = 0.68
```

The system outputs a decision:

```
High Risk → Decline or flag for manual review
Portfolio default rate reduced: 20.1% → 15.5% (~22.7% risk reduction)
```

Built on 2M+ rows of Lending Club data with a leakage-free feature pipeline, XGBoost model, and a live Streamlit app that simulates real approval decisions.

---

## Results

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.73 |
| PR-AUC | 0.41 |
| KS Statistic | 0.34 |
| Portfolio default rate | 20.1% → 15.5% (~22.7% reduction) |
| Approval rate maintained | ~80% |

---

## Why it stands out

- **2M+ rows, 160 columns** — real scale, not a toy dataset
- **Leakage-free by design** — post-origination columns explicitly removed
- **Business decision simulation** — threshold optimization tied to real lending outcomes
- **Interesting insight surfaced** — verified borrowers show higher default rates (lender-suspicion bias)
- **Full deployment** — live Streamlit app with real-time borrower scoring

---

## System Design

### 1. Data preparation
- Raw dataset: 2M+ rows, 160 columns
- Filtered to completed loans only — avoids label uncertainty from active loans
- Columns with >60% missing data removed

### 2. Leakage removal
Post-origination columns removed — these are only known *after* a loan is repaid, so including them would let the model cheat:
- Total payments received
- Recoveries
- Outstanding principal
- Last payment amount

### 3. Feature engineering
Domain-specific features created to capture borrower financial stress:

| Feature | Formula |
|---------|---------|
| Loan-to-Income Ratio | `loan_amount / annual_income` |
| Installment-to-Income Ratio | `monthly_payment / monthly_income` |
| Credit Pressure Ratio | `total_balance / total_credit_limit` |
| Delinquencies Per Year | `delinquencies / credit_history_years` |

Missing indicator flags created for sparse variables (e.g. `mths_since_last_delinq_missing_flag`) to preserve information rather than impute blindly.

### 4. Modeling
- Logistic Regression — baseline
- XGBoost — final production model (handles nonlinear relationships, complex feature interactions)

### 5. Decision layer
- Threshold not fixed at 0.5 — evaluated across the full range
- Portfolio default rate simulated at each threshold
- Final threshold selected to minimize defaults while maintaining ~80% approval rate

---

## Key Design Decisions

| Decision | Reason |
|----------|--------|
| Removed post-origination columns | Prevents data leakage — model must only use info available at approval time |
| Missing indicator flags | Preserves signal from sparse variables instead of losing it through imputation |
| XGBoost over Logistic Regression | Better captures nonlinear credit risk patterns in tabular data |
| Threshold optimization | Fixed 0.5 threshold ignores business cost asymmetry between false positives and false negatives |

---

## Model Insight: Lender-Suspicion Bias

Feature importance analysis revealed something counterintuitive:

> **Verified borrowers sometimes show higher default rates than non-verified borrowers.**

This happens because lenders tend to request verification when they already suspect higher risk. Verification status ends up correlated with lender suspicion — not borrower reliability. This was documented as a key model interpretation note, not treated as a bug.

---

## Live App

**[→ Open App](https://creditloanriskapp-ijs8szkybgwes6qkuynzh5.streamlit.app/)**

What you can do:
- Enter borrower financial information
- Get real-time probability of default
- See risk classification and approval decision
- Explore how engineered features are generated automatically from raw inputs

---

## Project Structure

```
credit-risk-decision-engine/
├── app/
│   └── streamlit_app.py        # Streamlit UI
├── src/                        # Data processing & modeling pipeline
├── models/
│   └── xgb_credit_model.pkl    # Trained model
├── artifacts/
│   ├── feature_schema.json
│   └── feature_defaults.json
├── reports/                    # Evaluation outputs
├── requirements.txt
└── README.md
```

---

## Running Locally

```bash
git clone https://github.com/niharnandala/credit_loan_risk_app.git
cd credit_loan_risk_app
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

---

## Tech Stack

`Python` · `XGBoost` · `Scikit-learn` · `Pandas` · `NumPy` · `Streamlit`

---

## Author

**Nihar Nandala**
[GitHub](https://github.com/niharnandala) · [LinkedIn](https://linkedin.com/in/niharnandala)
