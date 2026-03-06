import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# ==========================================
# Page setup
# ==========================================
st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="📉",
    layout="centered"
)

# ==========================================
# Load model and deployment artifacts
# ==========================================
@st.cache_resource
def load_model():
    return joblib.load("models/xgb_credit_model.pkl")

@st.cache_data
def load_defaults():
    with open("artifacts/feature_defaults.json", "r") as f:
        defaults = json.load(f)

    with open("artifacts/feature_schema.json", "r") as f:
        schema = json.load(f)

    return defaults, schema

model = load_model()
defaults, schema = load_defaults()

# ==========================================
# Helper functions
# ==========================================
def safe_set(row, col, value):
    """Only set the column if it exists in training schema."""
    if col in row:
        row[col] = value

def build_input_row():
    """
    Start with default values for every feature.
    Override the important raw inputs from the UI.
    Derive engineered features automatically.
    """
    row = defaults.copy()

    # -----------------------------
    # Raw user inputs
    # -----------------------------
    safe_set(row, "loan_amnt", float(loan_amnt))
    safe_set(row, "term", float(term))
    safe_set(row, "int_rate", float(int_rate))
    safe_set(row, "installment", float(installment))
    safe_set(row, "sub_grade", sub_grade)
    safe_set(row, "emp_length", float(emp_length))
    safe_set(row, "home_ownership", home_ownership)
    safe_set(row, "annual_inc", float(annual_inc))
    safe_set(row, "verification_status", verification_status)
    safe_set(row, "purpose", purpose)
    safe_set(row, "dti", float(dti))
    safe_set(row, "delinq_2yrs", float(delinq_2yrs))
    safe_set(row, "inq_last_6mths", float(inq_last_6mths))
    safe_set(row, "open_acc", float(open_acc))
    safe_set(row, "pub_rec", float(pub_rec))
    safe_set(row, "revol_bal", float(revol_bal))
    safe_set(row, "revol_util", float(revol_util))
    safe_set(row, "total_acc", float(total_acc))
    safe_set(row, "initial_list_status", initial_list_status)
    safe_set(row, "collections_12_mths_ex_med", float(collections_12_mths_ex_med))
    safe_set(row, "application_type", application_type)
    safe_set(row, "acc_now_delinq", float(acc_now_delinq))
    safe_set(row, "tot_coll_amt", float(tot_coll_amt))
    safe_set(row, "acc_open_past_24mths", float(acc_open_past_24mths))
    safe_set(row, "bc_util", float(bc_util))
    safe_set(row, "delinq_amnt", float(delinq_amnt))
    safe_set(row, "mort_acc", float(mort_acc))
    safe_set(row, "mths_since_recent_inq", float(mths_since_recent_inq))
    safe_set(row, "num_actv_rev_tl", float(num_actv_rev_tl))
    safe_set(row, "num_tl_120dpd_2m", float(num_tl_120dpd_2m))
    safe_set(row, "num_tl_30dpd", float(num_tl_30dpd))
    safe_set(row, "num_tl_90g_dpd_24m", float(num_tl_90g_dpd_24m))
    safe_set(row, "pct_tl_nvr_dlq", float(pct_tl_nvr_dlq))
    safe_set(row, "percent_bc_gt_75", float(percent_bc_gt_75))
    safe_set(row, "pub_rec_bankruptcies", float(pub_rec_bankruptcies))
    safe_set(row, "tax_liens", float(tax_liens))
    safe_set(row, "tot_hi_cred_lim", float(tot_hi_cred_lim))
    safe_set(row, "total_bal_ex_mort", float(total_bal_ex_mort))

    # structural missing columns
    safe_set(row, "mths_since_last_delinq", float(mths_since_last_delinq))
    safe_set(row, "mths_since_recent_bc", float(mths_since_recent_bc))

    # -----------------------------
    # Derived / engineered features
    # -----------------------------
    # credit history years
    safe_set(row, "credit_history_years", float(credit_history_years))

    # ratios
    if annual_inc > 0:
        safe_set(row, "loan_to_income", float(loan_amnt / annual_inc))
        safe_set(row, "installment_to_income", float(installment / (annual_inc / 12.0)))
    else:
        safe_set(row, "loan_to_income", np.nan)
        safe_set(row, "installment_to_income", np.nan)

    if tot_hi_cred_lim > 0:
        safe_set(row, "credit_pressure_ratio", float(total_bal_ex_mort / tot_hi_cred_lim))
    else:
        safe_set(row, "credit_pressure_ratio", np.nan)

    safe_set(row, "delinq_per_year", float(delinq_2yrs / (credit_history_years + 1)))
    safe_set(row, "inq_per_year", float(inq_last_6mths / (credit_history_years + 1)))

    # -----------------------------
    # Structural missing flags
    # We assume 999 means "not applicable / no event"
    # -----------------------------
    if "mths_since_last_delinq_missing_flag" in row:
        row["mths_since_last_delinq_missing_flag"] = int(mths_since_last_delinq == 999)

    if "mths_since_recent_bc_missing_flag" in row:
        row["mths_since_recent_bc_missing_flag"] = int(mths_since_recent_bc == 999)

    if "mths_since_recent_inq_missing_flag" in row:
        row["mths_since_recent_inq_missing_flag"] = int(mths_since_recent_inq == 999)

    # -----------------------------
    # Zero-meaning flags
    # -----------------------------
    zero_flag_map = {
        "delinq_2yrs_is_zero": delinq_2yrs,
        "pub_rec_is_zero": pub_rec,
        "tax_liens_is_zero": tax_liens,
        "pub_rec_bankruptcies_is_zero": pub_rec_bankruptcies,
        "collections_12_mths_ex_med_is_zero": collections_12_mths_ex_med,
        "acc_now_delinq_is_zero": acc_now_delinq,
        "delinq_amnt_is_zero": delinq_amnt,
        "num_tl_30dpd_is_zero": num_tl_30dpd,
        "num_tl_120dpd_2m_is_zero": num_tl_120dpd_2m,
        "num_tl_90g_dpd_24m_is_zero": num_tl_90g_dpd_24m,
    }

    for flag_col, raw_value in zero_flag_map.items():
        if flag_col in row:
            row[flag_col] = int(raw_value == 0)

    # -----------------------------
    # Final dataframe in exact schema order
    # -----------------------------
    X = pd.DataFrame([[row.get(col, np.nan) for col in schema]], columns=schema)

    return X

# ==========================================
# App UI
# ==========================================
st.title("📉 Credit Risk Predictor")
st.write(
    "Estimate the probability that a loan applicant will default using the trained Lending Club credit risk model."
)

with st.expander("How this app works"):
    st.write(
        """
        This app does not rely on only a few raw inputs.
        It combines:
        - borrower-entered raw features
        - automatically derived engineered features
        - safe defaults for less important fields

        This keeps the form usable while still matching the trained model schema.
        """
    )

st.sidebar.header("Decision Settings")
decision_threshold = st.sidebar.slider(
    "Approval threshold (PD cutoff)",
    min_value=0.05,
    max_value=0.95,
    value=0.66,
    step=0.01
)

st.sidebar.markdown("**Decision rule**")
st.sidebar.write("Approve if predicted default probability < threshold")
st.sidebar.write("Manual review / reject otherwise")

# ==========================================
# Input sections
# ==========================================
st.subheader("1. Loan Details")
col1, col2 = st.columns(2)

with col1:
    loan_amnt = st.number_input("Loan Amount ($)", min_value=1000.0, max_value=40000.0, value=10000.0, step=500.0)
    term = st.selectbox("Term (months)", options=[36, 60], index=0)
    int_rate = st.number_input("Interest Rate (%)", min_value=5.0, max_value=30.0, value=12.0, step=0.1)
    installment = st.number_input("Monthly Installment ($)", min_value=50.0, max_value=2000.0, value=350.0, step=10.0)

with col2:
    purpose = st.selectbox(
        "Purpose",
        options=[
            "debt_consolidation", "credit_card", "home_improvement", "major_purchase",
            "small_business", "car", "medical", "moving", "vacation", "house", "other"
        ],
        index=0
    )
    application_type = st.selectbox("Application Type", options=["INDIVIDUAL", "JOINT"], index=0)
    initial_list_status = st.selectbox("Initial List Status", options=["f", "w"], index=0)
    sub_grade = st.text_input("Sub Grade", value="B3")

st.subheader("2. Borrower Profile")
col3, col4 = st.columns(2)

with col3:
    annual_inc = st.number_input("Annual Income ($)", min_value=10000.0, max_value=500000.0, value=60000.0, step=1000.0)
    emp_length = st.selectbox("Employment Length (years)", options=list(range(0, 11)), index=5)
    home_ownership = st.selectbox("Home Ownership", options=["RENT", "MORTGAGE", "OWN", "OTHER"], index=0)

with col4:
    verification_status = st.selectbox("Verification Status", options=["Not Verified", "Source Verified", "Verified"], index=0)
    dti = st.number_input("Debt-to-Income Ratio", min_value=0.0, max_value=40.0, value=15.0, step=0.5)
    credit_history_years = st.number_input("Credit History Length (years)", min_value=0.5, max_value=50.0, value=10.0, step=0.5)

st.subheader("3. Credit Behavior")
col5, col6 = st.columns(2)

with col5:
    delinq_2yrs = st.number_input("Delinquencies in Last 2 Years", min_value=0.0, max_value=20.0, value=0.0, step=1.0)
    inq_last_6mths = st.number_input("Inquiries in Last 6 Months", min_value=0.0, max_value=20.0, value=1.0, step=1.0)
    open_acc = st.number_input("Open Accounts", min_value=0.0, max_value=100.0, value=8.0, step=1.0)
    total_acc = st.number_input("Total Accounts", min_value=0.0, max_value=150.0, value=20.0, step=1.0)
    mort_acc = st.number_input("Mortgage Accounts", min_value=0.0, max_value=20.0, value=1.0, step=1.0)
    acc_open_past_24mths = st.number_input("Accounts Opened in Last 24 Months", min_value=0.0, max_value=30.0, value=3.0, step=1.0)
    num_actv_rev_tl = st.number_input("Active Revolving Trades", min_value=0.0, max_value=50.0, value=4.0, step=1.0)

with col6:
    pub_rec = st.number_input("Public Records", min_value=0.0, max_value=20.0, value=0.0, step=1.0)
    pub_rec_bankruptcies = st.number_input("Public Record Bankruptcies", min_value=0.0, max_value=10.0, value=0.0, step=1.0)
    tax_liens = st.number_input("Tax Liens", min_value=0.0, max_value=10.0, value=0.0, step=1.0)
    acc_now_delinq = st.number_input("Accounts Currently Delinquent", min_value=0.0, max_value=10.0, value=0.0, step=1.0)
    collections_12_mths_ex_med = st.number_input("Collections Last 12 Months", min_value=0.0, max_value=20.0, value=0.0, step=1.0)
    num_tl_30dpd = st.number_input("Trades 30+ DPD", min_value=0.0, max_value=20.0, value=0.0, step=1.0)
    num_tl_120dpd_2m = st.number_input("Trades 120+ DPD (2 Months)", min_value=0.0, max_value=20.0, value=0.0, step=1.0)
    num_tl_90g_dpd_24m = st.number_input("Trades 90+ DPD (24 Months)", min_value=0.0, max_value=20.0, value=0.0, step=1.0)

st.subheader("4. Utilization & Limits")
col7, col8 = st.columns(2)

with col7:
    revol_bal = st.number_input("Revolving Balance ($)", min_value=0.0, max_value=200000.0, value=12000.0, step=500.0)
    revol_util = st.number_input("Revolving Utilization (%)", min_value=0.0, max_value=150.0, value=45.0, step=1.0)
    bc_util = st.number_input("Bankcard Utilization (%)", min_value=0.0, max_value=150.0, value=40.0, step=1.0)
    percent_bc_gt_75 = st.number_input("Percent Bankcards > 75% Utilization", min_value=0.0, max_value=100.0, value=10.0, step=1.0)

with col8:
    tot_hi_cred_lim = st.number_input("Total High Credit Limit ($)", min_value=1000.0, max_value=500000.0, value=50000.0, step=1000.0)
    total_bal_ex_mort = st.number_input("Total Balance Excluding Mortgage ($)", min_value=0.0, max_value=300000.0, value=20000.0, step=500.0)
    tot_coll_amt = st.number_input("Total Collection Amount Ever Owed ($)", min_value=0.0, max_value=100000.0, value=0.0, step=100.0)
    delinq_amnt = st.number_input("Delinquent Amount ($)", min_value=0.0, max_value=100000.0, value=0.0, step=100.0)
    pct_tl_nvr_dlq = st.number_input("Percent Trades Never Delinquent", min_value=0.0, max_value=100.0, value=85.0, step=1.0)

st.subheader("5. Recency Inputs")
col9, col10 = st.columns(2)

with col9:
    mths_since_recent_inq = st.number_input(
        "Months Since Recent Inquiry (999 if none)",
        min_value=0.0,
        max_value=999.0,
        value=6.0,
        step=1.0
    )

with col10:
    mths_since_last_delinq = st.number_input(
        "Months Since Last Delinquency (999 if none)",
        min_value=0.0,
        max_value=999.0,
        value=999.0,
        step=1.0
    )

    mths_since_recent_bc = st.number_input(
        "Months Since Recent Bankcard (999 if none)",
        min_value=0.0,
        max_value=999.0,
        value=3.0,
        step=1.0
    )

st.divider()

if st.button("Predict Risk"):
    X_input = build_input_row()
    prob = float(model.predict_proba(X_input)[0][1])

    st.subheader("Prediction Result")
    st.metric("Predicted Probability of Default", f"{prob:.3f}")

    if prob < 0.20:
        st.success("Low Risk")
        risk_label = "Low Risk"
    elif prob < 0.40:
        st.warning("Medium Risk")
        risk_label = "Medium Risk"
    else:
        st.error("High Risk")
        risk_label = "High Risk"

    if prob < decision_threshold:
        st.info("Decision: Approve")
    else:
        st.info("Decision: Manual Review / Reject")

    st.write("---")
    st.write("### Interpretation")
    st.write(f"This applicant falls into the **{risk_label}** band based on the trained credit risk model.")
    st.write(
        f"The current business threshold is **{decision_threshold:.2f}**. "
        f"Applicants with predicted default probability below this cutoff are considered acceptable under the current policy."
    )

    with st.expander("Show engineered features used"):
        preview_cols = [
            c for c in [
                "credit_history_years",
                "loan_to_income",
                "installment_to_income",
                "credit_pressure_ratio",
                "delinq_per_year",
                "inq_per_year"
            ] if c in X_input.columns
        ]
        st.dataframe(X_input[preview_cols])

    with st.expander("Show full model input row"):
        st.dataframe(X_input)