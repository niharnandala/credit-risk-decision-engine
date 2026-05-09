import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

@st.cache_resource
def load_explainer(_pipeline):
    """
    Extract the XGBoost classifier from the pipeline and build a SHAP TreeExplainer.
    SHAP talks directly to the classifier — not the full sklearn pipeline.
    """
    classifier = _pipeline.named_steps["classifier"]
    explainer = shap.TreeExplainer(classifier)
    return explainer

model = load_model()
defaults, schema = load_defaults()
explainer = load_explainer(model)

# ==========================================
# Helper functions
# ==========================================
def safe_set(row, col, value):
    if col in row:
        row[col] = value

def calculate_installment(loan_amnt, int_rate, term):
    monthly_rate = (int_rate / 100) / 12
    if monthly_rate == 0:
        return loan_amnt / term
    return loan_amnt * (monthly_rate * (1 + monthly_rate) ** term) / ((1 + monthly_rate) ** term - 1)

def build_input_row():
    row = defaults.copy()
    installment = calculate_installment(loan_amnt, int_rate, term)

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
    safe_set(row, "mths_since_last_delinq", float(mths_since_last_delinq))
    safe_set(row, "mths_since_recent_bc", float(mths_since_recent_bc))
    safe_set(row, "credit_history_years", float(credit_history_years))

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

    if "mths_since_last_delinq_missing_flag" in row:
        row["mths_since_last_delinq_missing_flag"] = int(mths_since_last_delinq == 999)
    if "mths_since_recent_bc_missing_flag" in row:
        row["mths_since_recent_bc_missing_flag"] = int(mths_since_recent_bc == 999)
    if "mths_since_recent_inq_missing_flag" in row:
        row["mths_since_recent_inq_missing_flag"] = int(mths_since_recent_inq == 999)

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

    X = pd.DataFrame([[row.get(col, np.nan) for col in schema]], columns=schema)
    return X, installment


def get_shap_values(pipeline, X_input):
    """
    Step 1: Transform X through preprocessor only.
    Step 2: Run SHAP TreeExplainer on transformed data against classifier.
    Returns: shap_values (1D), feature_names (list), transformed_values (1D)
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    X_transformed = preprocessor.transform(X_input)

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

    # Clean sklearn prefixes e.g. "num__revol_util" -> "revol_util"
    feature_names = [
        name.split("__")[-1] if "__" in name else name
        for name in feature_names
    ]

    shap_vals = explainer.shap_values(X_transformed)

    # XGBoost binary: shap_values shape is (n_samples, n_features)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    vals_1d = shap_vals[0] if len(shap_vals.shape) > 1 else shap_vals

    try:
        transformed_1d = X_transformed[0]
    except Exception:
        transformed_1d = X_transformed.toarray()[0]

    return vals_1d, feature_names, transformed_1d


def plot_shap_bar(shap_values, feature_names, top_n=12):
    """Horizontal bar chart — red = increases risk, blue = decreases risk."""
    pairs = sorted(
        zip(shap_values, feature_names),
        key=lambda x: abs(x[0]),
        reverse=True
    )[:top_n]

    vals = [p[0] for p in pairs][::-1]
    names = [p[1] for p in pairs][::-1]
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in vals]

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.45)))
    bars = ax.barh(names, vals, color=colors, edgecolor="none", height=0.6)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP Value (impact on default probability)", fontsize=10)
    ax.set_title("Feature Contributions — This Prediction", fontsize=12, fontweight="bold", pad=10)

    for bar, val in zip(bars, vals):
        x_pos = val + (0.002 if val >= 0 else -0.002)
        ha = "left" if val >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center", ha=ha, fontsize=8.5)

    red_patch = mpatches.Patch(color="#d62728", label="Increases default risk ↑")
    blue_patch = mpatches.Patch(color="#1f77b4", label="Decreases default risk ↓")
    ax.legend(handles=[red_patch, blue_patch], loc="lower right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def plain_english_explanation(shap_values, feature_names, top_n=5):
    """Plain English summary of top drivers for stakeholders."""
    pairs = sorted(
        zip(shap_values, feature_names),
        key=lambda x: abs(x[0]),
        reverse=True
    )[:top_n]

    lines = []
    for val, name in pairs:
        direction = "increased" if val > 0 else "decreased"
        strength = "strongly" if abs(val) > 0.05 else "slightly"
        clean_name = name.replace("_", " ").title()
        lines.append(
            f"- **{clean_name}** {strength} {direction} the predicted default risk "
            f"(contribution: {val:+.3f})"
        )
    return "\n".join(lines)


# ==========================================
# App UI
# ==========================================
st.title("📉 Credit Risk Predictor")
st.write(
    "Estimate the probability that a loan applicant will default "
    "using the trained Lending Club credit risk model."
)

with st.expander("ℹ️ How this app works"):
    st.write("""
        - Enter borrower details across the sections below.
        - **Monthly installment is auto-calculated** from loan amount, interest rate, and term.
        - Engineered features (loan-to-income, credit pressure, etc.) are derived automatically.
        - Enter **999** for any "months since..." field if the event has never occurred.
        - After prediction, the app shows **why** the model made that decision — which features
          drove risk up or down, with exact SHAP contribution values.
    """)

with st.expander("📋 Input Guide — Valid Ranges"):
    st.markdown("""
    | Field | Valid Range | Typical Value |
    |-------|-------------|---------------|
    | Loan Amount | $1,000 – $40,000 | $10,000 |
    | Interest Rate | 5% – 31% | 12% |
    | Annual Income | $10,000 – $500,000 | $60,000 |
    | DTI | 0 – 45 | 15 |
    | Revolving Utilization | 0% – 150% | 45% |
    | Revolving Balance | $0 – $150,000 | $12,000 |
    | Open Accounts | 0 – 90 | 8 |
    | Total Accounts | 0 – 150 | 20 |
    | Delinquencies (2yr) | 0 – 20 | 0 |
    | Inquiries (6mo) | 0 – 20 | 1 |
    | Credit History | 0.5 – 50 years | 10 |
    | Sub Grade | A1 – G5 | B3 |
    | Months Since fields | 0 – 200, or 999 if never | — |
    """)

# ==========================================
# Sidebar
# ==========================================
st.sidebar.header("⚙️ Decision Settings")
decision_threshold = st.sidebar.slider(
    "Approval threshold (PD cutoff)",
    min_value=0.05, max_value=0.95, value=0.66, step=0.01,
    help="Approve if predicted PD < threshold."
)
st.sidebar.markdown("**Decision rule**")
st.sidebar.write("✅ Approve if predicted PD < threshold")
st.sidebar.write("⚠️ Manual review / reject otherwise")
st.sidebar.markdown("---")
top_n_features = st.sidebar.slider(
    "Top N features to explain", min_value=5, max_value=20, value=12, step=1
)

# ==========================================
# Section 1: Loan Details
# ==========================================
st.subheader("1. Loan Details")
col1, col2 = st.columns(2)

with col1:
    loan_amnt = st.number_input("Loan Amount ($)", min_value=1000.0, max_value=40000.0, value=10000.0, step=500.0,
        help="Lending Club range: $1,000 – $40,000")
    term = st.selectbox("Term (months)", options=[36, 60], index=0,
        help="Only 36 or 60 month terms available")
    int_rate = st.number_input("Interest Rate (%)", min_value=5.0, max_value=31.0, value=12.0, step=0.1,
        help="Range: 5% – 31%. Higher sub-grades carry higher rates.")

with col2:
    purpose = st.selectbox("Purpose",
        options=["debt_consolidation","credit_card","home_improvement","major_purchase",
                 "small_business","car","medical","moving","vacation","house","other"], index=0)
    application_type = st.selectbox("Application Type", options=["INDIVIDUAL", "JOINT"], index=0)
    initial_list_status = st.selectbox("Initial List Status", options=["f", "w"], index=0,
        help="'w' = whole loan, 'f' = fractional.")
    sub_grade_options = [
        "A1","A2","A3","A4","A5","B1","B2","B3","B4","B5",
        "C1","C2","C3","C4","C5","D1","D2","D3","D4","D5",
        "E1","E2","E3","E4","E5","F1","F2","F3","F4","F5",
        "G1","G2","G3","G4","G5"
    ]
    sub_grade = st.selectbox("Sub Grade", options=sub_grade_options, index=7,
        help="A1 = lowest risk, G5 = highest risk.")

_installment_preview = calculate_installment(loan_amnt, int_rate, term)
st.info(f"📊 Auto-calculated Monthly Installment: **${_installment_preview:.2f}**  \n"
        f"_(Based on ${loan_amnt:,.0f} at {int_rate}% over {term} months)_")

# ==========================================
# Section 2: Borrower Profile
# ==========================================
st.subheader("2. Borrower Profile")
col3, col4 = st.columns(2)

with col3:
    annual_inc = st.number_input("Annual Income ($)", min_value=10000.0, max_value=500000.0, value=60000.0, step=1000.0,
        help="Gross annual income. Typical: $40,000 – $80,000.")
    emp_length = st.selectbox("Employment Length (years)", options=list(range(0, 11)), index=5,
        help="0 = less than 1 year, 10 = 10+ years")
    home_ownership = st.selectbox("Home Ownership", options=["RENT","MORTGAGE","OWN","OTHER"], index=0)

with col4:
    verification_status = st.selectbox("Verification Status",
        options=["Not Verified","Source Verified","Verified"], index=0,
        help="Note: Verified borrowers can show higher default rates (lender-suspicion bias).")
    dti = st.number_input("Debt-to-Income Ratio", min_value=0.0, max_value=45.0, value=15.0, step=0.5,
        help="Monthly debt / monthly income × 100. Typical: 10–25. High risk: >35.")
    credit_history_years = st.number_input("Credit History Length (years)", min_value=0.5, max_value=50.0, value=10.0, step=0.5,
        help="Years since earliest credit line. Typical: 5–20.")

# ==========================================
# Section 3: Credit Behavior
# ==========================================
st.subheader("3. Credit Behavior")
col5, col6 = st.columns(2)

with col5:
    delinq_2yrs = st.number_input("Delinquencies in Last 2 Years", min_value=0.0, max_value=20.0, value=0.0, step=1.0,
        help="30+ day delinquencies. Most borrowers: 0. Risk flag: ≥2.")
    inq_last_6mths = st.number_input("Inquiries in Last 6 Months", min_value=0.0, max_value=20.0, value=1.0, step=1.0,
        help="Hard credit inquiries. Typical: 0–2. High: ≥5.")
    open_acc = st.number_input("Open Accounts", min_value=0.0, max_value=90.0, value=8.0, step=1.0,
        help="Open credit lines. Typical: 5–15.")
    total_acc = st.number_input("Total Accounts", min_value=0.0, max_value=150.0, value=20.0, step=1.0,
        help="Total credit lines ever. Typical: 10–35.")
    mort_acc = st.number_input("Mortgage Accounts", min_value=0.0, max_value=20.0, value=1.0, step=1.0)
    acc_open_past_24mths = st.number_input("Accounts Opened in Last 24 Months", min_value=0.0, max_value=30.0, value=3.0, step=1.0)
    num_actv_rev_tl = st.number_input("Active Revolving Trades", min_value=0.0, max_value=50.0, value=4.0, step=1.0,
        help="Active revolving credit lines. Typical: 2–8.")

with col6:
    pub_rec = st.number_input("Public Records", min_value=0.0, max_value=20.0, value=0.0, step=1.0,
        help="Derogatory public records. Most borrowers: 0.")
    pub_rec_bankruptcies = st.number_input("Public Record Bankruptcies", min_value=0.0, max_value=10.0, value=0.0, step=1.0)
    tax_liens = st.number_input("Tax Liens", min_value=0.0, max_value=10.0, value=0.0, step=1.0)
    acc_now_delinq = st.number_input("Accounts Currently Delinquent", min_value=0.0, max_value=10.0, value=0.0, step=1.0)
    collections_12_mths_ex_med = st.number_input("Collections Last 12 Months (excl. medical)", min_value=0.0, max_value=20.0, value=0.0, step=1.0)
    num_tl_30dpd = st.number_input("Trades 30+ DPD", min_value=0.0, max_value=20.0, value=0.0, step=1.0)
    num_tl_120dpd_2m = st.number_input("Trades 120+ DPD (last 2 months)", min_value=0.0, max_value=20.0, value=0.0, step=1.0)
    num_tl_90g_dpd_24m = st.number_input("Trades 90+ DPD (last 24 months)", min_value=0.0, max_value=20.0, value=0.0, step=1.0)

# ==========================================
# Section 4: Utilization & Limits
# ==========================================
st.subheader("4. Utilization & Limits")
col7, col8 = st.columns(2)

with col7:
    revol_bal = st.number_input("Revolving Balance ($)", min_value=0.0, max_value=150000.0, value=12000.0, step=500.0,
        help="Total revolving balance. Typical: $5,000 – $20,000.")
    revol_util = st.number_input("Revolving Utilization (%)", min_value=0.0, max_value=150.0, value=45.0, step=1.0,
        help="Revolving balance / limit. High risk: >75%.")
    bc_util = st.number_input("Bankcard Utilization (%)", min_value=0.0, max_value=150.0, value=40.0, step=1.0,
        help="Bankcard balance / limit. Typical: 20–60%.")
    percent_bc_gt_75 = st.number_input("Percent Bankcards > 75% Utilization", min_value=0.0, max_value=100.0, value=10.0, step=1.0)

with col8:
    tot_hi_cred_lim = st.number_input("Total High Credit Limit ($)", min_value=1000.0, max_value=500000.0, value=50000.0, step=1000.0,
        help="Sum of all high credit limits. Typical: $20,000 – $100,000.")
    total_bal_ex_mort = st.number_input("Total Balance Excluding Mortgage ($)", min_value=0.0, max_value=300000.0, value=20000.0, step=500.0)
    tot_coll_amt = st.number_input("Total Collection Amount Ever ($)", min_value=0.0, max_value=100000.0, value=0.0, step=100.0,
        help="Most borrowers: $0.")
    delinq_amnt = st.number_input("Delinquent Amount ($)", min_value=0.0, max_value=100000.0, value=0.0, step=100.0,
        help="Most borrowers: $0.")
    pct_tl_nvr_dlq = st.number_input("Percent Trades Never Delinquent", min_value=0.0, max_value=100.0, value=85.0, step=1.0,
        help="Good borrowers: >90%.")

# ==========================================
# Section 5: Recency
# ==========================================
st.subheader("5. Recency")
st.caption("Enter **999** for any field if the event has never occurred.")
col9, col10 = st.columns(2)

with col9:
    mths_since_recent_inq = st.number_input("Months Since Recent Inquiry", min_value=0.0, max_value=999.0, value=6.0, step=1.0,
        help="Range: 0–25. Enter 999 if no inquiries ever.")

with col10:
    mths_since_last_delinq = st.number_input("Months Since Last Delinquency", min_value=0.0, max_value=999.0, value=999.0, step=1.0,
        help="Range: 0–180. Enter 999 if never delinquent.")
    mths_since_recent_bc = st.number_input("Months Since Recent Bankcard", min_value=0.0, max_value=999.0, value=3.0, step=1.0,
        help="Range: 0–60. Enter 999 if never.")

st.divider()

# ==========================================
# Predict + Explain
# ==========================================
if st.button("🔍 Predict Risk", type="primary"):
    X_input, installment_used = build_input_row()
    prob = float(model.predict_proba(X_input)[0][1])

    # Results
    st.subheader("Prediction Result")
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.metric("Probability of Default", f"{prob:.3f}")
    with col_r2:
        st.metric("Monthly Installment", f"${installment_used:.2f}")

    if prob < 0.20:
        st.success("🟢 Low Risk")
        risk_label = "Low Risk"
    elif prob < 0.40:
        st.warning("🟡 Moderate Risk")
        risk_label = "Moderate Risk"
    else:
        st.error("🔴 High Risk")
        risk_label = "High Risk"

    if prob < decision_threshold:
        st.info(f"✅ Decision: **Approve** (PD {prob:.3f} < threshold {decision_threshold:.2f})")
    else:
        st.info(f"⚠️ Decision: **Manual Review / Reject** (PD {prob:.3f} ≥ threshold {decision_threshold:.2f})")

    # SHAP Explanation
    st.divider()
    st.subheader("🔍 Why this prediction?")
    st.caption(
        "Each bar shows how much a feature pushed the default probability **up** (red) "
        "or **down** (blue) for this specific borrower."
    )

    try:
        shap_vals, feat_names, _ = get_shap_values(model, X_input)

        fig = plot_shap_bar(shap_vals, feat_names, top_n=top_n_features)
        st.pyplot(fig)
        plt.close()

        # Stakeholder summary
        st.subheader("📋 Stakeholder Summary")
        st.markdown(
            f"This borrower has a predicted default probability of **{prob:.1%}**, "
            f"classified as **{risk_label}**.\n\n"
            f"The top factors driving this prediction are:\n\n"
            + plain_english_explanation(shap_vals, feat_names, top_n=5)
        )

        # Full SHAP table
        with st.expander("📊 Full feature contribution table"):
            shap_df = pd.DataFrame({
                "Feature": feat_names,
                "SHAP Value": shap_vals
            }).sort_values("SHAP Value", key=abs, ascending=False).reset_index(drop=True)
            shap_df["Direction"] = shap_df["SHAP Value"].apply(
                lambda x: "↑ Increases risk" if x > 0 else "↓ Decreases risk"
            )
            st.dataframe(shap_df, use_container_width=True)

    except Exception as e:
        st.warning(f"Explanation could not be generated: {e}")
        st.write("Prediction above is still valid.")

    # Engineered features
    with st.expander("⚙️ Auto-derived engineered features"):
        preview_cols = [c for c in [
            "credit_history_years", "loan_to_income", "installment_to_income",
            "credit_pressure_ratio", "delinq_per_year", "inq_per_year"
        ] if c in X_input.columns]
        st.dataframe(X_input[preview_cols])

    with st.expander("🔬 Full model input row"):
        st.dataframe(X_input)