# Credit Risk Prediction System (Lending Club)

**Live Web App:**
[https://creditloanriskapp-ijs8szkybgwes6qkuynzh5.streamlit.app/](https://creditloanriskapp-ijs8szkybgwes6qkuynzh5.streamlit.app/)

This project builds a complete **credit risk prediction system** using historical Lending Club loan data. The goal is to estimate the probability that a borrower will default on a loan and show how lenders can use model predictions to make smarter approval decisions.

The project demonstrates a full machine learning workflow — from raw data cleaning to a deployed interactive web application.

---

# Project Overview

Lenders constantly face a trade‑off:

* Approve more loans to grow revenue
* Avoid approving borrowers who may default

To support this decision process, this project trains a machine learning model that estimates the **Probability of Default (PD)** for each borrower.

These predicted probabilities can then be used to simulate different credit approval policies and measure their impact on portfolio risk.

---

# Live Demo

You can interact with the deployed model here:

**[https://creditloanriskapp-ijs8szkybgwes6qkuynzh5.streamlit.app/](https://creditloanriskapp-ijs8szkybgwes6qkuynzh5.streamlit.app/)**

The web application allows users to:

* Enter borrower financial information
* Automatically generate engineered credit features
* Estimate probability of default
* Classify the borrower into risk categories
* Simulate a credit approval decision

This demonstrates how the model could be used in a real lending environment.

---

# Dataset

**Source:** Lending Club Loan Dataset
* The raw dataset contained 2M+ rows and 160 columns 

The dataset contains historical loan records including:

* Borrower financial attributes
* Credit history information
* Loan characteristics
* Loan repayment outcomes

### Target Variable

```
0 = Fully Paid
1 = Default / Charged Off
```

Only **completed loans** were used in training to avoid label uncertainty.

---

# Data Preparation

Before training the model, several important preprocessing steps were applied.

## 1. Removing Data Leakage

Certain variables contain information that is only known **after a loan has already been issued or repaid**.

Examples include:

* total payments received
* recoveries
* outstanding principal
* last payment amount

Including these would allow the model to "cheat" by using future information.

These columns were removed to ensure the model only uses information available at **loan approval time**.

---

## 2. Handling Missing Values

Columns with more than **60% missing data** were removed.

For remaining variables:

* Numerical features were imputed using **median values**
* Categorical features were imputed using the **most frequent category**

Certain variables represent events that may never have happened (for example, months since last delinquency).

For these features, additional **missing indicator flags** were created to preserve that information.

Example:

```
mths_since_last_delinq_missing_flag
mths_since_recent_inq_missing_flag
```

---

## 3. Feature Engineering

To better capture borrower financial stress and credit behavior, several domain‑specific features were created.

Examples include:

**Loan‑to‑Income Ratio**

```
loan_to_income = loan_amount / annual_income
```

**Installment‑to‑Income Ratio**

```
installment_to_income = monthly_payment / monthly_income
```

**Credit Pressure Ratio**

```
total_balance / total_credit_limit
```

**Delinquencies Per Year**

```
delinquencies / credit_history_years
```

These engineered features help the model better capture **repayment pressure and credit usage behavior**.

---

# Modeling Approach

Two models were trained and evaluated.

## Logistic Regression

Used as a baseline model.

Advantages:

* Interpretable
* Simple to train
* Provides a benchmark for comparison

---

## XGBoost (Final Model)

Gradient boosted trees were used for the final production model.

Why XGBoost:

* Handles nonlinear relationships
* Captures complex feature interactions
* Performs well on structured tabular data

---

# Model Performance

Evaluation metrics used:

* ROC‑AUC
* PR‑AUC
* F1 Score
* KS Statistic

Final model performance:

```
ROC‑AUC ≈ 0.73
PR‑AUC ≈ 0.41
KS Statistic ≈ 0.34
```

These results show strong ability to distinguish between safe borrowers and risky borrowers.

---

# Business Decision Simulation

Instead of using a fixed classification threshold like **0.50**, the model was evaluated using a **portfolio decision framework**.

Different probability thresholds were tested to measure how approval decisions affect portfolio risk.

Example outcome:

```
Baseline default rate: 20.1%

Optimized portfolio default rate: 15.5%

Risk reduction: ~22.7%
```

This demonstrates how predictive models can help lenders:

* Reduce credit losses
* Control portfolio risk
* Allocate capital more efficiently

---

# Model Insights

Feature importance analysis identified several strong predictors of default risk.

Key drivers include:

* Loan sub‑grade
* Loan term
* Debt‑to‑income ratio
* Credit utilization
* Recent credit inquiries
* Loan‑to‑income ratio

### Interesting Observation

In the dataset, **verified borrowers sometimes show higher default rates than non‑verified borrowers**.

This occurs because lenders often verify borrowers when they already suspect higher risk. In such cases, verification status becomes correlated with **lender suspicion rather than borrower reliability**.

---

# Deployment

The trained model is deployed as a **Streamlit web application**.

Application workflow:

1. Collect borrower inputs
2. Generate engineered features automatically
3. Construct the full feature vector
4. Apply the trained preprocessing pipeline
5. Predict probability of default
6. Display risk classification and approval decision

This demonstrates a realistic end‑to‑end machine learning deployment.

---

# Tech Stack

Python
Pandas
NumPy
Scikit‑learn
XGBoost
Streamlit

---

# Repository Structure

```
app/
    streamlit_app.py

models/
    xgb_credit_model.pkl

artifacts/
    feature_schema.json
    feature_defaults.json

src/
    data processing and modeling pipeline

reports/
    evaluation outputs
```

---

# Running the Project Locally

Clone the repository:

```
git clone https://github.com/niharnandala/credit_loan_risk_app.git
cd credit_loan_risk_app
```

Install dependencies:

```
pip install -r requirements.txt
```

Run the web application:

```
streamlit run app/streamlit_app.py
```

---

# Author

**Nihar**
Machine Learning & Data Science Projects
