# Credit Risk Modeling - Lending Club

This project predicts loan default probability using Lending Club loan data.

## What the project includes

- Data cleaning and leakage removal
- Feature engineering
- Logistic Regression baseline
- XGBoost and LightGBM modeling
- Threshold optimization
- Business impact simulation
- Decile lift analysis
- KS statistic
- Streamlit deployment

## Key results

- ROC-AUC: ~0.73
- PR-AUC: ~0.41
- KS Statistic: ~0.34
- Profit-based threshold improved portfolio quality and reduced default concentration

## Deployment

This repository includes a Streamlit app for live prediction.

## Run locally

```bash
streamlit run app/streamlit_app.py