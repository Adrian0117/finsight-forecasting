# FinSight: Financial Forecasting System

> A multi-task machine learning system built on 2.2M Lending Club loan records — covering default risk, borrower churn, loan volume forecasting, and credit demand segmentation.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## Overview

FinSight demonstrates an end-to-end ML pipeline applied to real-world financial data. Using a single dataset (Lending Club accepted loans), the project tackles four distinct business problems — each with its own feature engineering, model comparison, and evaluation strategy.

| Module                    | Business Question                             | Task Type               | Champion Model    |
| ------------------------- | --------------------------------------------- | ----------------------- | ----------------- |
| 1. Loan Default Risk      | Which borrowers are likely to default?        | Binary classification   | XGBoost           |
| 2. Borrower Churn         | Which customers won't return after repayment? | Binary classification   | XGBoost           |
| 3. Loan Volume Forecast   | How much will be funded next quarter?         | Time series regression  | Prophet + XGBoost |
| 4. Credit Demand Forecast | What is demand per loan grade next month?     | Multi-output regression | XGBoost           |

---

## Results Summary

| Module                 | Metric   | Linear Regression | Random Forest | XGBoost  |
| ---------------------- | -------- | :---------------: | :-----------: | :------: |
| Loan Default Risk      | AUC-ROC  |       0.71        |     0.88      | **0.91** |
| Borrower Churn         | F1 Score |       0.64        |     0.75      | **0.78** |
| Loan Volume Forecast   | MAPE     |       12.3%       |     7.8%      | **5.9%** |
| Credit Demand Forecast | RMSE     |       184.2       |     97.6      | **81.4** |

> Results are approximate and will vary based on train/test split and hyperparameter tuning.

---

## Project Structure

```
finsight-forecasting/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                        # Lending Club accepted_2007_to_2018Q4.csv
│   └── processed/                  # Cleaned data per module
├── notebooks/
│   ├── 01_loan_default_risk.ipynb
│   ├── 02_borrower_churn.ipynb
│   ├── 03_loan_volume_forecast.ipynb
│   └── 04_credit_demand_forecast.ipynb
├── src/
│   ├── data_pipeline.py            # Shared cleaning functions
│   ├── feature_engineering.py      # Per-module feature construction
│   └── model_utils.py              # Training, evaluation, model saving
├── models/                         # Saved .pkl model files
├── dashboard/
│   └── app.py                      # Streamlit app (4 tabs)
└── reports/
    └── model_comparison.md         # Full model comparison report
```

---

## Dataset

**Source:** [Lending Club Loan Data — Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

Download the `accepted_2007_to_2018Q4.csv` file and place it in `data/raw/`.

- 2.26 million loan records
- 150+ features including loan grade, interest rate, income, DTI, employment length
- Date range: 2007–2018

> The `rejected` CSV is not used in this project — it lacks loan outcome data required for all four modules.

---

## Module Details

### Module 1 — Loan Default Risk

Predicts whether a borrower will default (Charged Off) or fully repay their loan.

- **Target:** `loan_status` → Fully Paid = 0, Charged Off = 1
- **Key features:** `grade`, `int_rate`, `annual_inc`, `dti`, `delinq_2yrs`, `revol_util`
- **Challenge:** Class imbalance (~80% repaid, ~20% default) — handled with SMOTE
- **Explainability:** SHAP values for top feature contributions

### Module 2 — Borrower Churn

Predicts which borrowers will not take a second loan within 12 months of repayment.

- **Target:** Engineered from `member_id` and `issue_d` — churn = 1 if no second loan within 12 months
- **Key features:** `purpose`, `emp_length`, `home_ownership`, `loan_amnt`, `term`
- **Challenge:** Label must be constructed manually from raw data

### Module 3 — Loan Volume Forecast

Forecasts the total loan amount funded per month for the next 3 months.

- **Target:** Monthly aggregation of `funded_amnt` by `issue_d`
- **Key features:** Month, quarter, lag features (1/3/6 month), rolling averages, interest rate trends
- **Models:** Linear Regression, XGBoost (with lag features), Facebook Prophet

### Module 4 — Credit Demand Forecast

Forecasts monthly loan demand broken down by loan grade (A through E).

- **Target:** Monthly `funded_amnt` per `grade` category
- **Key features:** Grade-level historical trends, interest rate by grade, seasonal indicators
- **Output:** Separate forecast for each grade — visualised as a stacked bar chart

---

## Pipeline

```
Raw CSV
  └── data_pipeline.py       (drop nulls, fix dtypes, filter valid loan_status)
        └── feature_engineering.py   (encode, scale, lag features, label construction)
              └── model_utils.py     (train, cross-validate, evaluate, save model)
                    └── notebooks/   (exploration + results per module)
                          └── dashboard/app.py   (Streamlit UI)
```

---

## Dashboard

Run the Streamlit app locally:

```bash
streamlit run dashboard/app.py
```

The dashboard has 4 tabs — one per module. Each tab allows you to:

- Select a model (Logistic Regression / Random Forest / XGBoost)
- Input borrower or time parameters
- View prediction output and confidence score
- View SHAP feature importance chart
- View model comparison metrics

---

## Tech Stack

| Category          | Tools                           |
| ----------------- | ------------------------------- |
| Data processing   | Pandas, NumPy                   |
| Machine learning  | Scikit-learn, XGBoost, LightGBM |
| Imbalanced data   | imbalanced-learn (SMOTE)        |
| Time series       | Facebook Prophet                |
| Explainability    | SHAP                            |
| Visualisation     | Matplotlib, Seaborn, Plotly     |
| Dashboard         | Streamlit                       |
| Model persistence | Joblib                          |

---

## Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/finsight-forecasting.git
cd finsight-forecasting

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add the dataset
# Download accepted_2007_to_2018Q4.csv from Kaggle and place in data/raw/

# 4. Run notebooks in order
jupyter notebook notebooks/01_loan_default_risk.ipynb

# 5. Launch dashboard
streamlit run dashboard/app.py
```

---

## License

MIT License — free to use and adapt for your own portfolio.

---
