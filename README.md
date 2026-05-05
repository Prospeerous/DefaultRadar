<div align="center">

# DefaultRadar

### Intelligent Credit Risk Assessment Powered by Machine Learning

*Predicting loan defaults before they happen — with full explainability.*

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2-189AB4?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAyTDIgN2wxMCA1IDEwLTV6bTAgMTBMMyAxMmw5IDQgOS00em0wIDVsLTkgNCA5IDQgOS00eiIvPjwvc3ZnPg==)](https://xgboost.ai)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.57-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-00C49F?style=for-the-badge)](https://shap.readthedocs.io)

---

**ROC-AUC 94.49%** &nbsp;·&nbsp; **6 Models Benchmarked** &nbsp;·&nbsp; **32,581 Borrowers** &nbsp;·&nbsp; **Zero Overfitting**

</div>

---

## What is DefaultRadar?

DefaultRadar is a production-grade credit risk prediction system that determines whether a loan applicant is likely to default — and explains *exactly why*.

Banks and lenders lose billions annually to bad loans. Traditional scoring models are opaque and slow to adapt. DefaultRadar benchmarks six machine learning algorithms head-to-head, selects the best via cross-validated tuning, and wraps the result in an interactive web application where any loan application can be assessed in real time, with SHAP-powered explanations that show which factors drove the decision.

This is not just a model. It's a complete, deployable risk assessment pipeline.

---

## Live Demo

> Launch the Streamlit app locally:
> ```bash
> streamlit run app/app.py
> ```

| Light Mode | Dark Mode |
|---|---|
| Clean white interface with full model insights | Full dark theme with matching charts and tables |

Features of the demo app:
- Fill in any borrower profile using the sidebar form
- Instant prediction with a probability gauge and risk badge (Low / Medium / High)
- SHAP waterfall chart explaining the top 10 factors for *that specific applicant*
- Auto-generated plain-English explanation of the decision
- Full model comparison table across all 6 algorithms
- Light / Dark mode toggle

---

## Model Performance

Six classifiers were trained, evaluated, and compared on the same 80/20 stratified split after SMOTE balancing.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Overfit Gap |
|---|---|---|---|---|---|---|
| **XGBoost** ✅ | 92.77% | 91.89% | 73.35% | 81.58% | **94.49%** | **4.73 pp** |
| Random Forest | 91.96% | 86.86% | 74.40% | 80.15% | 92.80% | 7.20 pp |
| Gradient Boosting | 91.15% | 85.24% | 71.87% | 77.99% | 92.29% | 5.49 pp |
| Decision Tree | 90.47% | 80.69% | 74.05% | 77.23% | 90.46% | 6.42 pp |
| SVC | 89.44% | 78.49% | 71.10% | 74.61% | 89.55% | 6.42 pp |
| Logistic Regression | 82.03% | 57.92% | 64.56% | 61.06% | 83.42% | 8.93 pp |

> **Overfit Gap** = Train AUC − Test AUC. XGBoost is the only model with a gap below 5 pp, confirming strong generalisation. Random Forest memorised the training set perfectly (Train AUC = 1.000) — textbook overfitting.

**XGBoost was chosen** for deployment. Regularised via `subsample=0.8` and `colsample_bytree=0.8`, fine-tuned with 5-fold GridSearchCV across 40 hyperparameter combinations.

---

## Pipeline Architecture

```
Raw Data (32,581 rows × 12 features)
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  SECTION 2  Data Loading & Inspection                       │
│             Shape, dtypes, missing values, class balance    │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  SECTION 3  Exploratory Data Analysis                       │
│             Distributions · Correlations · Outliers         │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  SECTION 4  Preprocessing                                   │
│             Outlier capping · Median imputation             │
│             Ordinal encoding (loan_grade A→G)               │
│             One-hot encoding (home_ownership, loan_intent)  │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  SECTION 5  SMOTE  (training set only)                      │
│             78 / 22  →  50 / 50  class balance              │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  SECTION 6  StandardScaler  (fit on train, transform both)  │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  SECTION 7  Model Training  (6 classifiers)                 │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  SECTION 8  Evaluation                                      │
│             Metrics table · Confusion matrices · ROC curves │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  SECTION 9  Hyperparameter Tuning  (GridSearchCV, CV=5)     │
│             40 XGBoost combinations, scoring = ROC-AUC      │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  SECTION 10  SHAP Explainability                            │
│              Summary · Importance · Waterfall per borrower  │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  Streamlit Web App  —  real-time prediction + explanation   │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Findings

**What predicts default most strongly?**

| # | Finding |
|---|---------|
| 1 | **Loan Grade** is the single strongest predictor. Grade D–G borrowers default at substantially higher rates. |
| 2 | **Loan-to-Income Ratio** matters more than the raw loan amount. A $5k loan on a $12k income is riskier than a $20k loan on a $100k income. |
| 3 | **Prior default on file** is the most direct signal — past behaviour is the strongest predictor of future behaviour. |
| 4 | **Interest rate** is a symptom of risk, not a cause. Lenders already price in risk; high rates confirm an existing assessment. |
| 5 | **Home ownership** (MORTGAGE vs RENT) reflects financial stability. Mortgage holders default at lower rates than renters. |

---

## Project Structure

```
DefaultRadar/
│
├── app/
│   ├── app.py                  # Streamlit web application
│   ├── model.pkl               # Trained XGBoost model
│   ├── scaler.pkl              # Fitted StandardScaler
│   └── feature_names.pkl       # Ordered feature list
│
├── .streamlit/
│   └── config.toml             # Light theme base for native widgets
│
├── credit-risk-loan-default-prediction.ipynb   # Full ML notebook (11 sections)
├── credit_risk_dataset.csv                     # Dataset (32,581 rows)
├── credit_risk_dataset.csv.zip                 # Compressed dataset
├── requirements.txt
│
├── class_distribution.png      # EDA chart — class imbalance
├── numeric_distributions.png   # EDA chart — feature distributions
├── categorical_analysis.png    # EDA chart — categorical default rates
├── correlation_heatmap.png     # EDA chart — feature correlations
├── boxplots.png                # EDA chart — outlier detection
├── smote_balance.png           # Before/after SMOTE balance
├── model_comparison.png        # All 6 models side-by-side
├── confusion_matrices.png      # Confusion matrices for all models
├── roc_curves.png              # ROC curves — all models on one chart
├── shap_summary.png            # SHAP beeswarm summary
├── shap_importance.png         # SHAP mean absolute importance
└── shap_waterfall.png          # SHAP waterfall — single prediction
```

---

## Quickstart

### 1. Clone

```bash
git clone https://github.com/Prospeerous/DefaultRadar.git
cd DefaultRadar
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebook

Open `credit-risk-loan-default-prediction.ipynb` in Jupyter and run all cells top to bottom. This reproduces the full pipeline and regenerates all charts.

```bash
jupyter notebook credit-risk-loan-default-prediction.ipynb
```

### 4. Launch the web app

```bash
streamlit run app/app.py
```

Then open **http://localhost:8501** in your browser.

> The `app/` folder already contains the pre-trained model artifacts, so you can run the web app directly without re-running the notebook.

---

## Dataset

**Credit Risk Dataset** — sourced from [Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)

| Feature | Description |
|---|---|
| `person_age` | Borrower age (years) |
| `person_income` | Annual income ($) |
| `person_home_ownership` | RENT / OWN / MORTGAGE / OTHER |
| `person_emp_length` | Employment length (years) |
| `loan_intent` | Purpose: EDUCATION / MEDICAL / PERSONAL / etc. |
| `loan_grade` | Credit grade: A (best) → G (worst) |
| `loan_amnt` | Loan amount requested ($) |
| `loan_int_rate` | Interest rate (%) |
| `loan_percent_income` | Loan amount ÷ annual income |
| `cb_person_default_on_file` | Prior default on credit bureau (Y/N) |
| `cb_person_cred_hist_length` | Credit history length (years) |
| `loan_status` | **Target** — 0 = repaid, 1 = defaulted |

Class distribution: **78.2% non-default / 21.8% default** (imbalanced → corrected with SMOTE)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Data | pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| ML Models | scikit-learn, XGBoost |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Explainability | SHAP |
| Web App | Streamlit |
| Model Persistence | joblib |
| Notebook | Jupyter |

---

## Why This Matters

Credit risk is one of the highest-stakes applications of machine learning. A missed default (false negative) costs the lender the full principal. A wrongly rejected application (false positive) costs a creditworthy borrower access to capital and costs the lender revenue.

Getting this right requires more than accuracy — it demands:
- **Calibrated probability estimates**, not just binary labels
- **Explainability**, because regulators require banks to justify rejections
- **Robustness to class imbalance**, because defaults are inherently rare
- **A generalisation check**, because a model that memorises training data is useless in production

DefaultRadar addresses all four.

---

<div align="center">

Built with Python · XGBoost · SHAP · Streamlit

*Abigael Mwangi — Your top Data Scientist*

</div>
