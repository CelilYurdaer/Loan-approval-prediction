# Loan Approval Prediction

A machine learning classification project that predicts loan approval outcomes based on applicant financial and demographic data. Compares Logistic Regression, Decision Tree, and Random Forest models with probability calibration and threshold optimization.

---

## Project Overview

Lending institutions need to assess credit risk quickly and accurately. This project builds and evaluates three classification models to predict whether a loan application will be approved or rejected, using features like income, credit score, loan amount, and asset holdings.

**Key questions explored:**

- Which applicant features are the strongest predictors of loan approval?
- How do different ML models compare on this classification task?
- What is the trade-off between precision (avoiding bad loans) and recall (not rejecting good applicants)?
- Does the model generalize well, or is it overfitting to the training data?
- Is the default 0.5 decision threshold optimal for financial risk?
- Are the model's predicted probabilities reliable enough to use as default risk estimates?

## Key Findings

| Insight | Detail |
|---------|--------|
| **Approval rate** | ~62% of applications approved |
| **Top predictor** | CIBIL score has the strongest correlation with approval |
| **Best model** | Random Forest achieves highest F1 and ROC-AUC scores |
| **Generalization** | Cross-validation (leak-free via Pipeline) confirms consistent performance |
| **Engineered features** | Loan-to-income ratio and total assets improve prediction accuracy |
| **Outliers** | IQR analysis + with/without outlier model comparison to measure impact |
| **Threshold** | Optimal decision threshold found via F1 sweep across 0.1–0.9 range |
| **Calibration** | Platt scaling applied for reliable probability estimates (Brier score reported) |

## Visualizations

The script generates six chart images:

- **`eda_dashboard.png`** — Approval distribution, CIBIL score by status, loan-to-income boxplots, feature correlations
- **`correlation_heatmap.png`** — Full feature correlation matrix
- **`model_evaluation.png`** — Confusion matrices and ROC curves for all three models
- **`model_comparison.png`** — Side-by-side metric comparison (accuracy, precision, recall, F1, ROC-AUC)
- **`feature_importance.png`** — MDI (tree-based) and permutation importance side by side
- **`calibration_plot.png`** — Before/after calibration curve with Brier scores and probability distributions

## Tools & Technologies

- **Python 3** — core language
- **pandas & NumPy** — data manipulation
- **scikit-learn** — Pipeline, model training, evaluation, calibration, permutation importance
- **Matplotlib & Seaborn** — visualization

## Project Structure

```
loan-approval-prediction/
│
├── loan_prediction.py          # Main analysis & ML pipeline
├── loan_approval_dataset.csv   # Raw dataset
├── eda_dashboard.png           # EDA visualizations
├── correlation_heatmap.png     # Feature correlations
├── model_evaluation.png        # Confusion matrices & ROC curves
├── model_comparison.png        # Metric comparison chart
├── feature_importance.png      # MDI + permutation importance
├── calibration_plot.png        # Probability calibration curves
└── README.md                   # This file
```

## How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/CelilYurdaer/loan-approval-prediction.git
   cd loan-approval-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

3. **Run the analysis**
   ```bash
   python loan_prediction.py
   ```

## Analysis Pipeline

```
Load → Validate → Clean & Encode → Engineer Features → EDA → Outlier Analysis → Train (Pipeline) → Calibrate → Evaluate → Insights
```

1. **Data Loading** — Read CSV, strip whitespace from columns and values
2. **Validation** — Check required columns, target values, numeric ranges, and missing data thresholds
3. **Cleaning & Encoding** — Drop ID column, handle missing values, binary LabelEncoder for 2-value columns, one-hot encoding for 3+ value columns
4. **Feature Engineering** — Create total_assets, loan_to_income, loan_to_assets, income_per_dependent, asset_to_income, cibil_tier (Int64), with median imputation for NaN ratios
5. **EDA** — Target distribution, correlation analysis, descriptive statistics
6. **Outlier Detection** — IQR-based analysis + model comparison with/without outliers to measure their impact
7. **Model Training** — sklearn Pipeline (StandardScaler → Model) with class_weight="balanced" and leak-free 5-fold cross-validation
8. **Threshold Optimization** — F1 sweep across 0.1–0.9 to find optimal decision boundary for financial risk
9. **Probability Calibration** — CalibratedClassifierCV (Platt scaling) for reliable default probability estimates
10. **Evaluation** — Confusion matrix, ROC-AUC, precision/recall/F1, permutation importance
11. **Insights Summary** — Plain-language takeaways

## Models Compared

| Model | Why Include It |
|-------|---------------|
| **Logistic Regression** | Linear baseline — fast, interpretable, good for understanding feature direction |
| **Decision Tree** | Non-linear — captures complex rules but prone to overfitting |
| **Random Forest** | Ensemble method — combines many trees to reduce overfitting, typically best performer |

All models use `class_weight="balanced"` to handle the 62/38 class imbalance.

## ML Concepts Demonstrated

- **sklearn Pipeline** — chains StandardScaler + Model to prevent data leakage during cross-validation
- **Train/test split with stratification** — preserves class balance across sets
- **Cross-validation (leak-free)** — Pipeline ensures scaler is fit only on each CV training fold
- **class_weight="balanced"** — penalizes minority class misclassification more heavily
- **Decision threshold optimization** — F1-maximizing threshold search for financial applications
- **Probability calibration** — Platt scaling via CalibratedClassifierCV with Brier score evaluation
- **Permutation importance** — unbiased feature ranking that avoids MDI's high-cardinality bias
- **Outlier impact analysis** — trains models with and without outliers to justify keep/remove decisions
- **Dataset validation** — automated sanity checks before processing
- **ROC-AUC** — evaluates classifier quality beyond simple accuracy
- **Precision vs. recall trade-off** — balancing false approvals against missed opportunities

## Dataset

The dataset contains 4,269 loan applications with 12 features including income, loan amount, CIBIL credit score, asset values, education level, and employment status. Source: [Kaggle Loan Approval Prediction Dataset](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset).

## Contact

Emir Celil Yurdaer — [celilyurdaer@protonmail.com](mailto:celilyurdaer@protonmail.com) — Feel free to open an issue or reach out with questions!

---

*Built as a portfolio project demonstrating end-to-end machine learning classification, from data validation to probability calibration.*
