"""
Loan Approval Prediction — Machine Learning Classification Project
===================================================================
Predicts whether a loan application will be approved or rejected
based on applicant financial and demographic features.

Pipeline:
    1. Data Loading & Inspection
    2. Data Cleaning & Preprocessing
    3. Exploratory Data Analysis (EDA)
    4. Feature Engineering
    5. Model Training (Logistic Regression, Decision Tree, Random Forest)
    6. Model Evaluation (Confusion Matrix, ROC-AUC, Classification Report)
    7. Feature Importance Analysis
    8. Insights Summary
"""

# ─────────────────────────────────────────────────────────────────────
# 1. IMPORTS
# ─────────────────────────────────────────────────────────────────────

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set global random seed for full reproducibility across numpy and sklearn.
np.random.seed(42)

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.inspection import permutation_importance

sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.1)
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"

PALETTE = sns.color_palette("viridis", 20)  # enough colors for all features


# ─────────────────────────────────────────────────────────────────────
# 2. DATA LOADING
# ─────────────────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV, strip whitespace from column names and string values."""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    # String columns often carry leading/trailing spaces from CSV exports.
    # Stripping prevents mismatches like " Approved" != "Approved".
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].str.strip()

    print(f"✓ Loaded {filepath}")
    print(f"  Shape : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Columns: {list(df.columns)}\n")
    return df


def validate_dataset(df: pd.DataFrame) -> bool:
    """Run sanity checks on the raw dataset before processing.

    Catches common data issues early:
    - Missing required columns
    - Empty dataset
    - Target variable has unexpected values
    - Numeric columns contain impossible values (e.g., negative income)

    Returns True if all checks pass, raises ValueError otherwise.
    """

    print("=" * 60)
    print("DATASET VALIDATION")
    print("=" * 60)

    required_cols = ["income_annum", "loan_amount", "loan_term",
                     "cibil_score", "loan_status"]

    # Check required columns exist
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    print("  ✓ All required columns present")

    # Check dataset is not empty
    if len(df) == 0:
        raise ValueError("Dataset is empty")
    print(f"  ✓ Dataset has {len(df):,} rows")

    # Check target variable values
    valid_targets = {"Approved", "Rejected"}
    actual_targets = set(df["loan_status"].unique())
    if not actual_targets.issubset(valid_targets):
        unexpected = actual_targets - valid_targets
        raise ValueError(f"Unexpected loan_status values: {unexpected}")
    print(f"  ✓ Target values valid: {actual_targets}")

    # Check numeric columns for impossible values
    checks = {
        "income_annum": ("≥ 0", lambda s: (s >= 0).all()),
        "loan_amount": ("> 0", lambda s: (s > 0).all()),
        "cibil_score": ("300-900 range", lambda s: s.between(300, 900).all()),
        "loan_term": ("> 0", lambda s: (s > 0).all()),
    }

    all_valid = True
    for col, (rule, check_fn) in checks.items():
        if col in df.columns:
            valid = check_fn(df[col].dropna())
            status = "✓" if valid else "⚠ FAILED"
            print(f"  {status} {col} {rule}")
            if not valid:
                n_invalid = (~check_fn(df[col].dropna())).sum() if not valid else 0
                print(f"      → {n_invalid} invalid values found")
                all_valid = False

    # Check for high missing percentage
    for col in df.columns:
        pct_missing = df[col].isna().sum() / len(df) * 100
        if pct_missing > 30:
            print(f"  ⚠ '{col}' has {pct_missing:.1f}% missing values")
            all_valid = False

    if all_valid:
        print("  ✓ All validation checks passed")
    else:
        print("  ⚠ Some checks failed — review warnings above")

    print()
    return all_valid


# ─────────────────────────────────────────────────────────────────────
# 3. DATA CLEANING & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values, duplicates, and encode categorical variables."""

    df = df.copy()

    # --- Drop loan_id: it's a row identifier, not a predictive feature -------
    if "loan_id" in df.columns:
        df = df.drop(columns=["loan_id"])

    # --- Duplicates -----------------------------------------------------------
    n_dupes = df.duplicated().sum()
    if n_dupes:
        df = df.drop_duplicates(keep="first")
        print(f"  Removed {n_dupes} duplicate row(s)")

    # --- Missing values -------------------------------------------------------
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        print("  Missing values per column:")
        for col, count in missing.items():
            pct = count / len(df) * 100
            print(f"    {col:30s} → {count:>5} ({pct:.1f}%)")

        # Numeric columns: fill with median (robust to outliers)
        num_cols = df.select_dtypes(include=np.number).columns
        for col in num_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

        # Categorical columns: fill with mode (most frequent value)
        cat_cols = df.select_dtypes(include=["object", "string"]).columns
        for col in cat_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode()[0])
    else:
        print("  No missing values ✓")

    # --- Encode target variable -----------------------------------------------
    # Convert "Approved"/"Rejected" to 1/0 for classification.
    df["loan_status"] = df["loan_status"].map({"Approved": 1, "Rejected": 0})

    # --- Encode categorical features ------------------------------------------
    # LabelEncoder maps text labels to integers (e.g., "No"→0, "Yes"→1).
    # This is safe for binary categories where 0/1 mapping is natural.
    # For categories with 3+ unordered values (e.g., "Red","Blue","Green"),
    # LabelEncoder would create a false numeric ordering (0 < 1 < 2).
    # In that case, pd.get_dummies() (one-hot encoding) should be used instead.
    le = LabelEncoder()
    categorical_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in categorical_cols:
        n_unique = df[col].nunique()
        if n_unique <= 2:
            df[col] = le.fit_transform(df[col])
            print(f"  Encoded '{col}' (binary): "
                  f"{dict(zip(le.classes_, le.transform(le.classes_)))}")
        else:
            # One-hot encode: creates separate 0/1 columns for each value.
            # drop_first=True avoids multicollinearity (dummy variable trap).
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True,
                                     dtype=int)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            print(f"  One-hot encoded '{col}' ({n_unique} values): "
                  f"{list(dummies.columns)}")

    print(f"  Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n")
    return df


# ─────────────────────────────────────────────────────────────────────
# 4. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────────────────────────────

def run_eda(df: pd.DataFrame) -> dict:
    """Generate descriptive statistics and explore feature relationships."""

    insights = {}

    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # --- Target distribution --------------------------------------------------
    target_dist = df["loan_status"].value_counts()
    approval_rate = target_dist[1] / len(df) * 100
    print(f"\n  Target distribution:")
    print(f"    Approved: {target_dist[1]:,} ({approval_rate:.1f}%)")
    print(f"    Rejected: {target_dist[0]:,} ({100 - approval_rate:.1f}%)")
    insights["approval_rate"] = approval_rate

    # --- Descriptive statistics -----------------------------------------------
    print("\n  DESCRIPTIVE STATISTICS:")
    print(df.describe().round(2).to_string())

    # --- Correlation with target ----------------------------------------------
    # Shows which features have the strongest linear relationship with
    # loan approval. High absolute values suggest predictive power.
    corr_with_target = (
        df.corr(numeric_only=True)["loan_status"]
        .drop("loan_status")
        .sort_values(key=abs, ascending=False)
    )
    insights["corr_with_target"] = corr_with_target
    print("\n  CORRELATION WITH LOAN APPROVAL:")
    for feat, val in corr_with_target.items():
        bar = "█" * int(abs(val) * 40)
        print(f"    {feat:30s} {val:+.3f}  {bar}")

    # --- Full correlation matrix ----------------------------------------------
    corr_matrix = df.corr(numeric_only=True).round(3)
    insights["correlation"] = corr_matrix

    print()
    return insights


# ─────────────────────────────────────────────────────────────────────
# 5. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features that may improve model performance."""

    df = df.copy()

    # Total assets: sum of all asset categories.
    # A single aggregated wealth indicator can be more predictive than
    # individual asset columns, especially for tree-based models.
    df["total_assets"] = (
        df["residential_assets_value"]
        + df["commercial_assets_value"]
        + df["luxury_assets_value"]
        + df["bank_asset_value"]
    )

    # Loan-to-income ratio: measures how large the loan is relative
    # to annual income. A high ratio signals higher repayment risk.
    df["loan_to_income"] = np.where(
        df["income_annum"] > 0,
        df["loan_amount"] / df["income_annum"],
        np.nan
    )

    # Loan-to-assets ratio: measures loan size relative to total wealth.
    # Applicants with high assets relative to their loan are safer bets.
    df["loan_to_assets"] = np.where(
        df["total_assets"] > 0,
        df["loan_amount"] / df["total_assets"],
        np.nan
    )

    # Income per dependent: disposable income per household member.
    # More dependents on the same income means less capacity to repay.
    df["income_per_dependent"] = np.where(
        df["no_of_dependents"] > 0,
        df["income_annum"] / df["no_of_dependents"],
        df["income_annum"]  # no dependents = full income available
    )

    # Asset-to-income ratio: overall financial health indicator.
    df["asset_to_income"] = np.where(
        df["income_annum"] > 0,
        df["total_assets"] / df["income_annum"],
        np.nan
    )

    # CIBIL score category: bin the credit score into risk tiers.
    # CIBIL ranges: 300-549 (Poor), 550-649 (Fair), 650-749 (Good), 750+ (Excellent)
    # Using Int64 (nullable integer) instead of float because tiers are
    # ordinal categories (0,1,2,3), not continuous values. Int64 can hold
    # NaN unlike regular int, which is needed if scores fall outside bins.
    df["cibil_tier"] = pd.cut(
        df["cibil_score"],
        bins=[0, 549, 649, 749, 900],
        labels=[0, 1, 2, 3]
    ).astype("Int64")

    # --- Handle NaN in engineered features ------------------------------------
    # Engineered ratios can produce NaN when divisor is 0.
    # Median imputation is used because:
    #   - fillna(0) would imply "no risk" which is misleading
    #   - dropping rows loses training data unnecessarily
    #   - median is robust to outliers unlike mean
    engineered_ratio_cols = ["loan_to_income", "loan_to_assets",
                             "asset_to_income"]
    for col in engineered_ratio_cols:
        n_nan = df[col].isna().sum()
        if n_nan > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  Filled {n_nan} NaN in '{col}' with median ({median_val:.2f})")

    new_features = ["total_assets", "loan_to_income", "loan_to_assets",
                    "income_per_dependent", "asset_to_income", "cibil_tier"]
    print(f"✓ Engineered {len(new_features)} new features: {', '.join(new_features)}\n")
    return df


# ─────────────────────────────────────────────────────────────────────
# 6. OUTLIER ANALYSIS
# ─────────────────────────────────────────────────────────────────────

def detect_outliers_iqr(series: pd.Series) -> pd.Series:
    """Return a boolean mask where True indicates an outlier (IQR method)."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    return (series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR)


def run_outlier_analysis(df: pd.DataFrame) -> pd.Series:
    """Detect and report outliers, return combined outlier mask."""

    print("=" * 60)
    print("OUTLIER ANALYSIS (IQR Method)")
    print("=" * 60)

    outlier_cols = ["income_annum", "loan_amount", "loan_to_income",
                    "total_assets", "cibil_score"]

    # Combined mask: True if the row is an outlier in ANY checked column
    combined_mask = pd.Series(False, index=df.index)

    for col in outlier_cols:
        valid_mask = df[col].notna()
        mask = detect_outliers_iqr(df[col])
        combined_mask = combined_mask | mask
        n_outliers = mask[valid_mask].sum()
        pct = n_outliers / valid_mask.sum() * 100

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        print(f"\n  {col}:")
        print(f"    Q1 = {Q1:,.2f}  |  Q3 = {Q3:,.2f}  |  IQR = {IQR:,.2f}")
        print(f"    Bounds: [{Q1 - 1.5*IQR:,.2f}, {Q3 + 1.5*IQR:,.2f}]")
        print(f"    Outliers: {n_outliers} ({pct:.1f}%)")

    total = combined_mask.sum()
    print(f"\n  Total rows flagged as outlier in any column: "
          f"{total} ({total/len(df)*100:.1f}%)")
    print()
    return combined_mask


def compare_outlier_impact(df: pd.DataFrame, outlier_mask: pd.Series):
    """Train the best model with and without outliers to measure impact.

    If removing outliers improves F1, the outliers were adding noise.
    If F1 drops, the outliers contained valid signal the model used.
    This comparison justifies the decision to keep or remove outliers.
    """

    print("=" * 60)
    print("OUTLIER IMPACT COMPARISON")
    print("=" * 60)

    df_clean = df[~outlier_mask].copy()
    print(f"  Full dataset:    {len(df):,} rows")
    print(f"  Without outliers: {len(df_clean):,} rows "
          f"({outlier_mask.sum()} removed)\n")

    results_comparison = {}

    for label, data in [("With outliers", df), ("Without outliers", df_clean)]:
        X = data.drop(columns=["loan_status"])
        y = data["loan_status"]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        scaler = StandardScaler()
        X_tr = pd.DataFrame(scaler.fit_transform(X_tr), columns=X.columns)
        X_te = pd.DataFrame(scaler.transform(X_te), columns=X.columns)

        model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42,
            class_weight="balanced"
        )
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1]

        f1 = f1_score(y_te, y_pred)
        roc = roc_auc_score(y_te, y_prob)
        results_comparison[label] = {"f1": f1, "roc_auc": roc}
        print(f"  {label:20s} → F1 = {f1:.4f}  |  ROC-AUC = {roc:.4f}")

    diff = results_comparison["Without outliers"]["f1"] - results_comparison["With outliers"]["f1"]
    if diff > 0.005:
        print(f"\n  → Removing outliers improves F1 by {diff:+.4f}. "
              f"Outliers were adding noise.")
    elif diff < -0.005:
        print(f"\n  → Removing outliers decreases F1 by {diff:+.4f}. "
              f"Outliers contain valid signal — keeping them.")
    else:
        print(f"\n  → Minimal difference ({diff:+.4f}). "
              f"Outliers have negligible impact on model performance.")
    print()
    return results_comparison


# ─────────────────────────────────────────────────────────────────────
# 7. VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────

def plot_eda_dashboard(df: pd.DataFrame, insights: dict):
    """Generate a 2×2 EDA dashboard."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    fig.suptitle("Loan Approval Prediction — EDA Dashboard",
                 fontsize=18, fontweight="bold", y=1.01)

    # ---- Plot 1: Target distribution ----------------------------------------
    ax = axes[0, 0]
    labels = ["Rejected", "Approved"]
    counts = [df["loan_status"].value_counts()[0],
              df["loan_status"].value_counts()[1]]
    colors_target = [PALETTE[1], PALETTE[7]]
    bars = ax.bar(labels, counts, color=colors_target, edgecolor="white")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                f"{count:,}", ha="center", fontweight="bold")
    ax.set_title("Loan Approval Distribution", fontweight="bold")
    ax.set_ylabel("Count")

    # ---- Plot 2: CIBIL score distribution by approval status ----------------
    ax = axes[0, 1]
    for status, color, label in [(1, PALETTE[7], "Approved"),
                                  (0, PALETTE[1], "Rejected")]:
        subset = df[df["loan_status"] == status]["cibil_score"]
        sns.kdeplot(subset, ax=ax, color=color, label=label, fill=True, alpha=0.3)
    ax.set_title("CIBIL Score Distribution by Status", fontweight="bold")
    ax.set_xlabel("CIBIL Score")
    ax.legend()

    # ---- Plot 3: Loan-to-income ratio by status ----------------------------
    ax = axes[1, 0]
    data_approved = df[df["loan_status"] == 1]["loan_to_income"].dropna()
    data_rejected = df[df["loan_status"] == 0]["loan_to_income"].dropna()
    bp = ax.boxplot([data_approved, data_rejected],
                    labels=["Approved", "Rejected"], patch_artist=True)
    bp["boxes"][0].set_facecolor(PALETTE[7])
    bp["boxes"][1].set_facecolor(PALETTE[1])
    for box in bp["boxes"]:
        box.set_alpha(0.7)
    ax.set_title("Loan-to-Income Ratio by Status", fontweight="bold")
    ax.set_ylabel("Loan / Annual Income")

    # ---- Plot 4: Correlation with target ------------------------------------
    ax = axes[1, 1]
    corr = insights["corr_with_target"]
    colors_corr = [PALETTE[7] if v > 0 else PALETTE[1] for v in corr.values]
    corr.plot(kind="barh", ax=ax, color=colors_corr, edgecolor="white")
    ax.set_title("Feature Correlation with Approval", fontweight="bold")
    ax.set_xlabel("Pearson Correlation")
    ax.axvline(x=0, color="black", linewidth=0.5)

    plt.tight_layout()
    fig.savefig("eda_dashboard.png", dpi=150)
    print("✓ EDA dashboard saved → eda_dashboard.png")
    plt.show()


def plot_correlation_heatmap(insights: dict):
    """Heatmap of the full feature correlation matrix."""
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(
        insights["correlation"], annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, square=True, ax=ax,
        linewidths=0.5, cbar_kws={"shrink": 0.8},
        annot_kws={"size": 8}
    )
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig("correlation_heatmap.png", dpi=150)
    print("✓ Heatmap saved → correlation_heatmap.png")
    plt.show()


# ─────────────────────────────────────────────────────────────────────
# 8. MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────

def prepare_data(df: pd.DataFrame):
    """Split data into features (X) and target (y), then train/test split.

    Scaling is NOT applied here. Instead, each model uses a Pipeline
    that includes StandardScaler as the first step. This ensures:
    1. Cross-validation does not leak test fold statistics into training
    2. The scaler is always fit only on training data
    """

    X = df.drop(columns=["loan_status"])
    y = df["loan_status"]

    # 80% training, 20% testing. random_state ensures reproducibility —
    # anyone running this script gets the exact same split.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Stratify preserves the approval/rejection ratio in both sets.
    # Without it, one set could randomly get 70% approved and the other 55%.
    print(f"  Train set: {X_train.shape[0]:,} samples")
    print(f"  Test set:  {X_test.shape[0]:,} samples")
    print(f"  Target ratio preserved: "
          f"{y_train.mean():.2%} train / {y_test.mean():.2%} test\n")

    return X_train, X_test, y_train, y_test


def find_optimal_threshold(y_true, y_prob):
    """Search for the decision threshold that maximizes F1 score.

    Default threshold is 0.5, but in financial applications this is
    often suboptimal. A lower threshold (e.g., 0.3) catches more
    true positives but increases false positives. A higher threshold
    (e.g., 0.7) is more conservative — fewer approvals, fewer bad loans.

    The optimal threshold balances precision and recall for the
    specific business context.
    """
    thresholds = np.arange(0.1, 0.9, 0.01)
    f1_scores = []
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        f1_scores.append(f1_score(y_true, y_pred_t))

    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx], thresholds, f1_scores


def train_models(X_train, X_test, y_train, y_test):
    """Train multiple classifiers using sklearn Pipeline and return results.

    Each model is wrapped in a Pipeline with StandardScaler as the first
    step. This prevents data leakage during cross-validation: the scaler
    is fit only on each CV training fold, not on the full training set.

    class_weight="balanced" adjusts the loss function to penalize
    misclassifications of the minority class more heavily.
    """

    print("=" * 60)
    print("MODEL TRAINING & EVALUATION")
    print("=" * 60)

    # Each Pipeline chains StandardScaler → Model.
    # During cross_val_score, sklearn splits X_train into 5 folds,
    # then for each fold: fits scaler on 4 folds, transforms the 5th,
    # then fits the model. No leakage.
    pipelines = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000, random_state=42, class_weight="balanced"
            ))
        ]),
        "Decision Tree": Pipeline([
            ("scaler", StandardScaler()),
            ("model", DecisionTreeClassifier(
                max_depth=5, random_state=42, class_weight="balanced"
            ))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42,
                class_weight="balanced"
            ))
        ]),
    }

    results = {}

    for name, pipe in pipelines.items():
        print(f"\n{'─' * 40}")
        print(f"  {name}")
        print(f"{'─' * 40}")

        # Train the pipeline (scaler + model)
        pipe.fit(X_train, y_train)

        # Predict on test set
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        # Cross-validation on unscaled data — Pipeline handles scaling
        # internally for each fold, preventing leakage.
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="f1")

        # Optimal threshold search
        best_thresh, best_f1_thresh, all_thresh, all_f1 = \
            find_optimal_threshold(y_test, y_prob)
        y_pred_optimized = (y_prob >= best_thresh).astype(int)

        # Metrics at default threshold (0.5)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        # Metrics at optimal threshold
        f1_opt = f1_score(y_test, y_pred_optimized)
        prec_opt = precision_score(y_test, y_pred_optimized)
        rec_opt = recall_score(y_test, y_pred_optimized)

        results[name] = {
            "pipeline": pipe,
            "model": pipe.named_steps["model"],
            "y_pred": y_pred,
            "y_pred_optimized": y_pred_optimized,
            "y_prob": y_prob,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc_auc,
            "cv_f1_mean": cv_scores.mean(),
            "cv_f1_std": cv_scores.std(),
            "optimal_threshold": best_thresh,
            "f1_at_optimal": f1_opt,
            "threshold_curve": (all_thresh, all_f1),
        }

        print(f"  Accuracy:    {acc:.4f}")
        print(f"  Precision:   {prec:.4f}")
        print(f"  Recall:      {rec:.4f}")
        print(f"  F1 Score:    {f1:.4f}")
        print(f"  ROC-AUC:     {roc_auc:.4f}")
        print(f"  CV F1 (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"  Optimal threshold: {best_thresh:.2f} "
              f"(F1 = {f1_opt:.4f}, Prec = {prec_opt:.4f}, Rec = {rec_opt:.4f})")
        print(f"\n  Classification Report (default threshold 0.5):")
        print(classification_report(y_test, y_pred,
                                     target_names=["Rejected", "Approved"]))

    return results


# ─────────────────────────────────────────────────────────────────────
# 9. PROBABILITY CALIBRATION
# ─────────────────────────────────────────────────────────────────────

def calibrate_best_model(results, X_train, X_test, y_train, y_test):
    """Calibrate the best model's probability outputs using Platt scaling.

    Raw model probabilities are not always well-calibrated: a predicted
    probability of 0.7 doesn't necessarily mean 70% of those cases are
    approved. CalibratedClassifierCV fits a sigmoid (Platt scaling) or
    isotonic regression to map raw outputs to true probabilities.

    This is critical in finance where the actual default probability
    drives loan pricing and risk assessment decisions.
    """

    print("=" * 60)
    print("PROBABILITY CALIBRATION")
    print("=" * 60)

    best_name = max(results, key=lambda k: results[k]["f1"])
    best_pipe = results[best_name]["pipeline"]

    # CalibratedClassifierCV uses internal cross-validation (cv=5) to
    # avoid overfitting the calibration to the training data.
    calibrated = CalibratedClassifierCV(best_pipe, cv=5, method="sigmoid")
    calibrated.fit(X_train, y_train)

    y_prob_raw = best_pipe.predict_proba(X_test)[:, 1]
    y_prob_cal = calibrated.predict_proba(X_test)[:, 1]

    # Calibration curve: bins predicted probabilities and compares
    # them to the actual fraction of positives in each bin.
    # A perfectly calibrated model follows the diagonal.
    prob_true_raw, prob_pred_raw = calibration_curve(
        y_test, y_prob_raw, n_bins=10, strategy="uniform"
    )
    prob_true_cal, prob_pred_cal = calibration_curve(
        y_test, y_prob_cal, n_bins=10, strategy="uniform"
    )

    # Brier score: mean squared error of probability estimates.
    # Lower is better. Measures both calibration and discrimination.
    brier_raw = np.mean((y_prob_raw - y_test) ** 2)
    brier_cal = np.mean((y_prob_cal - y_test) ** 2)

    print(f"  Model: {best_name}")
    print(f"  Brier score (raw):        {brier_raw:.4f}")
    print(f"  Brier score (calibrated): {brier_cal:.4f}")
    print(f"  {'→ Calibration improved probability estimates' if brier_cal < brier_raw else '→ Raw probabilities were already well-calibrated'}")

    # Plot calibration curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Probability Calibration — {best_name}",
                 fontsize=14, fontweight="bold", y=1.02)

    ax1.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfectly calibrated")
    ax1.plot(prob_pred_raw, prob_true_raw, "s-", color=PALETTE[1],
             label=f"Before (Brier={brier_raw:.4f})")
    ax1.plot(prob_pred_cal, prob_true_cal, "o-", color=PALETTE[7],
             label=f"After (Brier={brier_cal:.4f})")
    ax1.set_xlabel("Mean Predicted Probability")
    ax1.set_ylabel("Fraction of Positives")
    ax1.set_title("Calibration Curve", fontweight="bold")
    ax1.legend(loc="lower right")

    ax2.hist(y_prob_raw, bins=30, alpha=0.5, color=PALETTE[1],
             label="Raw", density=True)
    ax2.hist(y_prob_cal, bins=30, alpha=0.5, color=PALETTE[7],
             label="Calibrated", density=True)
    ax2.set_xlabel("Predicted Probability")
    ax2.set_ylabel("Density")
    ax2.set_title("Probability Distribution", fontweight="bold")
    ax2.legend()

    plt.tight_layout()
    fig.savefig("calibration_plot.png", dpi=150)
    print("✓ Calibration plot saved → calibration_plot.png")
    plt.show()

    return calibrated


# ─────────────────────────────────────────────────────────────────────
# 9. MODEL EVALUATION VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────

def plot_model_evaluation(results: dict, y_test):
    """Generate confusion matrices and ROC curves for all models."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Model Evaluation — Confusion Matrices & ROC Curves",
                 fontsize=16, fontweight="bold", y=1.01)

    model_names = list(results.keys())

    # ---- Row 1: Confusion Matrices ------------------------------------------
    for i, name in enumerate(model_names):
        ax = axes[0, i]
        cm = confusion_matrix(y_test, results[name]["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Rejected", "Approved"],
                    yticklabels=["Rejected", "Approved"],
                    cbar=False, annot_kws={"size": 14})
        ax.set_title(f"{name}", fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    # ---- Row 2: ROC Curves --------------------------------------------------
    # Plot all ROC curves on each subplot for comparison,
    # highlighting the current model.
    for i, name in enumerate(model_names):
        ax = axes[1, i]
        for other_name, other_res in results.items():
            fpr, tpr, _ = roc_curve(y_test, other_res["y_prob"])
            alpha = 1.0 if other_name == name else 0.25
            lw = 2.5 if other_name == name else 1.0
            ax.plot(fpr, tpr, alpha=alpha, lw=lw,
                    label=f'{other_name} ({other_res["roc_auc"]:.3f})')

        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random (0.500)")
        ax.set_title(f"ROC — {name}", fontweight="bold")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(fontsize=7, loc="lower right")

    plt.tight_layout()
    fig.savefig("model_evaluation.png", dpi=150)
    print("✓ Model evaluation saved → model_evaluation.png")
    plt.show()


def plot_model_comparison(results: dict):
    """Bar chart comparing all models across key metrics."""

    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    model_names = list(results.keys())

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metrics))
    width = 0.25

    for i, name in enumerate(model_names):
        values = [results[name][m] for m in metrics]
        bars = ax.bar(x + i * width, values, width, label=name,
                      color=PALETTE[i * 3], edgecolor="white")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", fontsize=8, fontweight="bold")

    ax.set_xticks(x + width)
    ax.set_xticklabels([m.upper().replace("_", " ") for m in metrics])
    ax.set_ylim(0, 1.1)
    ax.set_title("Model Comparison — Key Metrics", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score")
    ax.legend()
    plt.tight_layout()
    fig.savefig("model_comparison.png", dpi=150)
    print("✓ Model comparison saved → model_comparison.png")
    plt.show()


# ─────────────────────────────────────────────────────────────────────
# 10. FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────

def plot_feature_importance(results: dict, X_test, y_test, feature_names):
    """Compare tree-based feature importance with permutation importance.

    Tree-based importance (MDI) measures how much each feature reduces
    impurity across all splits. It is fast but biased toward high-cardinality
    and correlated features.

    Permutation importance shuffles each feature and measures how much
    the model's score drops. It is unbiased but slower. Showing both
    side-by-side reveals if MDI is giving misleading rankings.
    """

    best_name = max(results, key=lambda k: results[k]["f1"])
    best_model = results[best_name]["model"]

    print(f"\n  Best model: {best_name} (F1 = {results[best_name]['f1']:.4f})")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(f"Feature Importance — {best_name}",
                 fontsize=16, fontweight="bold", y=1.01)

    n_features = len(feature_names)
    colors = sns.color_palette("viridis", n_features)

    # ---- Left: Tree-based (MDI) importance ----------------------------------
    if hasattr(best_model, "feature_importances_"):
        mdi = pd.Series(
            best_model.feature_importances_, index=feature_names
        ).sort_values(ascending=True)

        mdi.plot(kind="barh", ax=ax1, color=colors, edgecolor="white")
        ax1.set_title("MDI (Tree-based)", fontweight="bold")
        ax1.set_xlabel("Importance")

        for i, val in enumerate(mdi.values[-3:]):
            ax1.text(val + 0.002, len(mdi) - 3 + i,
                     f"{val:.3f}", va="center", fontweight="bold")

    # ---- Right: Permutation importance --------------------------------------
    # n_repeats=10 shuffles each feature 10 times for stable estimates.
    perm_result = permutation_importance(
        best_model, X_test, y_test,
        n_repeats=10, random_state=42, scoring="f1"
    )
    perm = pd.Series(
        perm_result.importances_mean, index=feature_names
    ).sort_values(ascending=True)

    perm.plot(kind="barh", ax=ax2, color=colors, edgecolor="white")
    ax2.set_title("Permutation Importance", fontweight="bold")
    ax2.set_xlabel("Mean F1 Decrease")

    for i, val in enumerate(perm.values[-3:]):
        ax2.text(val + 0.002, len(perm) - 3 + i,
                 f"{val:.3f}", va="center", fontweight="bold")

    plt.tight_layout()
    fig.savefig("feature_importance.png", dpi=150)
    print("✓ Feature importance saved → feature_importance.png")
    plt.show()

    return perm


# ─────────────────────────────────────────────────────────────────────
# 11. INSIGHTS SUMMARY
# ─────────────────────────────────────────────────────────────────────

def print_insights(df, insights, results):
    """Summarize key findings from EDA and model evaluation."""

    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)

    best_name = max(results, key=lambda k: results[k]["f1"])
    best = results[best_name]
    top_corr = insights["corr_with_target"].index[0]
    top_corr_val = insights["corr_with_target"].values[0]

    insights_text = [
        f"1. The dataset contains {len(df):,} loan applications with a "
        f"{insights['approval_rate']:.1f}% approval rate.",

        f"2. The strongest predictor of loan approval is '{top_corr}' "
        f"(correlation: {top_corr_val:+.3f}).",

        f"3. The best-performing model is {best_name} with an F1 score of "
        f"{best['f1']:.4f} and ROC-AUC of {best['roc_auc']:.4f}.",

        f"4. Cross-validation F1 ({best['cv_f1_mean']:.4f} ± {best['cv_f1_std']:.4f}) "
        f"is consistent with test F1 ({best['f1']:.4f}), indicating the model "
        f"{'generalizes well and is not overfitting.' if abs(best['cv_f1_mean'] - best['f1']) < 0.03 else 'may have some variance between data splits.'}",

        f"5. Precision ({best['precision']:.4f}) and recall ({best['recall']:.4f}) "
        f"are {'well-balanced' if abs(best['precision'] - best['recall']) < 0.05 else 'not equally balanced'} — "
        f"{'the model is equally good at avoiding false approvals and catching true approvals.' if abs(best['precision'] - best['recall']) < 0.05 else 'there is a trade-off between false approvals and missed legitimate applicants.'}",

        f"6. Optimal decision threshold is {best['optimal_threshold']:.2f} "
        f"(vs default 0.5), yielding F1 = {best['f1_at_optimal']:.4f}. "
        f"{'Threshold optimization improved F1.' if best['f1_at_optimal'] > best['f1'] else 'Default threshold was already near-optimal.'}",
    ]

    for line in insights_text:
        print(f"\n  {line}")
    print()


# ─────────────────────────────────────────────────────────────────────
# 12. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # --- Data pipeline ---
    raw_df = load_data("loan_approval_dataset.csv")
    validate_dataset(raw_df)
    clean_df = clean_data(raw_df)
    final_df = engineer_features(clean_df)

    # --- EDA ---
    eda_insights = run_eda(final_df)
    outlier_mask = run_outlier_analysis(final_df)

    # --- Visualizations (EDA) ---
    plot_eda_dashboard(final_df, eda_insights)
    plot_correlation_heatmap(eda_insights)

    # --- Modeling ---
    X_train, X_test, y_train, y_test = prepare_data(final_df)
    model_results = train_models(X_train, X_test, y_train, y_test)

    # --- Outlier impact comparison ---
    outlier_comparison = compare_outlier_impact(final_df, outlier_mask)

    # --- Probability calibration ---
    calibrated_model = calibrate_best_model(
        model_results, X_train, X_test, y_train, y_test
    )

    # --- Visualizations (Models) ---
    plot_model_evaluation(model_results, y_test)
    plot_model_comparison(model_results)
    importances = plot_feature_importance(
        model_results, X_test, y_test, X_train.columns
    )

    # --- Summary ---
    print_insights(final_df, eda_insights, model_results)

    print("✅ Analysis complete. Outputs: eda_dashboard.png, "
          "correlation_heatmap.png, model_evaluation.png, "
          "model_comparison.png, feature_importance.png, "
          "calibration_plot.png")
