#!/usr/bin/env python3
"""
Standalone validation tests for the Bank Marketing predictive pipeline.
Run with: python tests/test_pipeline.py
These tests verify data integrity, preprocessing correctness, and model validity
without requiring the full notebook execution.
"""
import sys
import os
import numpy as np
import pandas as pd

# ── Configuration ────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'bank-full.csv')
RANDOM_STATE = 42
EXPECTED_ROWS = 45211
EXPECTED_COLS = 17
TARGET_COL = 'y'
LEAKAGE_COL = 'duration'

passed = 0
failed = 0

def test(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name} {f'-- {detail}' if detail else ''}")

# ══════════════════════════════════════════════════════════════════════════════
# TEST SUITE 1: Data Integrity
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST SUITE 1: Data Integrity")
print("=" * 60)

df = pd.read_csv(DATA_PATH, sep=';')

test("Dataset loads successfully", df is not None)
test(f"Row count is {EXPECTED_ROWS}", len(df) == EXPECTED_ROWS, f"got {len(df)}")
test(f"Column count is {EXPECTED_COLS}", len(df.columns) == EXPECTED_COLS, f"got {len(df.columns)}")
test("No null values in dataset", df.isnull().sum().sum() == 0, f"found {df.isnull().sum().sum()} nulls")
test("Target column exists", TARGET_COL in df.columns)
test("Target has exactly 2 classes", df[TARGET_COL].nunique() == 2, f"got {df[TARGET_COL].nunique()}")
test("Duration column exists (for leakage check)", LEAKAGE_COL in df.columns)

# Class imbalance check
pos_rate = (df[TARGET_COL] == 'yes').mean()
test("Positive class rate ~11.7%", 0.10 < pos_rate < 0.13, f"got {pos_rate:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# TEST SUITE 2: Preprocessing Validation
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST SUITE 2: Preprocessing Validation")
print("=" * 60)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Prepare features
df_model = df.copy()
df_model['y'] = (df_model['y'] == 'yes').astype(int)
X = df_model.drop(columns=['y', LEAKAGE_COL])
y = df_model['y']

test("Duration excluded from features", LEAKAGE_COL not in X.columns)
test("Target not in features", TARGET_COL not in X.columns or (TARGET_COL in X.columns and X[TARGET_COL].dtype != object))

# Check feature types
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()
test("Numeric features identified", len(numeric_features) > 0, f"found {len(numeric_features)}")
test("Categorical features identified", len(categorical_features) > 0, f"found {len(categorical_features)}")
test("All columns accounted for", len(numeric_features) + len(categorical_features) == len(X.columns))

# Stratified split
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=RANDOM_STATE, stratify=y_temp
)

test("Train size ~60%", abs(len(X_train) / len(X) - 0.60) < 0.01, f"got {len(X_train)/len(X):.3f}")
test("Val size ~20%", abs(len(X_val) / len(X) - 0.20) < 0.01, f"got {len(X_val)/len(X):.3f}")
test("Test size ~20%", abs(len(X_test) / len(X) - 0.20) < 0.01, f"got {len(X_test)/len(X):.3f}")

# Stratification check
train_rate = y_train.mean()
val_rate = y_val.mean()
test_rate = y_test.mean()
test("Stratification preserved (train)", abs(train_rate - pos_rate) < 0.005, f"train={train_rate:.4f}")
test("Stratification preserved (val)", abs(val_rate - pos_rate) < 0.005, f"val={val_rate:.4f}")
test("Stratification preserved (test)", abs(test_rate - pos_rate) < 0.005, f"test={test_rate:.4f}")

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop=None, sparse_output=False, handle_unknown='ignore'), categorical_features)
    ]
)
X_train_proc = preprocessor.fit_transform(X_train)
X_val_proc = preprocessor.transform(X_val)
X_test_proc = preprocessor.transform(X_test)

test("No NaN after preprocessing (train)", not np.isnan(X_train_proc).any())
test("No NaN after preprocessing (val)", not np.isnan(X_val_proc).any())
test("No NaN after preprocessing (test)", not np.isnan(X_test_proc).any())
test("Train/Val/Test same feature count", X_train_proc.shape[1] == X_val_proc.shape[1] == X_test_proc.shape[1])

# Scaling check: numeric features should have mean ~0, std ~1 on training data
numeric_transformed = X_train_proc[:, :len(numeric_features)]
test("Numeric features scaled (mean ~0)", abs(numeric_transformed.mean()) < 0.01, f"mean={numeric_transformed.mean():.4f}")
test("Numeric features scaled (std ~1)", abs(numeric_transformed.std() - 1.0) < 0.05, f"std={numeric_transformed.std():.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# TEST SUITE 3: Model Sanity Checks
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST SUITE 3: Model Sanity Checks")
print("=" * 60)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Quick baseline model to verify pipeline works end-to-end
lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE)
lr.fit(X_train_proc, y_train)
y_val_proba = lr.predict_proba(X_val_proc)[:, 1]
auc = roc_auc_score(y_val, y_val_proba)

test("Baseline model trains without error", True)
test("Predictions are probabilities [0, 1]", y_val_proba.min() >= 0 and y_val_proba.max() <= 1)
test("Baseline AUC > 0.70 (above random)", auc > 0.70, f"AUC={auc:.4f}")
test("Baseline AUC < 0.90 (no leakage)", auc < 0.90, f"AUC={auc:.4f}")

# Leakage detection test: model WITH duration should have suspiciously high AUC
X_with_dur = df_model.drop(columns=['y'])  # includes duration
X_temp_d, X_test_d, y_temp_d, y_test_d = train_test_split(X_with_dur, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
X_train_d, X_val_d, y_train_d, y_val_d = train_test_split(X_temp_d, y_temp_d, test_size=0.25, random_state=RANDOM_STATE, stratify=y_temp_d)

preprocessor_d = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features + [LEAKAGE_COL]),
        ('cat', OneHotEncoder(drop=None, sparse_output=False, handle_unknown='ignore'), categorical_features)
    ]
)
X_train_d_proc = preprocessor_d.fit_transform(X_train_d)
X_val_d_proc = preprocessor_d.transform(X_val_d)

lr_d = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE)
lr_d.fit(X_train_d_proc, y_train_d)
auc_d = roc_auc_score(y_val_d, lr_d.predict_proba(X_val_d_proc)[:, 1])

test("Leakage detection: duration inflates AUC", auc_d - auc > 0.05, f"with={auc_d:.4f}, without={auc:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# TEST SUITE 4: Output File Checks
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST SUITE 4: Output File Checks")
print("=" * 60)

project_dir = os.path.join(os.path.dirname(__file__), '..')

required_files = [
    'Bank_Marketing_Analysis.ipynb',
    'report.pdf',
    'requirements.txt',
    'README.md',
    'environment.yml',
]
for f in required_files:
    path = os.path.join(project_dir, f)
    test(f"{f} exists", os.path.exists(path))

# Check for expected figures
expected_figures = [
    'fig_target_distribution.png',
    'fig_correlation_matrix.png',
    'fig_roc_curves.png',
    'fig_feature_importance.png',
    'fig_confusion_matrix_standalone.png',
    'fig_fairness_audit.png',
    'fig_learning_curve.png',
    'fig_temporal_analysis.png',
    'fig_unknown_analysis.png',
    'fig_threshold_analysis.png',
]
for f in expected_figures:
    path = os.path.join(project_dir, f)
    test(f"Figure {f} exists", os.path.exists(path))

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
total = passed + failed
print(f"RESULTS: {passed}/{total} tests passed, {failed} failed")
print("=" * 60)

if failed > 0:
    print("\n⚠ Some tests failed - review the output above.")
    sys.exit(1)
else:
    print("\n✅ All tests passed.")
    sys.exit(0)
