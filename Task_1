# =============================================================================
# TASK 1: Credit Scoring Model
# CodeAlpha Machine Learning Internship
# =============================================================================
# Dataset Source: German Credit Dataset
# Download URL : https://www.kaggle.com/datasets/uciml/german-credit
#              : https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
# Note: This script uses a realistic synthetic dataset with the same structure
#       as the German Credit dataset. Replace the data-loading block with the
#       real CSV once downloaded from the URL above.
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_score,
    recall_score, f1_score, accuracy_score
)
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("       TASK 1: CREDIT SCORING MODEL")
print("       CodeAlpha ML Internship")
print("=" * 60)

# =============================================================================
# STEP 1 — DATA LOADING
# =============================================================================
# --- To use the REAL German Credit Dataset ---
# 1. Download from: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
# 2. Save as 'german_credit.csv' in this folder.
# 3. Uncomment the line below and comment out the synthetic-data block.
# df = pd.read_csv('german_credit.csv')

print("\n[INFO] Generating synthetic German-Credit-style dataset...")
np.random.seed(42)
n = 1000

data = {
    'age':            np.random.randint(18, 75, n),
    'income':         np.random.randint(15000, 120000, n),
    'loan_amount':    np.random.randint(1000, 50000, n),
    'loan_duration':  np.random.randint(6, 72, n),
    'credit_history': np.random.choice(['good', 'fair', 'bad'], n, p=[0.6, 0.3, 0.1]),
    'employment_years': np.random.uniform(0, 30, n).round(1),
    'num_credits':    np.random.randint(1, 5, n),
    'housing':        np.random.choice(['own', 'rent', 'free'], n, p=[0.5, 0.4, 0.1]),
    'savings':        np.random.randint(0, 50000, n),
    'checking_balance': np.random.randint(-500, 20000, n),
    'purpose':        np.random.choice(['car', 'furniture', 'education', 'business', 'other'], n),
    'existing_credits': np.random.randint(1, 4, n),
    'dependents':     np.random.randint(0, 3, n),
    'phone':          np.random.choice([0, 1], n),
    'foreign_worker': np.random.choice([0, 1], n, p=[0.05, 0.95]),
}
df = pd.DataFrame(data)

# Build a realistic target based on financial logic
score = (
      (df['income'] / 120000) * 40
    + (df['savings'] / 50000) * 20
    + (df['employment_years'] / 30) * 15
    - (df['loan_amount'] / 50000) * 20
    - (df['loan_duration'] / 72) * 10
    + (df['credit_history'] == 'good').astype(int) * 15
    - (df['credit_history'] == 'bad').astype(int) * 25
    + np.random.normal(0, 5, n)
)
df['creditworthy'] = (score > score.median()).astype(int)

print(f"Dataset shape: {df.shape}")
print(f"Target distribution:\n{df['creditworthy'].value_counts()}\n")

# =============================================================================
# STEP 2 — FEATURE ENGINEERING & PREPROCESSING
# =============================================================================
print("[INFO] Preprocessing data...")

# Encode categorical features
le = LabelEncoder()
for col in ['credit_history', 'housing', 'purpose']:
    df[col] = le.fit_transform(df[col])

# Debt-to-income ratio (feature engineering)
df['debt_to_income'] = df['loan_amount'] / (df['income'] + 1)
df['savings_to_loan'] = df['savings'] / (df['loan_amount'] + 1)

X = df.drop('creditworthy', axis=1)
y = df['creditworthy']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# =============================================================================
# STEP 3 — MODEL TRAINING
# =============================================================================
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree':        DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest':        RandomForestClassifier(n_estimators=100, random_state=42),
}

results = {}
print("\n" + "-" * 60)
print(f"{'Model':<22} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6}")
print("-" * 60)

for name, model in models.items():
    X_tr = X_train_sc if name == 'Logistic Regression' else X_train
    X_te = X_test_sc  if name == 'Logistic Regression' else X_test

    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    acc   = accuracy_score(y_test, y_pred)
    prec  = precision_score(y_test, y_pred)
    rec   = recall_score(y_test, y_pred)
    f1    = f1_score(y_test, y_pred)
    auc   = roc_auc_score(y_test, y_prob)

    results[name] = {'model': model, 'y_pred': y_pred, 'y_prob': y_prob,
                     'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'auc': auc,
                     'X_te': X_te}
    print(f"{name:<22} {acc:>6.3f} {prec:>6.3f} {rec:>6.3f} {f1:>6.3f} {auc:>6.3f}")

print("-" * 60)

# =============================================================================
# STEP 4 — DETAILED REPORT FOR BEST MODEL
# =============================================================================
best_name = max(results, key=lambda k: results[k]['auc'])
best = results[best_name]
print(f"\n[INFO] Best model: {best_name} (AUC = {best['auc']:.3f})")
print("\nClassification Report:")
print(classification_report(y_test, best['y_pred'],
                             target_names=['Not Creditworthy', 'Creditworthy']))

# =============================================================================
# STEP 5 — VISUALISATIONS
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Task 1 — Credit Scoring Model Results', fontsize=14, fontweight='bold')

# --- ROC Curves ---
ax = axes[0]
for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r['y_prob'])
    ax.plot(fpr, tpr, label=f"{name} (AUC={r['auc']:.3f})")
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves')
ax.legend(fontsize=8)

# --- Confusion Matrix ---
ax = axes[1]
cm = confusion_matrix(y_test, best['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Not CW', 'CW'], yticklabels=['Not CW', 'CW'])
ax.set_title(f'Confusion Matrix\n({best_name})')
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')

# --- Feature Importance (Random Forest) ---
ax = axes[2]
rf = results['Random Forest']['model']
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
imp.plot(kind='barh', ax=ax, color='steelblue')
ax.set_title('Top 10 Feature Importances\n(Random Forest)')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig('task1_credit_scoring_results.png', dpi=150, bbox_inches='tight')
print("\n[INFO] Plot saved as 'task1_credit_scoring_results.png'")
print("\n[DONE] Task 1 — Credit Scoring Model complete.")
