# =============================================================================
# TASK 4: Disease Prediction from Medical Data
# CodeAlpha Machine Learning Internship
# =============================================================================
# Datasets Used:
#   1. Breast Cancer Wisconsin — sklearn built-in (no download needed)
#   2. Diabetes (Pima Indians) — sklearn built-in (no download needed)
#
# For additional real-world datasets, download from:
#   Heart Disease UCI  : https://archive.ics.uci.edu/dataset/45/heart+disease
#   Diabetes (Kaggle)  : https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
#   Breast Cancer      : https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, accuracy_score,
    precision_score, recall_score, f1_score
)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("      TASK 4: DISEASE PREDICTION FROM MEDICAL DATA")
print("      CodeAlpha ML Internship")
print("=" * 60)

# =============================================================================
# HELPER FUNCTION — evaluate and store metrics
# =============================================================================
def evaluate_model(name, model, X_tr, X_te, y_tr, y_te, scaled=False, results=None):
    scaler = StandardScaler()
    if scaled:
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1] if hasattr(model, 'predict_proba') else None

    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec  = recall_score(y_te, y_pred, zero_division=0)
    f1   = f1_score(y_te, y_pred, zero_division=0)
    auc  = roc_auc_score(y_te, y_prob) if y_prob is not None else 0.0

    if results is not None:
        results[name] = {'model': model, 'y_pred': y_pred, 'y_prob': y_prob,
                         'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'auc': auc}
    return acc, prec, rec, f1, auc, y_pred, y_prob

# =============================================================================
# DATASET A — BREAST CANCER WISCONSIN
# =============================================================================
print("\n" + "=" * 60)
print("  DATASET A: Breast Cancer Wisconsin")
print("  Task: Classify tumours as Malignant (1) or Benign (0)")
print("=" * 60)

bc = load_breast_cancer()
X_bc = pd.DataFrame(bc.data, columns=bc.feature_names)
y_bc = bc.target   # 0 = malignant, 1 = benign

print(f"\nDataset shape : {X_bc.shape}")
print(f"Classes       : {bc.target_names}")
print(f"Class balance : Benign={sum(y_bc==1)}, Malignant={sum(y_bc==0)}")
print("\nFeature statistics:")
print(X_bc.describe().round(2))

X_tr_bc, X_te_bc, y_tr_bc, y_te_bc = train_test_split(
    X_bc, y_bc, test_size=0.2, random_state=42, stratify=y_bc
)

MODELS_BC = {
    'Logistic Regression':  (LogisticRegression(max_iter=1000, random_state=42),  True),
    'SVM':                  (SVC(probability=True, kernel='rbf', random_state=42), True),
    'Random Forest':        (RandomForestClassifier(n_estimators=100, random_state=42), False),
    'XGBoost':              (XGBClassifier(n_estimators=100, random_state=42,
                                           eval_metric='logloss', verbosity=0), False),
    'Gradient Boosting':    (GradientBoostingClassifier(n_estimators=100, random_state=42), False),
}

results_bc = {}
print("\n" + "-" * 70)
print(f"{'Model':<22} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6}")
print("-" * 70)
for mname, (model, scale) in MODELS_BC.items():
    acc, prec, rec, f1, auc, y_pred, y_prob = evaluate_model(
        mname, model, X_tr_bc.copy(), X_te_bc.copy(),
        y_tr_bc, y_te_bc, scaled=scale, results=results_bc
    )
    print(f"{mname:<22} {acc:>6.3f} {prec:>6.3f} {rec:>6.3f} {f1:>6.3f} {auc:>6.3f}")
print("-" * 70)

best_bc = max(results_bc, key=lambda k: results_bc[k]['auc'])
print(f"\n[INFO] Best model (Breast Cancer): {best_bc}  AUC={results_bc[best_bc]['auc']:.3f}")
print("\nDetailed Report:")
print(classification_report(y_te_bc, results_bc[best_bc]['y_pred'],
                             target_names=['Malignant', 'Benign']))

# =============================================================================
# DATASET B — DIABETES (Pima Indians synthetic proxy)
# =============================================================================
print("\n" + "=" * 60)
print("  DATASET B: Diabetes Prediction")
print("  Source: Realistic synthetic data matching Pima Indians features")
print("  (Real Kaggle dataset: kaggle.com/datasets/uciml/pima-indians-diabetes-database)")
print("=" * 60)

np.random.seed(0)
n = 768
diab_data = {
    'Pregnancies':        np.random.randint(0, 18, n),
    'Glucose':            np.random.normal(120, 30, n).clip(0, 200).astype(int),
    'BloodPressure':      np.random.normal(70, 12, n).clip(0, 120).astype(int),
    'SkinThickness':      np.random.normal(25, 10, n).clip(0, 60).astype(int),
    'Insulin':            np.random.exponential(80, n).clip(0, 500).astype(int),
    'BMI':                np.random.normal(32, 7, n).clip(15, 65).round(1),
    'DiabetesPedigreeFunction': np.random.exponential(0.5, n).clip(0.08, 2.5).round(3),
    'Age':                np.random.randint(21, 82, n),
}
df_diab = pd.DataFrame(diab_data)

# Logistic-like probability → binary outcome
log_odds = (
    -6
    + 0.03  * df_diab['Glucose']
    + 0.015 * df_diab['BMI']
    + 0.01  * df_diab['Age']
    + 0.5   * df_diab['DiabetesPedigreeFunction']
    + 0.05  * df_diab['Pregnancies']
    + np.random.normal(0, 0.8, n)
)
prob = 1 / (1 + np.exp(-log_odds))
df_diab['Outcome'] = (prob > 0.5).astype(int)

X_diab = df_diab.drop('Outcome', axis=1)
y_diab = df_diab['Outcome']

print(f"\nDataset shape : {X_diab.shape}")
print(f"Class balance : No Diabetes={sum(y_diab==0)}, Diabetes={sum(y_diab==1)}")

X_tr_d, X_te_d, y_tr_d, y_te_d = train_test_split(
    X_diab, y_diab, test_size=0.2, random_state=42, stratify=y_diab
)

MODELS_D = {
    'Logistic Regression': (LogisticRegression(max_iter=1000, random_state=42), True),
    'SVM':                 (SVC(probability=True, kernel='rbf', random_state=42), True),
    'Random Forest':       (RandomForestClassifier(n_estimators=100, random_state=42), False),
    'XGBoost':             (XGBClassifier(n_estimators=100, random_state=42,
                                          eval_metric='logloss', verbosity=0), False),
}

results_d = {}
print("\n" + "-" * 70)
print(f"{'Model':<22} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6}")
print("-" * 70)
for mname, (model, scale) in MODELS_D.items():
    acc, prec, rec, f1, auc, y_pred, y_prob = evaluate_model(
        mname, model, X_tr_d.copy(), X_te_d.copy(),
        y_tr_d, y_te_d, scaled=scale, results=results_d
    )
    print(f"{mname:<22} {acc:>6.3f} {prec:>6.3f} {rec:>6.3f} {f1:>6.3f} {auc:>6.3f}")
print("-" * 70)

best_d = max(results_d, key=lambda k: results_d[k]['auc'])
print(f"\n[INFO] Best model (Diabetes): {best_d}  AUC={results_d[best_d]['auc']:.3f}")
print("\nDetailed Report:")
print(classification_report(y_te_d, results_d[best_d]['y_pred'],
                             target_names=['No Diabetes', 'Diabetes']))

# =============================================================================
# VISUALISATIONS — Combined dashboard
# =============================================================================
fig = plt.figure(figsize=(20, 10))
fig.suptitle('Task 4 — Disease Prediction Dashboard', fontsize=14, fontweight='bold')

# --- ROC Curves: Breast Cancer ---
ax1 = fig.add_subplot(2, 4, 1)
for nm, r in results_bc.items():
    if r['y_prob'] is not None:
        fpr, tpr, _ = roc_curve(y_te_bc, r['y_prob'])
        ax1.plot(fpr, tpr, label=f"{nm} ({r['auc']:.2f})", linewidth=1.2)
ax1.plot([0,1],[0,1],'k--'); ax1.legend(fontsize=6)
ax1.set_title('ROC — Breast Cancer'); ax1.set_xlabel('FPR'); ax1.set_ylabel('TPR')

# --- Confusion Matrix: Best BC model ---
ax2 = fig.add_subplot(2, 4, 2)
cm_bc = confusion_matrix(y_te_bc, results_bc[best_bc]['y_pred'])
sns.heatmap(cm_bc, annot=True, fmt='d', cmap='Blues', ax=ax2,
            xticklabels=['Mal','Ben'], yticklabels=['Mal','Ben'])
ax2.set_title(f'Confusion — {best_bc}\n(Breast Cancer)')
ax2.set_ylabel('Actual'); ax2.set_xlabel('Predicted')

# --- Feature Importance: BC Random Forest ---
ax3 = fig.add_subplot(2, 4, 3)
rf_bc = results_bc['Random Forest']['model']
feat_imp = pd.Series(rf_bc.feature_importances_, index=X_bc.columns).sort_values(ascending=False).head(8)
feat_imp.plot(kind='barh', ax=ax3, color='steelblue')
ax3.set_title('Top Feature Importances\n(Breast Cancer RF)'); ax3.invert_yaxis()
ax3.tick_params(labelsize=7)

# --- Accuracy Comparison: BC ---
ax4 = fig.add_subplot(2, 4, 4)
model_names = list(results_bc.keys())
accs = [results_bc[m]['auc'] for m in model_names]
colors = ['gold' if m == best_bc else 'steelblue' for m in model_names]
bars = ax4.bar([m.replace(' ', '\n') for m in model_names], accs, color=colors)
ax4.set_title('AUC Comparison\n(Breast Cancer)')
ax4.set_ylabel('ROC-AUC'); ax4.set_ylim(0.8, 1.01)
ax4.tick_params(axis='x', labelsize=7)
for bar, v in zip(bars, accs):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f'{v:.3f}', ha='center', va='bottom', fontsize=7)

# --- ROC Curves: Diabetes ---
ax5 = fig.add_subplot(2, 4, 5)
for nm, r in results_d.items():
    if r['y_prob'] is not None:
        fpr, tpr, _ = roc_curve(y_te_d, r['y_prob'])
        ax5.plot(fpr, tpr, label=f"{nm} ({r['auc']:.2f})", linewidth=1.2)
ax5.plot([0,1],[0,1],'k--'); ax5.legend(fontsize=7)
ax5.set_title('ROC — Diabetes'); ax5.set_xlabel('FPR'); ax5.set_ylabel('TPR')

# --- Confusion Matrix: Best Diabetes model ---
ax6 = fig.add_subplot(2, 4, 6)
cm_d = confusion_matrix(y_te_d, results_d[best_d]['y_pred'])
sns.heatmap(cm_d, annot=True, fmt='d', cmap='Oranges', ax=ax6,
            xticklabels=['No Diab','Diab'], yticklabels=['No Diab','Diab'])
ax6.set_title(f'Confusion — {best_d}\n(Diabetes)')
ax6.set_ylabel('Actual'); ax6.set_xlabel('Predicted')

# --- Feature importance: Diabetes XGBoost ---
ax7 = fig.add_subplot(2, 4, 7)
xgb_d = results_d['XGBoost']['model']
feat_imp_d = pd.Series(xgb_d.feature_importances_, index=X_diab.columns).sort_values(ascending=False)
feat_imp_d.plot(kind='barh', ax=ax7, color='tomato')
ax7.set_title('Feature Importances\n(Diabetes XGBoost)'); ax7.invert_yaxis()
ax7.tick_params(labelsize=8)

# --- AUC Comparison: Diabetes ---
ax8 = fig.add_subplot(2, 4, 8)
model_names_d = list(results_d.keys())
aucs_d = [results_d[m]['auc'] for m in model_names_d]
colors_d = ['gold' if m == best_d else 'tomato' for m in model_names_d]
bars2 = ax8.bar([m.replace(' ', '\n') for m in model_names_d], aucs_d, color=colors_d)
ax8.set_title('AUC Comparison\n(Diabetes)')
ax8.set_ylabel('ROC-AUC'); ax8.set_ylim(0.5, 1.01)
ax8.tick_params(axis='x', labelsize=7)
for bar, v in zip(bars2, aucs_d):
    ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f'{v:.3f}', ha='center', va='bottom', fontsize=7)

plt.tight_layout()
plt.savefig('task4_disease_prediction_results.png', dpi=150, bbox_inches='tight')
print("\n[INFO] Dashboard saved as 'task4_disease_prediction_results.png'")
print("\n[DONE] Task 4 — Disease Prediction complete.")
