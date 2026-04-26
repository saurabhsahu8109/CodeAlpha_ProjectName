# CodeAlpha_ProjectName

# CodeAlpha Machine Learning Internship
## Submitted Tasks: Task 1, Task 3, Task 4

---

## 📁 Repository Structure

```
CodeAlpha_ML_Internship/
├── Task1_Credit_Scoring_Model.py           # Task 1 — Credit Scoring
├── Task3_Handwritten_Character_Recognition.py  # Task 3 — CNN digit classifier
├── Task4_Disease_Prediction.py             # Task 4 — Disease prediction
├── CodeAlpha_ML_All_Tasks.py               # Combined runner (all 3 tasks)
├── requirements.txt                        # Python dependencies
└── README.md                               # This file
```

---

## ✅ Task 1 — Credit Scoring Model

**Objective:** Predict an individual's creditworthiness from financial data.

**Dataset:**
- Style: German Credit Dataset
- Real dataset URL: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
- Kaggle mirror: https://www.kaggle.com/datasets/uciml/german-credit
- The script generates a realistic synthetic dataset with the same features if the real CSV is not present.

**Approach:**
- Feature engineering: debt-to-income ratio, savings-to-loan ratio
- Models: Logistic Regression, Decision Tree, Random Forest
- Evaluation: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Visualization: ROC curves, Confusion Matrix, Feature Importances

**Results:**
| Model | Accuracy | AUC |
|---|---|---|
| Logistic Regression | 92.0% | 0.984 |
| Decision Tree | 85.0% | 0.900 |
| Random Forest | 89.5% | 0.969 |

---

## ✅ Task 3 — Handwritten Character Recognition

**Objective:** Identify handwritten digits (0–9) using a Convolutional Neural Network.

**Dataset:**
- Used: Scikit-learn Digits dataset (1,797 samples, 8×8 pixel images, offline, no download needed)
- Full MNIST (28×28): https://www.kaggle.com/datasets/hojjatk/mnist-dataset
- EMNIST (letters): https://www.kaggle.com/datasets/crawford/emnist

**Approach:**
- CNN architecture: Conv2D → BatchNorm → MaxPooling → Dropout → Dense
- Optimizer: Adam with ReduceLROnPlateau and EarlyStopping
- Evaluation: Accuracy, Classification Report, Confusion Matrix

**Results:**
- Test Accuracy: **99.17%**
- Test Loss: 0.0242

---

## ✅ Task 4 — Disease Prediction from Medical Data

**Objective:** Predict disease presence using patient medical data.

**Datasets:**
1. **Breast Cancer Wisconsin** — built into scikit-learn (no download needed)
   - Real data: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
2. **Diabetes (Pima Indians style)** — realistic synthetic data
   - Real data: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

**Approach:**
- Models: Logistic Regression, SVM, Random Forest, XGBoost, Gradient Boosting
- Feature importance analysis
- Full evaluation dashboard (ROC curves, confusion matrices, AUC comparison)

**Results (Breast Cancer):**
| Model | AUC |
|---|---|
| Logistic Regression | 0.995 |
| SVM | 0.995 |
| Random Forest | 0.994 |

---

## 🔧 Setup Instructions

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run individual tasks
```bash
python Task1_Credit_Scoring_Model.py
python Task3_Handwritten_Character_Recognition.py
python Task4_Disease_Prediction.py
```

### 3. Run all tasks at once
```bash
python CodeAlpha_ML_All_Tasks.py
```

---

## 📦 Requirements

See `requirements.txt` for the full list. Main libraries used:
- scikit-learn
- TensorFlow / Keras
- XGBoost
- pandas, numpy
- matplotlib, seaborn

---

## 🌐 Dataset Download Instructions

| Task | Dataset | URL |
|---|---|---|
| Task 1 | German Credit | https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data |
| Task 3 | MNIST Digits | https://www.kaggle.com/datasets/hojjatk/mnist-dataset |
| Task 3 | EMNIST | https://www.kaggle.com/datasets/crawford/emnist |
| Task 4 | Breast Cancer | Built-in sklearn / https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data |
| Task 4 | Diabetes | https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database |

> Note: Tasks 3 and 4 work fully offline using built-in sklearn datasets. Task 1 uses a synthetic dataset with the same structure as the German Credit dataset.

---

*CodeAlpha Machine Learning Internship | Completed Tasks: 1, 3, 4*
