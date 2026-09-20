"""
Heart Disease Prediction - Model Training Pipeline
====================================================
Trains and compares several classifiers on the UCI Cleveland Heart
Disease dataset, then saves the best-performing model + scaler for
use by the Streamlit app.

Run with:  python train_model.py
"""

import json
import warnings

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

matplotlib.use("Agg")
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

DATA_PATH = "data/heart.csv"
REPORT_DIR = "report/assets"
MODEL_DIR = "model"

import os

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)
print(df.describe().T)

# ---------------------------------------------------------------
# 2. EDA plots (saved for the report)
# ---------------------------------------------------------------
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="target", palette=["#4C72B0", "#DD8452"])
plt.title("Target Distribution (0 = No Disease, 1 = Disease)")
plt.xlabel("Heart Disease")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{REPORT_DIR}/target_distribution.png", dpi=150)
plt.close()

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{REPORT_DIR}/correlation_heatmap.png", dpi=150)
plt.close()

plt.figure(figsize=(6, 4))
sns.histplot(data=df, x="age", hue="target", multiple="stack", bins=20, palette=["#4C72B0", "#DD8452"])
plt.title("Age Distribution by Heart Disease Status")
plt.tight_layout()
plt.savefig(f"{REPORT_DIR}/age_distribution.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 3. Train / test split + scaling
# ---------------------------------------------------------------
X = df[FEATURE_NAMES]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------
# 4. Train + compare multiple models
# ---------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=200),
    "SVM (RBF)": SVC(probability=True, random_state=42),
}

results = []
fitted_models = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    proba = model.predict_proba(X_test_scaled)[:, 1]

    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "F1 Score": f1_score(y_test, preds),
        "ROC AUC": roc_auc_score(y_test, proba),
        "CV Mean Accuracy": cv_scores.mean(),
        "CV Std": cv_scores.std(),
    }
    results.append(metrics)
    fitted_models[name] = model
    print(f"\n{name}")
    print(classification_report(y_test, preds))

results_df = pd.DataFrame(results).sort_values("ROC AUC", ascending=False)
print("\n=== Model Comparison ===")
print(results_df.to_string(index=False))
results_df.to_csv(f"{REPORT_DIR}/model_comparison.csv", index=False)

# ---------------------------------------------------------------
# 5. Hyperparameter tuning for the best model (Random Forest)
# ---------------------------------------------------------------
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5],
}
grid = GridSearchCV(
    RandomForestClassifier(random_state=42), param_grid, cv=5, scoring="roc_auc", n_jobs=-1
)
grid.fit(X_train_scaled, y_train)
best_model = grid.best_estimator_
print("\nBest RF params:", grid.best_params_)

best_preds = best_model.predict(X_test_scaled)
best_proba = best_model.predict_proba(X_test_scaled)[:, 1]
final_metrics = {
    "Accuracy": accuracy_score(y_test, best_preds),
    "Precision": precision_score(y_test, best_preds),
    "Recall": recall_score(y_test, best_preds),
    "F1 Score": f1_score(y_test, best_preds),
    "ROC AUC": roc_auc_score(y_test, best_proba),
}
print("\nFinal tuned model metrics:", final_metrics)

with open(f"{REPORT_DIR}/final_metrics.json", "w") as f:
    json.dump({"best_params": grid.best_params_, "metrics": final_metrics}, f, indent=2)

# ---------------------------------------------------------------
# 6. Plots for the report: confusion matrix, ROC curve, feature importance
# ---------------------------------------------------------------
plt.figure(figsize=(5, 5))
cm = confusion_matrix(y_test, best_preds)
ConfusionMatrixDisplay(cm, display_labels=["No Disease", "Disease"]).plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix - Tuned Random Forest")
plt.tight_layout()
plt.savefig(f"{REPORT_DIR}/confusion_matrix.png", dpi=150)
plt.close()

plt.figure(figsize=(6, 5))
RocCurveDisplay.from_estimator(best_model, X_test_scaled, y_test)
plt.title("ROC Curve - Tuned Random Forest")
plt.tight_layout()
plt.savefig(f"{REPORT_DIR}/roc_curve.png", dpi=150)
plt.close()

importances = pd.Series(best_model.feature_importances_, index=FEATURE_NAMES).sort_values()
plt.figure(figsize=(7, 5))
importances.plot(kind="barh", color="#4C72B0")
plt.title("Feature Importance - Random Forest")
plt.tight_layout()
plt.savefig(f"{REPORT_DIR}/feature_importance.png", dpi=150)
plt.close()

plt.figure(figsize=(7, 5))
sns.barplot(data=results_df, x="ROC AUC", y="Model", palette="viridis")
plt.title("Model Comparison (ROC AUC)")
plt.tight_layout()
plt.savefig(f"{REPORT_DIR}/model_comparison.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 7. Persist model + scaler for the app
# ---------------------------------------------------------------
joblib.dump(best_model, f"{MODEL_DIR}/heart_disease_model.pkl")
joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
joblib.dump(FEATURE_NAMES, f"{MODEL_DIR}/feature_names.pkl")

print("\nSaved model, scaler, and report assets.")
