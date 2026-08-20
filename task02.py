import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix, 
    roc_curve, 
    auc
)

# ==========================================
# 1. GENERATE & SPLIT DATASET
# ==========================================
# Generate synthetic binary classification dataset
X, y = make_classification(
    n_samples=1000, 
    n_features=10, 
    n_informative=6, 
    n_classes=2, 
    random_state=42
)

# Split into 80% training set and 20% test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================
# 2. INITIALIZE & TRAIN MODELS
# ==========================================
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}

print("=== MODEL PERFORMANCE SUMMARY ===")
for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for ROC Curve
    
    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    results[name] = {
        "model": model,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "accuracy": acc
    }
    print(f"{name} Accuracy: {acc * 100:.2f}%")

# ==========================================
# 3. DETAILED CLASSIFICATION REPORT (BEST MODEL)
# ==========================================
best_model_name = "Random Forest"
print(f"\nDetailed Report for {best_model_name}:")
print(classification_report(y_test, results[best_model_name]["y_pred"]))

# ==========================================
# 4. VISUALIZATION (CONFUSION MATRIX & ROC CURVE)
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Plot 1: Confusion Matrix for Random Forest ---
cm = confusion_matrix(y_test, results[best_model_name]["y_pred"])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
axes[0].set_title(f'Confusion Matrix ({best_model_name})')
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')
axes[0].set_xticklabels(['Class 0', 'Class 1'])
axes[0].set_yticklabels(['Class 0', 'Class 1'])

# --- Plot 2: ROC Curves for All Models ---
for name, data in results.items():
    fpr, tpr, _ = roc_curve(y_test, data["y_prob"])
    roc_auc = auc(fpr, tpr)
    axes[1].plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

# Plot baseline reference line
axes[1].plot([0, 1], [0, 1], 'k--', label='Random Guess (AUC = 0.50)')
axes[1].set_title('Receiver Operating Characteristic (ROC) Curve')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend(loc='lower right')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()