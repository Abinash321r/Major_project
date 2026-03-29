import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    log_loss
)

import joblib

# ============================================================
# GLOBAL PLOT STYLE
# ============================================================
plt.style.use("default")
plt.rcParams.update({
    "figure.figsize": (6, 4),
    "figure.dpi": 300,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "lines.linewidth": 2,
    "axes.spines.top": False,
    "axes.spines.right": False
})

# ============================================================
# 1. LOAD DATASET
# ============================================================
with open("light_samples.json", "r") as f:
    samples = json.load(f)

X, y = [], []
for s in samples:
    X.append([
        s["meanBrightness"],
        s["contrast"],
        s["brightnessRange"]
    ])
    y.append(int(s["label"]))

X = np.array(X, dtype=float)
y = np.array(y, dtype=int)

print("Total samples:", len(y))

# ============================================================
# 2. FEATURE NORMALIZATION
# ============================================================
X_norm = np.column_stack([
    X[:, 0] / 255.0,
    X[:, 1] / 128.0,
    X[:, 2] / 255.0
])

# ============================================================
# 3. TRAIN–TEST SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_norm,
    y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# ============================================================
# 4. TRAIN MODEL
# ============================================================
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# ============================================================
# 5. EVALUATION
# ============================================================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, digits=4)

print("\n=== ACCURACY ===")
print(accuracy)

print("\n=== CONFUSION MATRIX ===")
print(cm)

print("\n=== CLASSIFICATION REPORT ===")
print(report)

# ============================================================
# 6. CONFUSION MATRIX PLOT (UPDATED)
# ============================================================
plt.figure()
plt.imshow(cm, cmap="Blues")

# Axis labels
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")

# Class names
class_names = ["Bright", "Low Light"]

# Set ticks
plt.xticks([0, 1], class_names)
plt.yticks([0, 1], class_names)

# Add numbers inside cells
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j],
                 ha="center", va="center", color="black")

plt.colorbar()
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

# ============================================================
# 7. LEARNING CURVE
# ============================================================
train_sizes, train_scores, val_scores = learning_curve(
    model,
    X_norm,
    y,
    cv=5,
    scoring="accuracy",
    train_sizes=np.linspace(0.1, 1.0, 8)
)

train_acc_mean = train_scores.mean(axis=1)
val_acc_mean = val_scores.mean(axis=1)

plt.figure()
plt.plot(train_sizes, train_acc_mean, label="Training Accuracy")
plt.plot(train_sizes, val_acc_mean, label="Validation Accuracy")
plt.xlabel("Training Samples")
plt.ylabel("Accuracy")
plt.title("Learning Curve")
plt.legend()
plt.tight_layout()
plt.savefig("learning_curve.png")
plt.close()

# ============================================================
# 8. LOSS CURVE
# ============================================================
train_losses = []
val_losses = []

for size in train_sizes:
    size = int(size)

    X_part = X_train[:size]
    y_part = y_train[:size]

    temp_model = LogisticRegression(max_iter=2000)
    temp_model.fit(X_part, y_part)

    train_prob = temp_model.predict_proba(X_part)
    val_prob = temp_model.predict_proba(X_test)

    train_losses.append(log_loss(y_part, train_prob))
    val_losses.append(log_loss(y_test, val_prob))

plt.figure()
plt.plot(train_sizes, train_losses, label="Training Loss")
plt.plot(train_sizes, val_losses, label="Validation Loss")
plt.xlabel("Training Samples")
plt.ylabel("Log Loss")
plt.title("Loss Curve")
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve.png")
plt.close()

# ============================================================
# 9. SAVE MODEL
# ============================================================
joblib.dump(model, "light_model.pkl")

print("\nSaved output files:")
print("- confusion_matrix.png")
print("- learning_curve.png")
print("- loss_curve.png")
print("- light_model.pkl")