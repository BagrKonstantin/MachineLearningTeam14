import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

SEED, KERNEL, C_VALUES = 42, "rbf", [0.01, 1, 100]

# Dataset 1 (B.I.1)
X, y = make_blobs(n_samples=80, centers=[[-1.5, -1.5], [1.5, 1.5]],
                  cluster_std=1.0, random_state=42)

# D.1
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y,
                                          random_state=SEED)
print("D.1  C | training accuracy")
train_acc = {}
for C in C_VALUES:
    model = SVC(C=C, kernel=KERNEL).fit(X_tr, y_tr)
    train_acc[C] = accuracy_score(y_tr, model.predict(X_tr))
    print(f"  {C:>6} | {train_acc[C]:.3f}")

# D.2
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
val_mean, val_std = {}, {}
print("D.2  C | validation accuracy (mean ± std) | per fold")
for C in C_VALUES:
    scores = cross_val_score(SVC(C=C, kernel=KERNEL), X_tr, y_tr, cv=cv,
                             scoring="accuracy")
    val_mean[C], val_std[C] = scores.mean(), scores.std()
    print(f"  {C:>6} | {val_mean[C]:.3f} ± {val_std[C]:.3f} | {np.round(scores, 3)}")

# D.3
best = max(val_mean.values())
best_C = min(C for C in C_VALUES if np.isclose(val_mean[C], best))
final = SVC(C=best_C, kernel=KERNEL).fit(X_tr, y_tr)
y_pred = final.predict(X_te)
test_acc = accuracy_score(y_te, y_pred)
print(f"D.3  selected C = {best_C}, test accuracy = {test_acc:.3f}")
print("Confusion matrix (rows = actual, cols = predicted):")
print(confusion_matrix(y_te, y_pred))
print(f"Comparison for C = {best_C}:")
print(f"  D.1 training accuracy        {train_acc[best_C]:.3f}")
print(f"  D.2 validation accuracy      {val_mean[best_C]:.3f} ± {val_std[best_C]:.3f}")
print(f"  D.3 test accuracy            {test_acc:.3f}")
