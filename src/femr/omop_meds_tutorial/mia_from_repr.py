import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import sys
import os
from sklearn.ensemble import RandomForestClassifier

# Usage: python mia_from_repr.py train_repr_file.npy val_repr_file.npy
if len(sys.argv) != 3:
    print(f"Usage: python {sys.argv[0]} <train_repr_file.npy> <val_repr_file.npy>")
    sys.exit(1)

train_repr_file = sys.argv[1]
val_repr_file = sys.argv[2]

assert os.path.exists(train_repr_file), f"File not found: {train_repr_file}"
assert os.path.exists(val_repr_file), f"File not found: {val_repr_file}"

print(
    "Loading representations from", train_repr_file, "and", val_repr_file
)
train_repr = np.load(train_repr_file)
val_repr = np.load(val_repr_file)

# Split each into 80% train, 20% test
n_train = int(0.8 * len(train_repr))
n_val = int(0.8 * len(val_repr))

train_repr_train = train_repr[:n_train]
train_repr_test = train_repr[n_train:]
val_repr_train = val_repr[:n_val]
val_repr_test = val_repr[n_val:]

# Membership: 1 for train, 0 for val
X_train = np.concatenate([train_repr_train, val_repr_train])
y_train = np.concatenate([np.ones(len(train_repr_train)), np.zeros(len(val_repr_train))])
X_test = np.concatenate([train_repr_test, val_repr_test])
y_test = np.concatenate([np.ones(len(train_repr_test)), np.zeros(len(val_repr_test))])

print("Training classifier...")
# Train classifier (logistic regression)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
# clf = RandomForestClassifier(n_estimators=100)
# clf.fit(X_train, y_train)

# Check membership on train set
probs = clf.predict_proba(X_train)[:, 1]
auroc = roc_auc_score(y_train, probs)
print(f"MIA AUROC (membership prediction on train set): {auroc:.4f}")

# Predict membership on test set
probs = clf.predict_proba(X_test)[:, 1]
auroc = roc_auc_score(y_test, probs)
print(f"MIA AUROC (membership prediction on held-out 20%): {auroc:.4f}")

# Optionally, plot ROC curve
fpr, tpr, _ = roc_curve(y_test, probs)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f'AUROC = {auroc:.4f}')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Membership Inference Attack ROC (Representations)')
plt.legend()
plt.tight_layout()
plt.savefig('mia_repr_roc_curve_.png')
print("Saved ROC curve to mia_repr_roc_curve.png") 