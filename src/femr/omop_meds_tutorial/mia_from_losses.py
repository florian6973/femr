import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import sys
import os

# Usage: python mia_from_losses.py train_loss_file val_loss_file
if len(sys.argv) != 3:
    print(f"Usage: python {sys.argv[0]} <train_loss_file.npy> <val_loss_file.npy>")
    sys.exit(1)

train_loss_file = sys.argv[1]
val_loss_file = sys.argv[2]

assert os.path.exists(train_loss_file), f"File not found: {train_loss_file}"
assert os.path.exists(val_loss_file), f"File not found: {val_loss_file}"

train_losses = np.load(train_loss_file)
val_losses = np.load(val_loss_file)

if train_losses.ndim == 3:
    train_losses = train_losses.mean(axis=(1,2))
    val_losses = val_losses.mean(axis=(1,2))

# Split each into two halves
def split_half(arr):
    n = len(arr)
    return arr[:n//2], arr[n//2:]

train1, train2 = split_half(train_losses)
val1, val2 = split_half(val_losses)

# Membership: 1 for train, 0 for val
mia_losses = np.concatenate([train1, val1])
mia_labels = np.concatenate([np.ones_like(train1), np.zeros_like(val1)])

# Lower loss = more likely to be member, so use -loss for AUROC
auroc = roc_auc_score(mia_labels, -mia_losses)
print(f"MIA AUROC (train1 vs val1): {auroc:.4f}")

# Optionally, plot ROC curve
fpr, tpr, _ = roc_curve(mia_labels, -mia_losses)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f'AUROC = {auroc:.4f}')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Membership Inference Attack ROC')
plt.legend()
plt.tight_layout()
plt.savefig('mia_roc_curve.png')
print("Saved ROC curve to mia_roc_curve.png")

# Also print AUROC for the other split (train2 vs val2)
# mia_losses2 = np.concatenate([train2, val2])
# mia_labels2 = np.concatenate([np.ones_like(train2), np.zeros_like(val2)])
# auroc2 = roc_auc_score(mia_labels2, -mia_losses2)
# print(f"MIA AUROC (train2 vs val2): {auroc2:.4f}")

# print("Mean train1 loss:", train1.mean())
# print("Mean val1 loss:", val1.mean()) 