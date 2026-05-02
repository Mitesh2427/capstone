import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# LOAD MATRIX
# -----------------------------
cm = np.load("results_subject3_target93/confusion_matrix.npy")

classes = ["left_hand", "right_hand", "feet", "tongue"]

# -----------------------------
# NORMALIZE (better visibility)
# -----------------------------
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

# -----------------------------
# PLOT
# -----------------------------
plt.figure(figsize=(6,5))

sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".2f",
    cmap="Blues",          # 🔥 clean high contrast
    xticklabels=classes,
    yticklabels=classes,
    cbar=True,
    linewidths=0.5,
    linecolor='black'
)

plt.xlabel("Predicted", fontsize=12)
plt.ylabel("True", fontsize=12)
plt.title("Confusion Matrix - Hybrid (Subject 3)", fontsize=13)

plt.xticks(rotation=45)
plt.yticks(rotation=0)

plt.tight_layout()

# -----------------------------
# SAVE
# -----------------------------
plt.savefig("results_subject3_target93/confusion_matrix.png", dpi=300)
plt.close()

print("Saved: confusion_matrix.png")