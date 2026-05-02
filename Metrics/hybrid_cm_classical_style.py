import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# LOAD MATRIX
# -----------------------------
cm = np.load("Metrics/confusion_subject_3_quantum.npy")

classes = ["left_hand", "right_hand", "feet", "tongue"]

# -----------------------------
# PLOT
# -----------------------------
fig, ax = plt.subplots(figsize=(6, 5))

im = ax.imshow(cm, cmap="viridis")

# -----------------------------
# TICKS
# -----------------------------
ax.set_xticks(np.arange(len(classes)))
ax.set_yticks(np.arange(len(classes)))

ax.set_xticklabels(classes, rotation=45)
ax.set_yticklabels(classes)

ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")

# -----------------------------
# GRID (ONLY AXIS GRID, NOT CELL GRID)
# -----------------------------
ax.grid(True, which='major', color='white', linestyle='-', linewidth=1, alpha=0.6)

# -----------------------------
# ANNOTATE VALUES
# -----------------------------
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(
            j, i, str(cm[i, j]),
            ha="center", va="center",
            color="white" if cm[i, j] < cm.max()/2 else "black",
            fontsize=11
        )

# -----------------------------
# COLOR BAR
# -----------------------------
fig.colorbar(im)

# -----------------------------
# TITLE
# -----------------------------
plt.title("Confusion Matrix - Hybrid (Subject 3)")

plt.tight_layout()

# -----------------------------
# SAVE
# -----------------------------
plt.savefig("Metrics/confusion_subject_3_hybrid_final.png", dpi=300)
plt.close()

print("Saved: Metrics/confusion_subject_3_hybrid_final.png")