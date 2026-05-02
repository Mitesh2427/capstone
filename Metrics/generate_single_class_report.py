import numpy as np
import matplotlib.pyplot as plt

classes = ["left_hand", "right_hand", "feet", "tongue"]

# Replace with your actual values
p_c = [0.86, 1.00, 0.94, 0.92]
r_c = [1.00, 0.93, 0.97, 0.79]
f_c = [0.92, 0.96, 0.95, 0.85]

p_q = [0.90, 1.00, 1.00, 0.85]
r_q = [0.97, 0.87, 0.93, 0.97]
f_q = [0.93, 0.93, 0.96, 0.90]

x = np.arange(len(classes))
width = 0.22

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

colors = {
    "precision": "#4C72B0",
    "recall": "#DD8452",
    "f1": "#55A868"
}

# -------- Classical --------
axes[0].bar(x - width, p_c, width, color=colors["precision"])
axes[0].bar(x, r_c, width, color=colors["recall"])
axes[0].bar(x + width, f_c, width, color=colors["f1"])

axes[0].set_title("Classical Model", fontsize=12)
axes[0].set_xticks(x)
axes[0].set_xticklabels(classes, rotation=25)
axes[0].set_ylabel("Score")
axes[0].grid(axis='y', linestyle='--', alpha=0.4)

# -------- Hybrid --------
axes[1].bar(x - width, p_q, width, color=colors["precision"])
axes[1].bar(x, r_q, width, color=colors["recall"])
axes[1].bar(x + width, f_q, width, color=colors["f1"])

axes[1].set_title("Hybrid Model", fontsize=12)
axes[1].set_xticks(x)
axes[1].set_xticklabels(classes, rotation=25)
axes[1].grid(axis='y', linestyle='--', alpha=0.4)

# -------- SINGLE CLEAN LEGEND --------
handles = [
    plt.Rectangle((0,0),1,1,color=colors["precision"]),
    plt.Rectangle((0,0),1,1,color=colors["recall"]),
    plt.Rectangle((0,0),1,1,color=colors["f1"])
]

fig.legend(
    ["Precision", "Recall", "F1-score"],
    loc="upper right",
    bbox_to_anchor=(0.98, 0.98),  # 🔥 top-right of full figure
    frameon=False,
    fontsize=10
)

# -------- TITLE --------
fig.suptitle("Class-wise Performance Comparison (Subject 3)", fontsize=13)

# -------- LAYOUT FIX --------
plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.savefig("Metrics/final_paper_plot.png", dpi=300)
plt.show()