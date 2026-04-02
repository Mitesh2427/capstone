import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

# =========================
# INPUT DATA
# =========================

classical = np.array([
    85.840708,
    60.176991,
    92.035398,
    56.637168,
    50.442478,
    61.061947,
    82.300885,
    91.150442,
    83.185841
])

quantum = np.array([
    74.336283,
    52.212389,
    88.495575,
    49.557522,
    36.283186,
    53.982301,
    69.026549,
    89.380531,
    77.876106
])

subjects = np.arange(1, len(classical) + 1)

# =========================
# BASIC STATS
# =========================

classical_mean = classical.mean()
classical_std = classical.std()

quantum_mean = quantum.mean()
quantum_std = quantum.std()

# =========================
# PAIRED T-TEST
# =========================

t_stat, p_value = ttest_rel(classical, quantum)

# =========================
# IMPROVEMENT + EFFECT SIZE
# =========================

diff = classical - quantum
mean_improvement = diff.mean()
cohens_d = diff.mean() / diff.std()

# =========================
# SAVE SUBJECT-WISE TABLE
# =========================

df = pd.DataFrame({
    "Subject": subjects,
    "Classical": classical,
    "Quantum": quantum,
    "Difference (Classical - Quantum)": diff
})

df.to_csv("subject_comparison.csv", index=False)

# =========================
# SAVE SUMMARY TABLE
# =========================

summary = pd.DataFrame({
    "Metric": [
        "Classical Mean",
        "Classical Std",
        "Quantum Mean",
        "Quantum Std",
        "Mean Improvement",
        "T-statistic",
        "P-value",
        "Cohen's d"
    ],
    "Value": [
        classical_mean,
        classical_std,
        quantum_mean,
        quantum_std,
        mean_improvement,
        t_stat,
        p_value,
        cohens_d
    ]
})

summary.to_csv("summary_results.csv", index=False)

# =========================
# SAVE TEXT REPORT
# =========================

with open("analysis_report.txt", "w") as f:
    f.write("=== MODEL COMPARISON REPORT ===\n\n")

    f.write("Classical Model:\n")
    f.write(f"Mean Accuracy: {classical_mean:.2f}\n")
    f.write(f"Std Dev: {classical_std:.2f}\n\n")

    f.write("Quantum Model:\n")
    f.write(f"Mean Accuracy: {quantum_mean:.2f}\n")
    f.write(f"Std Dev: {quantum_std:.2f}\n\n")

    f.write("Paired T-Test:\n")
    f.write(f"T-statistic: {t_stat:.4f}\n")
    f.write(f"P-value: {p_value:.6f}\n\n")

    if p_value < 0.05:
        f.write("Result: Statistically significant difference (p < 0.05)\n\n")
    else:
        f.write("Result: No statistically significant difference (p > 0.05)\n\n")

    f.write(f"Mean Improvement (Classical - Quantum): {mean_improvement:.2f}\n")
    f.write(f"Cohen's d: {cohens_d:.4f}\n\n")

    f.write("Interpretation:\n")
    f.write("- Positive improvement → Classical performs better\n")
    f.write("- Higher Cohen's d → stronger effect size\n")

# =========================
# PRINT OUTPUT
# =========================

print("\n=== RESULTS ===")
print(summary)

print("\nFiles saved:")
print(" - subject_comparison.csv")
print(" - summary_results.csv")
print(" - analysis_report.txt")