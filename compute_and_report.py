import pandas as pd
import numpy as np

# -----------------------------
# FILE PATHS (UPDATED)
# -----------------------------
q1_path = "FINAL_RESULTS/final_results_run1_quantum.csv"
q2_path = "FINAL_RESULTS/final_results_run2_quantum.csv"

c1_path = "FINAL_RESULTS/final_results_run1_classical.csv"
c2_path = "FINAL_RESULTS/final_results_run2_classical.csv"


# -----------------------------
# LOAD DATA
# -----------------------------
q1 = pd.read_csv(q1_path)
q2 = pd.read_csv(q2_path)

c1 = pd.read_csv(c1_path)
c2 = pd.read_csv(c2_path)


# -----------------------------
# MERGE FUNCTION
# -----------------------------
def merge_runs(df1, df2):

    merged = df1.copy()

    merged = merged.rename(columns={
        "accuracy": "acc_run1",
        "params": "params_run1",
        "latency_ms": "lat_run1"
    })

    merged["acc_run2"] = df2["accuracy"]
    merged["params_run2"] = df2["params"]
    merged["lat_run2"] = df2["latency_ms"]

    merged["acc_mean"] = merged[["acc_run1", "acc_run2"]].mean(axis=1)
    merged["acc_std"] = merged[["acc_run1", "acc_run2"]].std(axis=1)

    merged["params_mean"] = merged[["params_run1", "params_run2"]].mean(axis=1)
    merged["lat_mean"] = merged[["lat_run1", "lat_run2"]].mean(axis=1)

    return merged


# -----------------------------
# MERGE BOTH
# -----------------------------
q_merged = merge_runs(q1, q2)
c_merged = merge_runs(c1, c2)

q_merged.to_csv("FINAL_RESULTS/quantum_merged.csv", index=False)
c_merged.to_csv("FINAL_RESULTS/classical_merged.csv", index=False)


# -----------------------------
# OVERALL METRICS
# -----------------------------
def summarize(df):
    return {
        "acc_mean": df["acc_mean"].mean(),
        "acc_std": df["acc_mean"].std(),
        "params_mean": df["params_mean"].mean(),
        "lat_mean": df["lat_mean"].mean()
    }

q_summary = summarize(q_merged)
c_summary = summarize(c_merged)


# -----------------------------
# PARAM REDUCTION
# -----------------------------
param_reduction = (
    (c_summary["params_mean"] - q_summary["params_mean"])
    / c_summary["params_mean"]
) * 100


# -----------------------------
# COMPARISON TABLE
# -----------------------------
comparison = pd.DataFrame({
    "metric": ["accuracy", "params", "latency_ms"],
    "classical": [
        c_summary["acc_mean"],
        c_summary["params_mean"],
        c_summary["lat_mean"]
    ],
    "quantum": [
        q_summary["acc_mean"],
        q_summary["params_mean"],
        q_summary["lat_mean"]
    ]
})

comparison.to_csv("FINAL_RESULTS/comparison_table.csv", index=False)


# -----------------------------
# SUMMARY TEXT
# -----------------------------
with open("FINAL_RESULTS/summary.txt", "w") as f:

    f.write("===== FINAL RESULTS SUMMARY =====\n\n")

    f.write("CLASSICAL MODEL\n")
    f.write(f"Mean Accuracy: {c_summary['acc_mean']:.2f}\n")
    f.write(f"Std Accuracy: {c_summary['acc_std']:.2f}\n")
    f.write(f"Params: {c_summary['params_mean']:.0f}\n")
    f.write(f"Latency: {c_summary['lat_mean']:.2f} ms\n\n")

    f.write("QUANTUM MODEL\n")
    f.write(f"Mean Accuracy: {q_summary['acc_mean']:.2f}\n")
    f.write(f"Std Accuracy: {q_summary['acc_std']:.2f}\n")
    f.write(f"Params: {q_summary['params_mean']:.0f}\n")
    f.write(f"Latency: {q_summary['lat_mean']:.2f} ms\n\n")

    f.write("COMPARISON\n")
    f.write(f"Parameter Reduction: {param_reduction:.2f}%\n")
    f.write(f"Accuracy Difference: {(q_summary['acc_mean'] - c_summary['acc_mean']):.2f}\n")


print("\nAll derived results saved in FINAL_RESULTS/")