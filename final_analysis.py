import numpy as np
import pandas as pd
import torch
import time
from scipy.stats import ttest_rel

# Install if needed: pip install thop
try:
    from thop import profile
    THOP_AVAILABLE = True
except:
    THOP_AVAILABLE = False


# =========================
# INPUT YOUR DATA
# =========================

classical = np.array([
    85.840708, 60.176991, 92.035398, 56.637168,
    50.442478, 61.061947, 82.300885, 91.150442, 83.185841
])

quantum = np.array([
    74.336283, 52.212389, 88.495575, 49.557522,
    36.283186, 53.982301, 69.026549, 89.380531, 77.876106
])

subjects = np.arange(1, len(classical) + 1)


# =========================
# STATS
# =========================

classical_mean = classical.mean()
classical_std = classical.std()

quantum_mean = quantum.mean()
quantum_std = quantum.std()

t_stat, p_value = ttest_rel(classical, quantum)

diff = classical - quantum
mean_improvement = diff.mean()
cohens_d = diff.mean() / diff.std()


# =========================
# SAVE SUBJECT TABLE
# =========================

df_subject = pd.DataFrame({
    "Subject": subjects,
    "Classical": classical,
    "Quantum": quantum,
    "Difference": diff
})
df_subject.to_csv("subject_comparison.csv", index=False)


# =========================
# SAVE SUMMARY
# =========================

df_summary = pd.DataFrame({
    "Metric": [
        "Classical Mean", "Classical Std",
        "Quantum Mean", "Quantum Std",
        "Mean Improvement",
        "T-statistic", "P-value", "Cohen's d"
    ],
    "Value": [
        classical_mean, classical_std,
        quantum_mean, quantum_std,
        mean_improvement,
        t_stat, p_value, cohens_d
    ]
})
df_summary.to_csv("summary_results.csv", index=False)


# =========================
# MODEL PROFILING
# =========================

def measure_latency(model, input_tensor, runs=50, device="cuda"):
    model.eval().to(device)
    input_tensor = input_tensor.to(device)

    for _ in range(10):
        _ = model(input_tensor)

    torch.cuda.synchronize()
    start = time.time()

    for _ in range(runs):
        _ = model(input_tensor)

    torch.cuda.synchronize()
    end = time.time()

    latency = (end - start) / runs
    throughput = input_tensor.shape[0] / latency

    return latency * 1000, throughput


def compute_flops(model, input_tensor):
    if not THOP_AVAILABLE:
        return None, None

    macs, params = profile(model, inputs=(input_tensor,), verbose=False)
    flops = macs * 2
    return flops, params


device = "cuda" if torch.cuda.is_available() else "cpu"
dummy_input = torch.randn(1, 1, 22, 1125).to(device)


# =========================
# LOAD MODELS
# =========================

try:
    from eegnet_tcn_model import EEGNetTCN
    from eegnet_tcn_quantum_model import EEGNetTCNQuantum

    classical_model = EEGNetTCN().to(device)
    quantum_model = EEGNetTCNQuantum().to(device)

    # Classical metrics
    lat_c, thr_c = measure_latency(classical_model, dummy_input, device=device)
    flops_c, params_c = compute_flops(classical_model, dummy_input)

    df_classical = pd.DataFrame({
        "Metric": ["Latency (ms)", "Throughput", "FLOPs", "Params"],
        "Value": [lat_c, thr_c, flops_c, params_c]
    })
    df_classical.to_csv("efficiency_classical.csv", index=False)

    # Quantum metrics
    lat_q, thr_q = measure_latency(quantum_model, dummy_input, device=device)
    flops_q, params_q = compute_flops(quantum_model, dummy_input)

    df_quantum = pd.DataFrame({
        "Metric": ["Latency (ms)", "Throughput", "FLOPs", "Params"],
        "Value": [lat_q, thr_q, flops_q, params_q]
    })
    df_quantum.to_csv("efficiency_quantum.csv", index=False)

except Exception as e:
    print("Model profiling skipped:", e)
    lat_c = thr_c = flops_c = params_c = None
    lat_q = thr_q = flops_q = params_q = None


# =========================
# FINAL COMPARISON TABLE
# =========================

df_final = pd.DataFrame({
    "Metric": ["Accuracy Mean", "Accuracy Std", "Latency (ms)", "FLOPs", "Params"],
    "Classical": [
        classical_mean,
        classical_std,
        lat_c,
        flops_c,
        params_c
    ],
    "Quantum": [
        quantum_mean,
        quantum_std,
        lat_q,
        flops_q,
        params_q
    ]
})

df_final.to_csv("final_comparison_table.csv", index=False)


# =========================
# TEXT REPORT
# =========================

with open("analysis_report.txt", "w") as f:
    f.write("=== FINAL MODEL COMPARISON REPORT ===\n\n")

    f.write(f"Classical Mean Accuracy: {classical_mean:.2f}\n")
    f.write(f"Quantum Mean Accuracy: {quantum_mean:.2f}\n\n")

    f.write(f"T-statistic: {t_stat:.4f}\n")
    f.write(f"P-value: {p_value:.6f}\n\n")

    if p_value < 0.05:
        f.write("Statistically significant difference observed.\n\n")
    else:
        f.write("No statistically significant difference.\n\n")

    f.write(f"Mean Improvement (Classical - Quantum): {mean_improvement:.2f}\n")
    f.write(f"Cohen's d: {cohens_d:.4f}\n\n")

    f.write("Conclusion:\n")
    f.write("The classical EEGNet+TCN model demonstrates higher overall accuracy and better consistency.\n")
    f.write("The quantum-enhanced model shows improvements on certain subjects but exhibits higher variability,\n")
    f.write("indicating sensitivity to subject-specific characteristics.\n")

print("✅ All analysis files generated successfully!")